"""Authenticated image and asynchronous video generation on a Colab VM."""
import base64
import binascii
import hmac
import io
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PROFILES = {
    # kind, width, height, frames, steps, fps
    "sd-turbo": ("image", 512, 512, 1, 1, 0),
    "z-image": ("image", 1024, 1024, 1, 9, 0),
    "wan21": ("video", 480, 288, 33, 24, 16),
    "wan22": ("video", 768, 432, 49, 30, 24),
    "framepack": ("video", 640, 384, 91, 25, 30),
    "pika": ("video", 848, 480, None, 6, 24),
}
FRAMEPACK_BASE = ("hunyuanvideo-community/HunyuanVideo", "e8c2aaa66fe3742a32c11a6766aecbf07c56e773")
FRAMEPACK_VISION = ("lllyasviel/flux_redux_bfl", "45b801affc54ff2af4e5daf1b282e0921901db87")
IMAGE_ROUTE = "/v1/images/generations"
VIDEO_ROUTE = "/v1/videos"
VIDEO_PATH = re.compile(r"/v1/videos/(video_[0-9a-f]{32})(/content)?")
MAX_BODY = 12 * 1024 ** 2
MAX_VIDEOS = 8


def download(profile, model_dir):
    from huggingface_hub import snapshot_download

    patterns = ["*.json", "*.txt", "*.model", "*.safetensors"]

    def fetch(repo, revision, directory, allow, ignore=None):
        return snapshot_download(repo_id=repo, revision=revision, local_dir=directory,
                                 allow_patterns=allow, ignore_patterns=ignore)

    ignore = ["*.fp16.safetensors", "*.bf16.safetensors"]
    if profile == "sd-turbo":
        patterns.remove("*.safetensors")
        patterns.append("*.fp16.safetensors")
        ignore = None
    fetch(os.environ["MODEL_REPO"], os.environ["MODEL_REVISION"], model_dir, patterns, ignore)
    if profile == "framepack":
        fetch(*FRAMEPACK_BASE, model_dir / "base", patterns, ["transformer/*", *ignore])
        fetch(*FRAMEPACK_VISION, model_dir / "vision",
              ["feature_extractor/*", "image_encoder/*.json", "image_encoder/*.safetensors"])


def load_pipeline(profile, model_dir):
    import torch
    from diffusers import AutoencoderKLWan, DiffusionPipeline, WanPipeline

    if profile.startswith("wan"):
        from transformers import BitsAndBytesConfig, UMT5EncoderModel

        # The full UMT5 checkpoint alone exceeds the T4's VRAM.
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16
        text_encoder = UMT5EncoderModel.from_pretrained(
            model_dir / "text_encoder", dtype=torch.bfloat16, device_map={"": 0},
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        )
        vae = AutoencoderKLWan.from_pretrained(model_dir, subfolder="vae", torch_dtype=torch.float32)
        vae.enable_tiling()
        return WanPipeline.from_pretrained(model_dir, vae=vae, text_encoder=text_encoder,
                                           torch_dtype=dtype, use_safetensors=True).to("cuda")
    if profile == "framepack":
        from diffusers import HunyuanVideoFramepackPipeline, HunyuanVideoFramepackTransformer3DModel
        from transformers import SiglipImageProcessor, SiglipVisionModel

        transformer = HunyuanVideoFramepackTransformer3DModel.from_pretrained(
            model_dir, torch_dtype=torch.bfloat16)
        image_encoder = SiglipVisionModel.from_pretrained(
            model_dir / "vision", subfolder="image_encoder", dtype=torch.float16)
        feature_extractor = SiglipImageProcessor.from_pretrained(
            model_dir / "vision", subfolder="feature_extractor")
        pipe = HunyuanVideoFramepackPipeline.from_pretrained(
            model_dir / "base", transformer=transformer, image_encoder=image_encoder,
            feature_extractor=feature_extractor, torch_dtype=torch.float16)
        pipe.enable_model_cpu_offload()
        pipe.vae.enable_tiling()
        return pipe
    options = {"variant": "fp16"} if profile == "sd-turbo" else {}
    dtype = torch.float16 if profile == "sd-turbo" else torch.bfloat16
    return DiffusionPipeline.from_pretrained(
        model_dir, torch_dtype=dtype, use_safetensors=True, **options).to("cuda")


def input_image(value):
    from PIL import Image

    match = re.fullmatch(r"data:image/(?:png|jpeg|webp);base64,([A-Za-z0-9+/=]+)", value or "")
    if not match:
        raise ValueError("image must be a PNG, JPEG or WebP data URL")
    try:
        with Image.open(io.BytesIO(base64.b64decode(match[1], validate=True))) as image:
            if image.format not in ("PNG", "JPEG", "WEBP") or image.width * image.height > 16_000_000:
                raise ValueError("image format or dimensions are unsupported")
            return image.convert("RGB")
    except (binascii.Error, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("invalid input image") from exc


class Media:
    def __init__(self, profile, model, recipe, gpu, generate, output):
        self.profile, self.model, self.recipe, self.gpu = profile, model, recipe, gpu
        self.kind, self.width, self.height, self.frames, self.steps, self.fps = PROFILES[profile]
        self.generate, self.output = generate, Path(output)
        self.lock = threading.Lock()
        self.jobs = {}
        self.in_flight = False
        self.started = self.last_request = time.time()

    def status(self):
        with self.lock:
            return {
                "model": self.model, "recipe": self.recipe, "service": "collabosm",
                "uptime_s": int(time.time() - self.started),
                "media": {"kind": self.kind, "image_input": self.profile == "framepack",
                          "width": self.width, "height": self.height},
                "dialects": [IMAGE_ROUTE if self.kind == "image" else VIDEO_ROUTE],
                "activity": {"in_flight": int(self.in_flight), "last_request_at": self.last_request},
                "machine": {"gpu": self.gpu},
            }

    def reserve(self, video):
        with self.lock:
            if self.in_flight:
                return None
            self.in_flight = True
            self.last_request = time.time()
            if not video:
                return True
            if len(self.jobs) >= MAX_VIDEOS:
                oldest = next(iter(self.jobs))
                (self.output / (oldest + ".mp4")).unlink(missing_ok=True)
                del self.jobs[oldest]
            job_id = "video_" + secrets.token_hex(16)
            job = {"id": job_id, "object": "video", "status": "in_progress",
                   "model": self.model, "created_at": int(self.last_request)}
            self.jobs[job_id] = job
            return job.copy()

    def finish(self, job_id=None, **fields):
        with self.lock:
            if job_id:
                self.jobs[job_id].update(fields)
            self.in_flight = False
            self.last_request = time.time()

    def run_video(self, job_id, prompt, seed, image):
        path = self.output / (job_id + ".mp4")
        started = time.monotonic()
        try:
            self.generate(prompt, seed, image, path)
            if path.stat().st_size <= 0:
                raise RuntimeError("empty MP4")
            self.finish(job_id, status="completed", seed=seed,
                        generation_seconds=round(time.monotonic() - started, 3))
        except Exception:
            traceback.print_exc()
            path.unlink(missing_ok=True)
            self.finish(job_id, status="failed", error={"message": "Video generation failed. See serve.log."})


class Handler(BaseHTTPRequestHandler):
    def reply(self, code, body):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def error(self, code, message):
        return self.reply(code, {"error": {"message": message}})

    def authorize(self):
        provided = self.headers.get("Authorization", "").encode("utf-8")
        if hmac.compare_digest(provided, self.server.authorization):
            return True
        self.error(401, "Unauthorized")
        return False

    def do_GET(self):
        if self.path == "/health":
            return self.reply(200, {"status": "ok"})
        if not self.authorize():
            return
        media = self.server.media
        if self.path == "/v1/models":
            return self.reply(200, {"object": "list", "data": [
                {"id": media.model, "object": "model", "owned_by": "collabosm"}]})
        if self.path == "/v1/status":
            return self.reply(200, media.status())
        route = VIDEO_PATH.fullmatch(self.path)
        if not route:
            return self.error(404, "Unknown endpoint")
        job_id, content = route.groups()
        with media.lock:
            job = media.jobs.get(job_id, {}).copy()
            if content and job.get("status") == "completed":
                # Open under the same lock as eviction. The reader owns the open file.
                artifact = (media.output / (job_id + ".mp4")).open("rb")
            else:
                artifact = None
        if not job:
            return self.error(404, "Unknown video")
        if not content:
            return self.reply(200, job)
        if artifact is None:
            return self.error(409, "Video is not ready")
        with artifact:
            self.video_content(artifact)

    def video_content(self, artifact):
        total = os.fstat(artifact.fileno()).st_size
        start, end, code = 0, total - 1, 200
        requested = self.headers.get("Range")
        if requested:
            match = re.fullmatch(r"bytes=([0-9]{0,18})-([0-9]{0,18})", requested)
            if match and match[1]:
                start = int(match[1])
                end = min(end, int(match[2])) if match[2] else end
            elif match and match[2]:
                start = max(0, total - int(match[2]))
            else:
                start = total
            if start > end:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{total}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            code = 206
        remaining = end - start + 1
        self.send_response(code)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Content-Disposition", 'inline; filename="collabosm.mp4"')
        self.send_header("Content-Length", str(remaining))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "private, no-store")
        if code == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{total}")
        self.end_headers()
        artifact.seek(start)
        while remaining:
            chunk = artifact.read(min(remaining, 64 * 1024))
            self.wfile.write(chunk)
            remaining -= len(chunk)

    def do_POST(self):
        if not self.authorize():
            return
        media = self.server.media
        video = media.kind == "video"
        if self.path != (VIDEO_ROUTE if video else IMAGE_ROUTE):
            return self.error(404, "Endpoint unavailable for the active recipe")
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_BODY:
                return self.error(413, "Request must be between 1 byte and 12 MiB")
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict):
                raise ValueError("JSON object required")
            prompt = body.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 4000:
                raise ValueError("prompt must contain 1–4000 characters")
            if body.get("model", media.model) != media.model:
                raise ValueError("model does not match the active recipe")
            if body.get("n", 1) != 1 or body.get("response_format", "b64_json") != "b64_json":
                raise ValueError("Only one output is supported; images use b64_json")
            if "size" in body and body["size"] != f"{media.width}x{media.height}":
                raise ValueError("size is fixed by the active recipe")
            seed = body.get("seed", secrets.randbelow(2 ** 31))
            if type(seed) is not int or not 0 <= seed < 2 ** 31:
                raise ValueError("seed must be an integer from 0 to 2147483647")
            image = None
            if media.profile == "framepack" or "image" in body:
                if media.profile != "framepack" or not isinstance(body.get("image"), str):
                    raise ValueError("FramePack requires an image data URL; other recipes take text only")
                image = input_image(body["image"])
        except (UnicodeError, ValueError) as exc:
            return self.error(400, str(exc))

        job = media.reserve(video)
        if job is None:
            return self.error(409, "A generation is already in progress")
        if video:
            threading.Thread(target=media.run_video, args=(job["id"], prompt, seed, image),
                             daemon=True).start()
            return self.reply(202, job)
        started = time.monotonic()
        try:
            png = media.generate(prompt, seed, None, None)
        except Exception:
            traceback.print_exc()
            return self.error(500, "Image generation failed. See serve.log.")
        finally:
            media.finish()
        return self.reply(200, {"created": int(time.time()), "seed": seed,
                                "generation_seconds": round(time.monotonic() - started, 3),
                                "data": [{"b64_json": base64.b64encode(png).decode("ascii")}]})

    def log_message(self, fmt, *args):
        pass


def generator(profile, model_dir):
    if profile == "pika":
        def pika(prompt, seed, image, path):
            with tempfile.TemporaryDirectory(prefix="pika-", dir=path.parent) as tmp:
                prompt_path = Path(tmp) / "prompt.txt"
                prompt_path.write_text(prompt, encoding="utf-8")
                video = Path(tmp) / "video.mp4"
                subprocess.run(
                    [str(Path(sys.executable).with_name("pika-h3-pro")), "--task", "t2va",
                     "--prompt-file", str(prompt_path), "--model-repo", os.environ["MODEL_REPO"],
                     "--revision", os.environ["MODEL_REVISION"], "--resolution", "480p",
                     "--aspect", "16:9", "--duration", "5", "--seed", str(seed),
                     "--output", str(video)],
                    env=dict(os.environ, HF_HOME=str(model_dir)), check=True, timeout=2400,
                )
                shutil.move(video, path)
        return pika

    import torch
    from diffusers.utils import export_to_video

    pipe = load_pipeline(profile, model_dir)
    kind, width, height, frames, steps, fps = PROFILES[profile]

    def generate(prompt, seed, image, path):
        options = {"prompt": prompt, "generator": torch.Generator().manual_seed(seed),
                   "width": width, "height": height, "num_inference_steps": steps}
        with torch.inference_mode():
            if kind == "image":
                output = pipe(**options, guidance_scale=0).images[0]
                buffer = io.BytesIO()
                output.save(buffer, "PNG")
                return buffer.getvalue()
            if profile == "framepack":
                options.update(image=image, sampling_type="inverted_anti_drifting", guidance_scale=9.0)
            else:
                options.update(guidance_scale=5.0, max_sequence_length=128)
            output = pipe(**options, num_frames=frames).frames[0]
            export_to_video(output, str(path), fps=fps)

    return generate


def main():
    profile = os.environ["MEDIA_PIPELINE"]
    model_dir = Path(os.environ["MODEL_DIR"])
    if sys.argv[1:] == ["--download"]:
        download(profile, model_dir)
        return
    get_gpu = ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"]
    name, memory = subprocess.check_output(get_gpu, text=True, timeout=10).splitlines()[0].rsplit(",", 1)
    print(f"{os.environ['MODEL_ID']} on {name}: loading model", flush=True)
    if profile == "pika":
        model_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="collabosm-media-", dir="/content") as output:
        media = Media(profile, os.environ["MODEL_ID"], os.environ["RECIPE"],
                      {"name": name.strip(), "total_mib": int(memory)}, generator(profile, model_dir), output)
        key = Path("/content/api-key.txt").read_text().strip()
        if not key:
            raise RuntimeError("API key is empty")
        with ThreadingHTTPServer(("127.0.0.1", int(os.environ["PORT"])), Handler) as server:
            server.authorization = ("Bearer " + key).encode()
            server.media = media
            print("Media server ready", flush=True)
            server.serve_forever()


if __name__ == "__main__":
    main()
