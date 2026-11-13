"""Local Colab profiles. Credentials stay in separate CLI homes."""
import json
import os
import tempfile
import threading
import uuid


class Accounts:
    def __init__(self, state_dir, original_home):
        self.lock = threading.RLock()
        self.directory = os.path.abspath(state_dir)
        self.path = os.path.join(self.directory, 'accounts.json')
        os.makedirs(self.directory, mode=0o700, exist_ok=True)
        if os.path.exists(self.path):
            with open(self.path, encoding='utf-8') as f:
                self.data = json.load(f)
        else:
            self.data = {'active': 'default', 'accounts': [
                {'id': 'default', 'label': 'Atual', 'home': os.path.abspath(original_home)}]}
            self._save()

    def _save(self):
        fd, temp = tempfile.mkstemp(dir=self.directory, prefix='.accounts-')
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp, self.path)
        finally:
            if os.path.exists(temp):
                os.unlink(temp)

    def snapshot(self):
        with self.lock:
            return {'active_account': self.data['active'], 'accounts': [
                {'id': a['id'], 'label': a['label']} for a in self.data['accounts']]}

    def home(self, account_id=None):
        with self.lock:
            if account_id is None:
                account_id = self.data['active']
            return next((a['home'] for a in self.data['accounts'] if a['id'] == account_id), None)

    def create(self, label):
        if not isinstance(label, str) or not label.strip() or len(label.strip()) > 80:
            return None
        with self.lock:
            account_id = uuid.uuid4().hex
            home = os.path.join(self.directory, 'account-homes', account_id)
            os.makedirs(home, mode=0o700)
            self.data['accounts'].append({'id': account_id, 'label': label.strip(), 'home': home})
            try:
                self._save()
            except Exception:
                self.data['accounts'].pop()
                raise
            return account_id

    def switch(self, account_id):
        with self.lock:
            if self.home(account_id) is None:
                return False
            old = self.data['active']
            self.data['active'] = account_id
            try:
                self._save()
            except Exception:
                self.data['active'] = old
                raise
            return True
