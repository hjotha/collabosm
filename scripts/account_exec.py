#!/usr/bin/env python3
"""Run a CLI script with the selected account home before any CLI imports."""
import os
import runpy
import sys

home, script, *args = sys.argv[1:]
os.environ['HOME'] = home
os.environ['USERPROFILE'] = home
sys.argv = [script] + args
sys.path.insert(0, os.path.dirname(script))
runpy.run_path(script, run_name='__main__')
