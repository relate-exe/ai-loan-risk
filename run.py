"""Start backend and frontend together:  python run.py"""
import subprocess
import sys
import time

backend = subprocess.Popen([sys.executable, "-m", "uvicorn", "backend.main:app", "--port", "8000"])
time.sleep(3)
frontend = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "frontend/app.py"])
try:
    frontend.wait()
except KeyboardInterrupt:
    pass
finally:
    backend.terminate()
    frontend.terminate()
