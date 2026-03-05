"""Launch the RITA Streamlit UI.

Usage:
    python run_ui.py           # default port 8501
    python run_ui.py --port 8502
"""
import argparse
import socket
import subprocess
import sys
import os

def is_port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) != 0

def find_free_port(start: int = 8501) -> int:
    for port in range(start, start + 10):
        if is_port_free(port):
            return port
    return start  # fallback — streamlit will report the error

parser = argparse.ArgumentParser(description="Launch RITA Streamlit UI")
parser.add_argument("--port", type=int, default=None, help="Port to run on (default: auto-select from 8501)")
args = parser.parse_args()

port = args.port if args.port else find_free_port(8501)
app_path = os.path.join("src", "rita", "interfaces", "streamlit_app.py")

print(f"Starting RITA UI at http://localhost:{port}")
subprocess.run(
    [sys.executable, "-m", "streamlit", "run", app_path, f"--server.port={port}"],
    check=True,
)
