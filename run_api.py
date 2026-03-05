"""Launch the RITA FastAPI REST server.

Usage:
    python run_api.py               # default port 8000
    python run_api.py --port 8080
    python run_api.py --reload      # auto-reload on code changes (dev mode)

Docs available at:
    http://localhost:8000/docs      (Swagger UI)
    http://localhost:8000/redoc     (ReDoc)
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

parser = argparse.ArgumentParser(description="Launch RITA REST API")
parser.add_argument("--port", type=int, default=8000)
parser.add_argument("--host", default="0.0.0.0")
parser.add_argument("--reload", action="store_true", help="Enable auto-reload (dev mode)")
args = parser.parse_args()

import uvicorn

print(f"Starting RITA API at http://localhost:{args.port}")
print(f"Swagger docs: http://localhost:{args.port}/docs")
uvicorn.run(
    "rita.interfaces.rest_api:app",
    host=args.host,
    port=args.port,
    reload=args.reload,
)
