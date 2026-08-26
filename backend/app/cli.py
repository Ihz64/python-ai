import sys
import argparse
import uvicorn
import asyncio

from backend.app.config import settings
from backend.app.database.init_db import init_db


def main():
    parser = argparse.ArgumentParser(description="CodeMind AI Platform CLI")
    subparsers = parser.add_subparsers(dest="command")

    start_parser = subparsers.add_parser("start", help="Start CodeMind AI backend server")
    start_parser.add_argument("--host", default="0.0.0.0", help="Host address")
    start_parser.add_argument("--port", type=int, default=8000, help="Port number")
    start_parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    subparsers.add_parser("initdb", help="Initialize or migrate database tables")
    subparsers.add_parser("test", help="Run test suite")

    args = parser.parse_args()

    if args.command == "start":
        print(f"Starting {settings.APP_NAME} server at http://{args.host}:{args.port}")
        uvicorn.run("backend.app.main:app", host=args.host, port=args.port, reload=args.reload)
    elif args.command == "initdb":
        print("Initializing database tables...")
        asyncio.run(init_db())
        print("Database initialized successfully.")
    elif args.command == "test":
        import pytest
        sys.exit(pytest.main(["backend/tests"]))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
