#!/usr/bin/env python3
"""
Root entry point for Python Mini AI.
Run with: python main.py
"""
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from python_mini_ai.main import main

if __name__ == "__main__":
    main()
