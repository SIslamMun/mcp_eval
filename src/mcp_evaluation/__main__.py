#!/usr/bin/env python3
"""
Main entry point for mcp_evaluation module.
Allows running: python -m mcp_evaluation instead of python -m src.mcp_evaluation.cli
"""

from .cli import main

if __name__ == "__main__":
    main()
