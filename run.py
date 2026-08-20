"""Backward-compatible entry point; prefer ``python agentic_optimizer.py``."""

from agentic_optimizer import main

if __name__ == "__main__":
    raise SystemExit(main())
