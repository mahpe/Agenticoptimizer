"""Backward-compatible entry point; prefer ``python methods/agentic_optimizer.py`` from the repo root."""

from agentic_optimizer import main

if __name__ == "__main__":
    raise SystemExit(main())
