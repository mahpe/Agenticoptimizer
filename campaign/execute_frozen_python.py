#!/usr/bin/env python3
"""Execute a Python runner from one captured byte buffer with a bound digest."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import sys


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(f"usage: {sys.argv[0]} RUNNER.py [ARGS ...]")
    runner = Path(sys.argv[1]).resolve()
    payload = runner.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    os.environ["HERMES_EXECUTED_RUNNER_SHA256"] = digest
    sys.argv = [str(runner), *sys.argv[2:]]
    namespace = {
        "__name__": "__main__",
        "__file__": str(runner),
        "__package__": None,
        "__cached__": None,
    }
    exec(compile(payload, str(runner), "exec"), namespace, namespace)


if __name__ == "__main__":
    main()
