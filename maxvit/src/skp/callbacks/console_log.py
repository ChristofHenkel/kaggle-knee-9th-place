"""Persist rank-zero training output beside experiment checkpoints."""

from __future__ import annotations

import os
import sys
import threading
import traceback
from datetime import datetime, timezone
from typing import TextIO

from lightning.pytorch import Callback


class _TeeStream:
    def __init__(self, console: TextIO, log_file: TextIO, lock: threading.Lock):
        self.console = console
        self.log_file = log_file
        self.lock = lock

    def write(self, text: str) -> int:
        with self.lock:
            console_result = self.console.write(text)
            self.log_file.write(text)
            self.log_file.flush()
        return console_result

    def flush(self) -> None:
        with self.lock:
            self.console.flush()
            self.log_file.flush()

    def isatty(self) -> bool:
        return self.console.isatty()

    def fileno(self) -> int:
        return self.console.fileno()

    @property
    def encoding(self) -> str | None:
        return self.console.encoding


class ConsoleLogCallback(Callback):
    """Mirror stdout and stderr to a line-buffered file on global rank zero."""

    def __init__(self, log_path: str):
        super().__init__()
        self.log_path = log_path
        self._log_file: TextIO | None = None
        self._stdout: TextIO | None = None
        self._stderr: TextIO | None = None

    def on_fit_start(self, trainer, pl_module) -> None:
        del pl_module
        if not trainer.is_global_zero or self._log_file is not None:
            return
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        self._log_file = open(self.log_path, "a", buffering=1, encoding="utf-8")
        timestamp = datetime.now(timezone.utc).isoformat()
        self._log_file.write(f"\n===== training started {timestamp} =====\n")
        self._stdout, self._stderr = sys.stdout, sys.stderr
        lock = threading.Lock()
        sys.stdout = _TeeStream(self._stdout, self._log_file, lock)
        sys.stderr = _TeeStream(self._stderr, self._log_file, lock)
        print(f"Mirroring console output to {self.log_path}")

    def on_fit_end(self, trainer, pl_module) -> None:
        del trainer, pl_module
        self._stop()

    def on_exception(self, trainer, pl_module, exception: BaseException) -> None:
        del trainer, pl_module
        if self._log_file is not None:
            traceback.print_exception(exception, file=sys.stderr)
        self._stop()

    def teardown(self, trainer, pl_module, stage: str) -> None:
        del trainer, pl_module, stage
        self._stop()

    def _stop(self) -> None:
        if self._log_file is None:
            return
        timestamp = datetime.now(timezone.utc).isoformat()
        print(f"===== training finished {timestamp} =====")
        if self._stdout is not None:
            sys.stdout = self._stdout
        if self._stderr is not None:
            sys.stderr = self._stderr
        self._log_file.close()
        self._log_file = None
        self._stdout = None
        self._stderr = None
