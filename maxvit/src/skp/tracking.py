"""Fail-closed MLflow tracking policy for experiment launches."""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from mlflow.tracking import MlflowClient


LOCAL_TRACKING_SCHEMES = {"", "file", "sqlite"}


def format_tracking_uri_for_log(tracking_uri: str | None) -> str | None:
    """Return a credential-redacted tracking URI for console messages."""
    if tracking_uri is None:
        return None
    parts = urlsplit(tracking_uri)
    if parts.username is None and parts.password is None:
        return tracking_uri
    hostname = parts.hostname or ""
    if parts.port is not None:
        hostname = f"{hostname}:{parts.port}"
    return urlunsplit(
        (parts.scheme, hostname, parts.path, parts.query, parts.fragment)
    )


def is_local_tracking_uri(tracking_uri: str | None) -> bool:
    """Return whether MLflow would use a local file or SQLite backend."""
    if tracking_uri is None or not tracking_uri.strip():
        return True
    return urlsplit(tracking_uri).scheme.lower() in LOCAL_TRACKING_SCHEMES


def _local_fallback_uri(save_dir: str | Path) -> str:
    return (Path(save_dir).resolve() / "mlruns").as_uri()


def validate_mlflow_tracking(
    tracking_uri: str | None,
    *,
    allow_local: bool,
    save_dir: str | Path,
    timeout_seconds: int = 5,
    dry_run: bool = False,
) -> str:
    """Resolve a tracking URI and prove it is usable before training starts.

    Remote backends are queried with the official MLflow client. Missing and
    explicitly local backends are rejected unless ``allow_local`` is true.
    When a configured remote backend is unavailable, local fallback also
    requires the same per-launch opt-in.
    """
    if dry_run:
        return (Path(save_dir).resolve() / "mlflow_dry_run").as_uri()

    normalized = tracking_uri.strip() if tracking_uri else None
    if is_local_tracking_uri(normalized):
        if not allow_local:
            rendered = format_tracking_uri_for_log(normalized) or "<unset>"
            raise RuntimeError(
                "Refusing to launch without a reachable non-local MLflow "
                f"backend. Configured tracking URI: {rendered}. Missing, file, "
                "and SQLite tracking are local. Start/configure the MLflow "
                "server, or rerun with --allow_local_mlflow only after local "
                "tracking has been explicitly authorized for this launch."
            )
        resolved = normalized or _local_fallback_uri(save_dir)
        warnings.warn(
            "Local MLflow tracking was explicitly enabled for this launch: "
            f"{format_tracking_uri_for_log(resolved)}",
            RuntimeWarning,
            stacklevel=2,
        )
        return resolved

    previous_timeout = os.environ.get("MLFLOW_HTTP_REQUEST_TIMEOUT")
    os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = str(timeout_seconds)
    try:
        MlflowClient(tracking_uri=normalized).search_experiments(max_results=1)
    except Exception as error:
        rendered = format_tracking_uri_for_log(normalized)
        if not allow_local:
            raise RuntimeError(
                f"MLflow preflight failed for {rendered}. No training run was "
                "started. Restore the tracking server/backend, or rerun with "
                "--allow_local_mlflow only after local tracking has been "
                "explicitly authorized for this launch."
            ) from error
        fallback = _local_fallback_uri(save_dir)
        warnings.warn(
            f"MLflow preflight failed for {rendered}; explicit local tracking "
            f"authorization is active, so this launch will use {fallback}.",
            RuntimeWarning,
            stacklevel=2,
        )
        return fallback
    finally:
        if previous_timeout is None:
            os.environ.pop("MLFLOW_HTTP_REQUEST_TIMEOUT", None)
        else:
            os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = previous_timeout
    return normalized
