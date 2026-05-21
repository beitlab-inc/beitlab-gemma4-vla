"""Optional local observability helpers for Rerun and MLflow."""

import json
import os
from pathlib import Path

import numpy as np


def ensure_parent(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def sanitize_params(params):
    clean = {}
    for key, value in params.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            clean[key] = value
        else:
            clean[key] = json.dumps(value)
    return clean


class RerunLogger:
    """Thin wrapper around the optional rerun-sdk package."""

    def __init__(self, mode="off", app_id="gemma4_vla", save_path=None, connect_url=None):
        self.mode = mode
        self.app_id = app_id
        self.save_path = save_path
        self.connect_url = connect_url
        self.rr = None

    @property
    def enabled(self):
        return self.rr is not None

    def start(self):
        if self.mode == "off":
            return self

        try:
            import rerun as rr
        except ImportError as exc:
            raise RuntimeError(
                "Rerun logging requires `rerun-sdk`. Install it with "
                "`uv sync --extra observability` or `uv add rerun-sdk`."
            ) from exc

        self.rr = rr

        if self.mode == "spawn":
            rr.init(self.app_id, spawn=True)
        elif self.mode == "save":
            if not self.save_path:
                raise ValueError("--rerun-path is required when --rerun-mode save is used")
            ensure_parent(self.save_path)
            rr.init(self.app_id)
            rr.save(self.save_path)
        elif self.mode == "connect":
            rr.init(self.app_id)
            if self.connect_url:
                rr.connect_grpc(self.connect_url)
            else:
                rr.connect_grpc()
        else:
            raise ValueError(f"Unsupported rerun mode: {self.mode}")

        return self

    def set_step(self, step, episode=None):
        if not self.enabled:
            return
        self.rr.set_time("step", sequence=int(step))
        if episode is not None:
            self.rr.set_time("episode", sequence=int(episode))

    def log_image(self, path, image):
        if self.enabled and image is not None:
            self.rr.log(path, self.rr.Image(image))

    def log_vector(self, path, vector, dim_name="dim"):
        if self.enabled and vector is not None:
            arr = np.asarray(vector)
            self.rr.log(path, self.rr.Tensor(arr, dim_names=(dim_name,)))

    def log_vector_series(self, path, vector):
        """Log all dimensions as Scalars at one entity path (one time-series plot)."""
        if not self.enabled or vector is None:
            return
        arr = np.asarray(vector, dtype=np.float64).ravel()
        self.rr.log(path, self.rr.Scalars(arr))

    def log_tensor(self, path, tensor, dim_names=None):
        if self.enabled and tensor is not None:
            arr = np.asarray(tensor)
            self.rr.log(path, self.rr.Tensor(arr, dim_names=dim_names))

    def log_scalar(self, path, value):
        if self.enabled and value is not None:
            self.rr.log(path, self.rr.Scalars(float(value)))

    def log_text(self, path, text):
        if self.enabled and text is not None:
            self.rr.log(path, self.rr.TextDocument(str(text)))


class MlflowRun:
    """Optional MLflow run wrapper with no-op behavior when disabled."""

    def __init__(
        self,
        enabled=False,
        tracking_uri=None,
        experiment_name="gemma4-vla",
        run_name=None,
    ):
        self.enabled_requested = enabled
        self.tracking_uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.mlflow = None
        self.run = None

    @property
    def enabled(self):
        return self.mlflow is not None

    def start(self, params=None):
        if not self.enabled_requested:
            return self

        try:
            import mlflow
        except ImportError as exc:
            raise RuntimeError(
                "MLflow logging requires `mlflow`. Install it with "
                "`uv sync --extra observability` or `uv add mlflow`."
            ) from exc

        self.mlflow = mlflow
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        self.run = mlflow.start_run(run_name=self.run_name)
        if params:
            mlflow.log_params(sanitize_params(params))
        artifact_uri = mlflow.get_artifact_uri()
        if artifact_uri.startswith("/mlflow") or artifact_uri.startswith("file:///mlflow"):
            print(
                "[mlflow] Warning: this run uses artifact URI "
                f"{artifact_uri!r}, which points to the MLflow container filesystem. "
                "Use a new experiment after switching the server to --artifacts-destination, "
                "or reset the MLflow volumes if you do not need old runs."
            )
        return self

    def log_params(self, params):
        if self.enabled:
            self.mlflow.log_params(sanitize_params(params))

    def log_metric(self, key, value, step=None):
        if self.enabled and value is not None:
            self.mlflow.log_metric(key, float(value), step=step)

    def log_metrics(self, metrics, step=None):
        if self.enabled:
            clean = {
                key: float(value)
                for key, value in metrics.items()
                if value is not None
            }
            if clean:
                self.mlflow.log_metrics(clean, step=step)

    def log_artifact(self, path, artifact_path=None):
        if self.enabled and path and Path(path).exists():
            try:
                self.mlflow.log_artifact(path, artifact_path=artifact_path)
            except Exception as exc:
                print(
                    "[mlflow] Warning: failed to log artifact "
                    f"{path!r} to {artifact_path!r}: {exc}"
                )

    def log_dict(self, value, artifact_file):
        if self.enabled:
            self.mlflow.log_dict(value, artifact_file)

    def end(self):
        if self.enabled:
            self.mlflow.end_run()
