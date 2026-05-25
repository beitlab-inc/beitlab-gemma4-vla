# Multi-Machine Training Guide

Train Gemma4VLA on different hardware (Jetson Thor, cloud GPUs) with automatic hardware detection and configuration.

## Quick Start

### Jetson Thor (Local)

Default behavior: auto-detects Jetson Thor and applies conservative defaults (batch_size=2, grad_accum_steps=4, gradient_checkpointing).

```bash
# Uses environment defaults (data from $PWD/data/metaworld_demos, output to $PWD/outputs)
uv run python -m robots.metaworld.scripts.train --mlflow --mlflow-system-metrics
```

### Cloud GPU (e.g., AWS Lambda, Paperspace)

Set environment variables before training:

```bash
export GEMMA4VLA_DATA_DIR=/mnt/data/metaworld_demos  # or S3 path
export GEMMA4VLA_OUTPUT_DIR=/mnt/outputs
export MLFLOW_TRACKING_URI=http://mlflow-server:5000
export GEMMA4VLA_MLFLOW=true

uv run python -m robots.metaworld.scripts.train --mlflow
```

### Desktop GPU (A100, H100, L40S)

Auto-detects larger VRAM and applies larger batch sizes:

```bash
# Machine-specific defaults apply: batch_size=16, grad_accum_steps=1, no checkpointing
uv run python -m robots.metaworld.scripts.train --mlflow
```

## Environment Variables

All environment variables are **optional** — code detects hardware and applies sensible defaults.

```bash
# Data and output paths (local or remote URLs)
export GEMMA4VLA_DATA_DIR=/path/to/data
export GEMMA4VLA_OUTPUT_DIR=/path/to/outputs

# MLflow configuration
export MLFLOW_TRACKING_URI=http://mlflow-server:5000
export GEMMA4VLA_MLFLOW=true

# Optional: Disable automatic defaults
export GEMMA4VLA_BATCH_SIZE=8
export GEMMA4VLA_GRAD_ACCUM=2
export GEMMA4VLA_PRECISION=fp16
```

## CLI Overrides

All CLI args override environment variables:

```bash
# Override batch size detected on your machine
uv run python -m robots.metaworld.scripts.train \
  --batch-size 4 \
  --max-steps 1000 \
  --mlflow

# Use different data directory than env var
uv run python -m robots.metaworld.scripts.train \
  --data-dir /mnt/custom_data \
  --output-dir /mnt/custom_outputs
```

## Hardware Detection

The code auto-detects and logs:

```
Machine Type: jetson_thor | cloud_a100 | cloud_h100 | cloud_l40s | desktop_gpu
Device: NVIDIA Thor | NVIDIA A100-40GB | ...
Available VRAM: 12.0 GB
Recommended batch size: 2
Recommended grad accum steps: 4
Recommended precision: bf16
```

### Detected Machine Types

| Device | Batch Size | Grad Accum | Gradient Ckpt | Precision |
|--------|-----------|-----------|--------------|-----------|
| Jetson Thor | 2 | 4 | yes | fp16/bf16 |
| Cloud < 16GB VRAM | 4 | 2 | yes | fp16/bf16 |
| Cloud 16-40GB VRAM | 8 | 1 | no | bf16 |
| Cloud > 40GB VRAM | 16 | 1 | no | bf16 |
| Desktop GPU (generic) | 8 | 1 | no | fp16/bf16 |

## Example: Training on Multiple Machines

### 1. Jetson Thor (Local Collection + Training)

```bash
# Data collection
MUJOCO_GL=egl uv run python -m robots.metaworld.scripts.collect_data \
  --episodes 100 \
  --output-dir data/metaworld_demos \
  --rerun-mode connect \
  --mlflow

# Training (auto-detects Jetson Thor)
export GEMMA4VLA_MLFLOW=true
export MLFLOW_TRACKING_URI=http://mlflow-server:5000

uv run python -m robots.metaworld.scripts.train \
  --mlflow \
  --mlflow-system-metrics
```

### 2. AWS Lambda GPU (Transfer Data + Train)

```bash
# Setup on Lambda instance
export GEMMA4VLA_DATA_DIR=/mnt/efs/data
export GEMMA4VLA_OUTPUT_DIR=/mnt/efs/outputs
export MLFLOW_TRACKING_URI=http://mlflow-vpn.internal:5000
export GEMMA4VLA_MLFLOW=true

# Copy data from S3 / EFS
aws s3 cp s3://my-bucket/metaworld_demos /mnt/efs/data --recursive

# Train (auto-detects GPU type)
uv run python -m robots.metaworld.scripts.train \
  --mlflow \
  --mlflow-system-metrics \
  --max-steps 5000
```

### 3. Paperspace (or other cloud GPU provider)

```bash
# Paperspace provides $PS_WORKSPACE and $PS_PROJECT_DIR
export GEMMA4VLA_DATA_DIR=$PS_WORKSPACE/data
export GEMMA4VLA_OUTPUT_DIR=$PS_WORKSPACE/outputs
export MLFLOW_TRACKING_URI=http://your-mlflow-server:5000
export GEMMA4VLA_MLFLOW=true

# Copy data
gsutil -m cp -r gs://your-bucket/metaworld_demos $PS_WORKSPACE/data

# Train
uv run python -m robots.metaworld.scripts.train --mlflow
```

## Monitoring Training Across Machines

All MLflow runs log:

- **Machine type** (jetson_thor, cloud_a100, etc.)
- **VRAM available** at start
- **Batch size** actually used
- **GPU metrics** (via pynvml): utilization, memory, temperature, power
- **Training loss**, validation loss
- **Throughput** (steps/sec)

View all runs in MLflow:

```bash
mlflow ui --backend-store-uri sqlite:////path/to/mlruns.db
```

Then open http://localhost:5000 and filter by `machine_type`.

## Troubleshooting

**Issue: "CUDA out of memory" on Jetson**
- The code detected Jetson but batch_size was overridden to 4+
- Let auto-detection work: remove `--batch-size` CLI arg

**Issue: Slow data loading on cloud**
- Set `num_workers=0` in DataLoader (multiprocessing overhead on cloud is high)
- Pre-download data to local NVMe instead of EBS/network storage

**Issue: MLflow server unreachable**
- Check `export MLFLOW_TRACKING_URI=...` is correct
- Verify network connectivity: `curl http://mlflow-server:5000/health`
- Fall back to local: `export GEMMA4VLA_MLFLOW=false`

## Distributed Training (Multi-GPU)

DDP is wired into `gemma4_vla.train.train()` — when `RANK`, `WORLD_SIZE`, and
`LOCAL_RANK` are set (i.e. you launched via `torchrun`), the trainer:

- Initialises NCCL via `dist.init_process_group`.
- Rebuilds the train/val DataLoaders with a `DistributedSampler` (shuffled
  on train, deterministic on val, with `set_epoch` each loop).
- Wraps the model in `DistributedDataParallel`.
- Disables `device_map="auto"` on the backbone so each rank holds its own
  copy on a single device.
- Restricts logging, MLflow, JSONL metrics, and checkpoint saves to rank 0
  (other ranks join `dist.barrier()` during eval/save).
- Seeds with `seed + rank` so each shard sees different augmentation noise.

Launch with `torchrun`:

```bash
torchrun --nproc_per_node=4 -m robots.metaworld.scripts.train \
  --config robots/metaworld/configs/metaworld_push.yaml \
  --data-dir data/metaworld_demos \
  --mlflow
```

Or the core entry point directly:

```bash
torchrun --nproc_per_node=4 -m gemma4_vla.train \
  --config robots/metaworld/configs/metaworld_push.yaml
```

Effective batch size scales linearly with `WORLD_SIZE`. `grad_accum_steps`
multiplies on top: effective = `batch_size × grad_accum_steps × WORLD_SIZE`.

`machine_config.py` is not yet aware of multi-GPU and still recommends the
per-rank defaults — review the picked `batch_size` when running DDP on
heterogeneous hardware.
