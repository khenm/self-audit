# PyTorch Training Template

A clean, general-purpose PyTorch DDP/FSDP training template for deep learning research. Built around composable Hydra configs and the `_target_` instantiation pattern — every component (model, loss, optimizer, scheduler, dataloader, logger) is swappable via config without touching Python code.

## Features

- **Composable Hydra configs** — swap model, data, optimizer, and loss from the CLI with no code changes
- **DDP + FSDP** — select strategy via `distributed.strategy: ddp|fsdp` in config
- **`self.where` scheduler convention** — all schedulers take a single `float ∈ [0,1]` representing training progress; linear warmup → cosine decay out of the box
- **Gradient accumulation** — `accum_steps` chunks batches and uses `model.no_sync()` to suppress redundant all-reduce
- **AMP** — bfloat16/float16 via `torch.amp.autocast`; preprocessing runs in fp32
- **Robust checkpointing** — backup-swap write pattern survives preemptions; auto-resumes from `checkpoint.pt`
- **Per-module gradient clipping** — glob-pattern groups with full-coverage validation
- **Glob-based module freezing** — patterns like `"*encoder*"` lock params and patch `.train()` permanently
- **Rank-0 only I/O** — all saves, summaries, and TensorBoard writes are gated on rank 0

## Installation

Requires Python 3.10+ and [`uv`](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/username/pytorch-template-code.git
cd pytorch-template-code
uv sync
```

## Project Structure

```text
pytorch-template-code/
├── configs/
│   ├── train.yaml           # Primary entry point — composes all defaults
│   ├── data/
│   │   └── imagenet.yaml
│   ├── model/
│   │   └── resnet18.yaml
│   ├── optim/
│   │   ├── adamw.yaml       # AdamW + linear warmup → cosine LR
│   │   └── sgd.yaml
│   └── loss/
│       └── focal_loss.yaml
├── scripts/
│   └── train.py             # Entrypoint
├── src/
│   ├── datasets/
│   │   └── dummy.py         # DummyDataset for smoke-testing
│   ├── losses/
│   │   ├── focal.py         # FocalLoss
│   │   └── generic.py       # CrossEntropy, MSE wrappers
│   ├── models/
│   │   └── model.py         # ResNet18Model
│   ├── utils/
│   │   ├── checkpoint.py    # CheckpointSaver + robust_torch_save
│   │   ├── dist.py          # Distributed rank helpers
│   │   ├── env.py           # Environment variable setup
│   │   ├── freeze.py        # Glob-pattern module freezing
│   │   ├── fsdp.py          # FSDP wrapping + mixed precision policy
│   │   ├── general.py       # AverageMeter, copy_data_to_device, seeds, …
│   │   ├── gradient_clip.py # Per-module GradientClipper
│   │   ├── logging.py       # Rank-aware logging setup
│   │   ├── optimizer.py     # OptimizerWrapper + construct_optimizers
│   │   └── tensorboard_writer.py
│   └── trainer.py           # Core DDP/FSDP Trainer
├── notebooks/
│   └── 01_eda_exploration.ipynb
├── pyproject.toml
└── uv.lock
```

## Usage

### Single-GPU

```bash
uv run python scripts/train.py --config train
```

### Multi-GPU DDP

```bash
uv run torchrun --nproc_per_node=8 scripts/train.py --config train
```

### Multi-GPU FSDP

```bash
uv run torchrun --nproc_per_node=8 scripts/train.py --config train distributed.strategy=fsdp
```

### Override config groups from the CLI

```bash
# Switch optimizer
uv run torchrun --nproc_per_node=8 scripts/train.py --config train optim=sgd

# Override individual values
uv run torchrun --nproc_per_node=8 scripts/train.py --config train max_epochs=50 optim.optimizer.lr=1e-3
```

## Configuration

`configs/train.yaml` is the single entry point. It composes defaults from four groups:

```yaml
defaults:
  - data: imagenet
  - model: resnet18
  - optim: adamw
  - loss: focal_loss
  - _self_
```

Each group file is self-contained and fully `_target_`-driven — Hydra instantiates objects directly. To add a new variant, drop a new file in the relevant group directory.

### Key top-level config keys

| Key | Description |
|-----|-------------|
| `exp_name` | Experiment name; used in log/checkpoint paths |
| `max_epochs` | Total training epochs |
| `accum_steps` | Gradient accumulation steps |
| `val_epoch_freq` | Run validation every N epochs |
| `distributed.strategy` | `ddp` (default) or `fsdp` |
| `checkpoint.resume_checkpoint_path` | Explicit resume path; if null, auto-discovers `checkpoint.pt` |
| `optim.frozen_module_names` | List of glob patterns for modules to freeze |

## Extending the Template

### Adding a new model

1. Create `src/models/my_model.py` with a plain `nn.Module` — no registry decorator needed.
2. Add a config file `configs/model/my_model.yaml`:

```yaml
_target_: src.models.my_model.MyModel
hidden_dim: 512
num_layers: 6
```

3. Launch with `--config train model=my_model`.

### Adding a new dataset

1. Create `src/datasets/my_dataset.py` as a `torch.utils.data.Dataset`.
2. Add `configs/data/my_dataset.yaml` with `_target_` pointing to your class.
3. The Trainer wraps it in a `DistributedSampler` automatically.

### Adding a new loss

1. Create `src/losses/my_loss.py` as an `nn.Module` with a standard `forward(preds, targets)`.
2. Add `configs/loss/my_loss.yaml` with `_target_`.

### Customising the forward pass / loss computation

Override `_model_inputs` and `_compute_loss` in a `Trainer` subclass:

```python
class MyTrainer(Trainer):
    def _model_inputs(self, batch):
        return {"images": batch["image"], "mask": batch["mask"]}

    def _compute_loss(self, outputs, batch):
        loss = self.loss_fn(outputs["logits"], batch["label"])
        return {"loss": loss, "aux_loss": outputs["aux"]}
```

---

## Research Paper README Template

If you are adapting this repository for a paper release, use the following template [PAPER.md](PAPER.md) to ensure clarity, rigor, and reproducibility.
