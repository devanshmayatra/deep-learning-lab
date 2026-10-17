# Deep Learning Lab

My hands-on journey through neural networks, deep learning, and computer vision—following Andrej Karpathy’s **Neural Networks: Zero to Hero** series, Michigan Online’s computer-vision lectures, and my own experiments.

I implement components, check their behavior, train models, and document what the results actually show.

The project now includes **MiniNN**, a small educational neural-network library built on PyTorch tensors and autograd.

## Repository Overview

| Folder | Focus |
|---|---|
| `micrograd/` | Automatic differentiation and neural-network fundamentals |
| `makemore/` | Character-level language modeling and neural-network training |
| `DeepVisionLab/` | MLP, CNN, and residual-network experiments on CIFAR-10 |
| `DeepVisionLab/mininn/` | Reusable layers, containers, optimizer, training utilities, and checkpoint helpers |

### Micrograd

Exploring computation graphs, the chain rule, backpropagation, and neural networks through the micrograd material.

Files:
- `value.py`
- `value.ipynb`
- `neural_net.ipynb`
- `index.ipynb`

### Makemore

Implementations and experiments following the makemore series, covering character-level modeling, embeddings, MLPs, initialization, activations, normalization, and backpropagation.

Notebooks:
- `makemore_1.ipynb`
- `makemore_2.ipynb`
- `makemore_3.ipynb`
- `makemore_4.ipynb`

### DeepVisionLab

Applying neural-network fundamentals to image classification.

Notebooks:
- `mlp_baseline.ipynb` — MLP baseline.
- `cnn_baseline.ipynb` — convolution, pooling, augmentation, and baseline CNN experiments.
- `residual_cnn.ipynb` — residual connections, BatchNorm, projection shortcuts, performance checks, and regularization experiments.

## What “From Scratch” Means

For the vision experiments, I implement layer operations, model composition, and Adam updates using PyTorch tensors.

PyTorch provides automatic differentiation and CPU/GPU computation. MiniNN does not implement its own tensor engine or autograd system.

Convolution was first implemented with explicit loops, then rewritten using `unfold` and matrix multiplication. The custom implementation is intended for understanding and experimentation rather than replacing optimized production kernels.

## MiniNN

MiniNN is the first reusable library milestone of this project.

| Module | Contents |
|---|---|
| `layers.py` | `Linear`, `Conv2D`, `ReLU`, `MaxPool2D`, `Flatten`, `BatchNorm2D`, `GlobalAvgPool2D`, `Dropout` |
| `containers.py` | `Sequential`, `ResidualBlock` |
| `optim.py` | Custom `Adam` optimizer |
| `training.py` | `train_one_epoch`, `evaluate` |
| `checkpoint.py` | `save_checkpoint`, `load_checkpoint` |
| `__init__.py` | Public exports |

### Implemented Features

- Convolution with configurable stride and zero padding.
- Max pooling and global average pooling.
- He initialization for convolutional layers feeding ReLU.
- Batch normalization with separate training and evaluation behavior.
- Inverted dropout with configurable probability.
- Identity and learned projection shortcuts.
- Recursive train/eval switching through nested containers.
- Explicit device and random-generator arguments for parameter initialization.
- Named model state, including BatchNorm running statistics.
- State-key and tensor-shape checks during loading.
- Adam with gradient clearing, parameter updates, and optimizer-state restoration.
- Sample-weighted loss aggregation and accuracy evaluation.
- Model-only and model-plus-optimizer checkpoints.

MiniNN is an educational implementation with a deliberately small API, not a drop-in replacement for `torch.nn`.

## Example Model

The projected residual CNN uses an identity block followed by a block that increases channels and reduces spatial size:

```python
import torch

from mininn import (
    BatchNorm2D,
    Conv2D,
    Flatten,
    Linear,
    ReLU,
    ResidualBlock,
    Sequential,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
g = torch.Generator(device=device).manual_seed(42)

model = Sequential([
    Conv2D(
        3, 16, 3, padding=1,
        device=device, generator=g
    ),
    BatchNorm2D(16, device=device),
    ReLU(),

    ResidualBlock(
        16, device=device, generator=g
    ),
    ResidualBlock(
        16, 32, stride=2,
        device=device, generator=g
    ),

    Flatten(),
    Linear(
        8192, 10, device=device, generator=g
    ),
])
```

| Stage | Output Shape |
|---|---|
| Input | `(B, 3, 32, 32)` |
| Conv → BatchNorm → ReLU | `(B, 16, 32, 32)` |
| Identity residual block | `(B, 16, 32, 32)` |
| Projected residual block | `(B, 32, 16, 16)` |
| Flatten | `(B, 8192)` |
| Linear | `(B, 10)` |

The final layer outputs **logits**, passed directly into cross-entropy.

### Residual Block Design

The current learned branch is:

```text
Conv → BatchNorm → ReLU → Conv → BatchNorm
```

The shortcut is:
- **Identity** when channels match and stride is 1.
- A learned **1 × 1 convolution** when channels or spatial dimensions change.

The branch and shortcut outputs are added elementwise. The current implementation has **no ReLU after the addition** and uses a convolution-only projection.

These are simplified residual CNNs, not implementations of a named standard architecture such as ResNet-18.

## Training with MiniNN

After constructing a fresh model and the data loaders:

```python
from mininn import (
    Adam,
    evaluate,
    save_checkpoint,
    train_one_epoch,
)

optimizer = Adam(model.parameters(), learning_rate=0.001)
best_val = float("-inf")

for epoch in range(1, 11):
    train_loss = train_one_epoch(
        model, augmented_train_loader, optimizer, device
    )
    metrics = evaluate(model, val_loader, device)

    if metrics["accuracy"] > best_val:
        best_val = metrics["accuracy"]

        save_checkpoint(
            "best_model.pt",
            model,
            optimizer=optimizer,
            epoch=epoch,
        )

    print(
        f"Epoch {epoch} | "
        f"Train loss: {train_loss:.4f} | "
        f"Val loss: {metrics['loss']:.4f} | "
        f"Val accuracy: {metrics['accuracy']:.2f}%"
    )
```

Create the optimizer once before the epoch loop so Adam’s moments and step counter persist.

`evaluate()` returns a dictionary containing:
- `"loss"` — average cross-entropy per example.
- `"accuracy"` — classification accuracy as a percentage.

## Dataset and Evaluation Setup

- **Dataset:** CIFAR-10.
- **Training:** 45,000 images.
- **Validation:** 5,000 images.
- **Test:** 10,000 images from the separate test split.
- **Split seed:** 42.
- **Batch size:** 64.
- **Optimizer:** manually implemented Adam.
- **Learning rate for recent experiments:** `0.001`.
- **Augmented training:** `RandomHorizontalFlip(p=0.5)` followed by `ToTensor()`.
- **Validation/test:** `ToTensor()` only.
- **Checkpoint criterion:** highest validation accuracy.

Training and validation preserve the same split indices. Separate dataset objects prevent training augmentation from affecting validation.

| Loader | Purpose |
|---|---|
| `augmented_train_loader` | Parameter updates using augmented images |
| `train_loader` | Evaluation on unmodified training images |
| `val_loader` | Experiment comparison and checkpoint selection |
| `test_loader` | Assessment after selecting a model |

## Experiment Results

### Initial CNN Baselines

| Experiment | Epochs Run | Best Validation Accuracy |
|---|---:|---:|
| One convolution block | 18 | 55.86% |
| Two convolution blocks | 10 | 61.56% |
| Two blocks + horizontal flips | 10 | 62.12% |
| Two blocks + horizontal flips, continued | 50 total | **66.50%** |

The selected checkpoint from the extended augmented run was **epoch 47**.

**Test accuracy: 66.24% — 6,624 correct predictions out of 10,000.**

This test result belongs specifically to the earlier two-block CNN, not the newer residual models.

### Residual Shortcut Comparison

A matched experiment compared the same learned branch with and without an identity shortcut:

- Identical starting parameter values.
- Equal parameter counts: **16,282**.
- Matched shuffle and augmentation seeds.
- Same learning rate and 10-epoch training budget.

| Variant | Best Validation Accuracy |
|---|---:|
| Plain branch: `F(x)` | **64.08%** |
| Residual branch: `F(x) + x` | 63.16% |

The shortcut did not improve the best validation result in this run. This does not establish that residual connections are generally worse; it is one experiment on a small network.

### BatchNorm and Library Milestones

| Experiment | Epochs Run | Best Validation Accuracy |
|---|---:|---:|
| Small residual CNN with BatchNorm in its branch | 10 | 64.94% |
| Fresh run using the packaged MiniNN training utilities | 10 | 64.54% |

The library run selected **epoch 10**, with validation loss **1.0259**.

### Projection, Global Average Pooling, and Dropout

These models use the larger backbone with identity and projection blocks.

| Classifier Variant | Training Accuracy at Selected Checkpoint | Validation Accuracy | Gap |
|---|---:|---:|---:|
| Flatten → Linear | **74.88%** | **65.24%** | 9.64 points |
| GlobalAvgPool → Linear | 63.44% | 62.16% | **1.28 points** |
| Flatten → Dropout(0.2) → Linear | 71.80% | 64.80% | 7.00 points |

Each experiment ran for 10 epochs and selected epoch 10 by validation accuracy.

Classifier parameter counts:
- `Linear(8192, 10)`: **81,930**.
- `Linear(32, 10)` after global average pooling: **330**.
- Dropout adds **no trainable parameters**.

**Takeaway:** GAP and dropout reduced the train–validation gap, but neither improved validation accuracy over the Flatten baseline in these runs. A smaller gap alone does not make a model better.

Except for the matched shortcut experiment, these are exploratory comparisons. Initialization, training randomness, architecture, and sometimes training duration differed. Results are from individual runs rather than multi-seed estimates.

The newer experiments have been evaluated on validation data, not given new test scores.

## Correctness and Performance Checks

Checks performed during implementation include:

- Known convolution outputs and weight/bias gradients.
- Max-pooling output and gradient routing.
- Identity-shortcut behavior with a zero residual branch.
- Gradient flow through the identity shortcut.
- Projection output shapes and optimizer updates.
- BatchNorm channel statistics and fixed evaluation buffers.
- Nested train/eval propagation.
- Model-state restoration, including parameters and buffers.
- Optimizer restoration reproducing the next update on a toy problem.
- Global average pooling’s equal gradient distribution.
- Dropout masking, expected-value scaling, and evaluation behavior.

These are targeted checks, not yet a comprehensive automated test suite.

### Convolution Benchmark

For input `(64, 3, 32, 32)`, 8 output filters, a `3 × 3` kernel, stride 1, and padding 1:

- Custom and built-in convolution outputs had a measured maximum absolute difference of **0.0** in the tested case.
- The float32 input occupied **0.75 MiB**.
- Explicitly extracted patches occupied **6.75 MiB**, or **9×** the input size.

| Implementation | Measured Time per Forward Pass |
|---|---:|
| Custom `unfold` + matrix multiplication | 0.955 ms |
| PyTorch `conv2d` | 0.039 ms |

The measured time ratio was approximately **24.53×**.

Timing included warm-up and GPU synchronization and excluded gradient tracking. This is a local, shape-specific forward-pass benchmark—not a claim about full training speed or all hardware.

## Checkpoints

MiniNN separates model state from optimizer state.

**Model state includes:**
- Weights and biases.
- BatchNorm gamma and beta.
- BatchNorm running means and variances.

**Optimizer state includes:**
- Learning rate.
- Adam’s first and second moments.
- Step counter.

```python
from mininn import load_checkpoint, save_checkpoint

# Save for training resumption
save_checkpoint(
    "training_checkpoint.pt",
    model,
    optimizer=optimizer,
    epoch=10,
)

# Load into an already-created matching model and optimizer
completed_epochs = load_checkpoint(
    "training_checkpoint.pt",
    model,
    optimizer=optimizer,
    device=device,
)
```

For inference, omit the optimizer and call `model.eval()` after loading. For training, call `model.train()`.

Important limitations:
- Recreate the same architecture and configuration before loading.
- Attach the optimizer to the matching model parameters in the same order.
- `map_location` does not move an existing model.
- Exact replay also requires random-generator and data-pipeline state, which the current helpers do not save.
- The older `cnn_aug_best.pt` uses the earlier parameter-list format and does not contain Adam state.
- The custom `state_dict()` returns cloned snapshots, so later training does not modify them.

## Current Implementation Scope

- `Conv2D` supports square kernels, integer stride, symmetric zero padding, and a bias.
- Grouped and dilated convolutions are not implemented.
- Max pooling currently uses no padding.
- Adam uses fixed betas `(0.9, 0.999)` and epsilon `1e-8`.
- Adam uses a shared step counter, appropriate for the current models where all parameters receive gradients each step.
- Dropout supports `0 <= p < 1`.
- Architecture settings such as dropout probability, stride, and shortcut configuration are recreated in code rather than inferred from checkpoints.

## Running the Notebooks

Dependencies include:
- Python
- PyTorch
- torchvision
- NumPy
- Matplotlib
- Jupyter or VS Code notebook support

Use a PyTorch installation compatible with your CPU or CUDA environment. Match random generators to the devices where random tensors are created.

Run notebook cells in order. Dataset paths are relative to the kernel’s working directory.

MiniNN currently lives inside `DeepVisionLab`, so run notebooks with that directory available on Python’s import path. The package is not yet distributed as an installable release.

Suggested order:

1. `micrograd/`
2. `makemore/`
3. `DeepVisionLab/mlp_baseline.ipynb`
4. `DeepVisionLab/cnn_baseline.ipynb`
5. `DeepVisionLab/residual_cnn.ipynb`

During package development, notebook autoreloading can help pick up saved changes:

```python
%load_ext autoreload
%autoreload 2
```

Create fresh model instances after changing constructors or architecture.

## Key Lessons

- Tensor shapes determine which operations and shortcuts are valid.
- A convolutional filter spans all input channels.
- Shared parameters accumulate gradients across their uses.
- Receptive-field size differs from kernel size and feature-map size.
- Identity shortcuts add no parameters; projection shortcuts do.
- BatchNorm parameters and running statistics serve different roles.
- `eval()` changes layer behavior; `no_grad()` disables gradient tracking.
- GAP removes spatial arrangement while preserving channel identities.
- Inverted dropout preserves activations in expectation.
- Optimizer history must persist across epochs and be restored for continued training.
- Equivalent mathematics can have very different runtime and memory costs.
- Controlled comparisons are needed to attribute changes to a particular component.
- Better validation performance matters more than simply minimizing the train–validation gap.

## Roadmap

- [x] Implement custom CNN layers using PyTorch tensors.
- [x] Build identity and projection residual blocks.
- [x] Implement BatchNorm, global average pooling, and dropout.
- [x] Package reusable components into MiniNN.
- [x] Remove notebook-global device and generator dependencies from library layers.
- [x] Add model and optimizer checkpoint interfaces.
- [x] Add reusable training and evaluation utilities.
- [x] Perform targeted numerical, gradient, and checkpoint checks.
- [x] Compare custom convolution with PyTorch’s built-in operation.
- [ ] Consolidate checks into an automated test suite.
- [ ] Add reproducible multi-seed experiment tooling.
- [ ] Save randomness and data-pipeline state for more complete resumption.
- [ ] Improve package installation and documentation.
- [ ] Explore additional architectures and computer-vision tasks.
- [ ] Continue recording experiments, limitations, and lessons.

## Acknowledgments

The micrograd and makemore work follows **Andrej Karpathy’s Neural Networks: Zero to Hero** series.

The computer-vision study also draws on **Michigan Online’s CNN architectures and hardware/software lectures**, alongside implementation exercises and CIFAR-10 experiments.

---

An ongoing learning repository—built one concept, implementation, and experiment at a time.