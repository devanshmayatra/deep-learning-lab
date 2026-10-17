# Deep Learning Lab

My hands-on journey through neural networks, deep learning, and computer vision—following Andrej Karpathy’s **Neural Networks: Zero to Hero** series, Michigan Online’s computer-vision lectures, and my own experiments.

I implement components, check their behavior, train models, and document what the results show.

The repository includes **MiniNN**, an educational neural-network library built on PyTorch tensors and autograd, alongside native PyTorch and transfer-learning experiments.

## Current Highlights

- Built custom convolution, pooling, normalization, dropout, and residual layers.
- Packaged reusable layers, optimizers, training loops, and checkpoint helpers into MiniNN.
- Implemented Adam and AdamW with optimizer-state restoration.
- Compared custom implementations against native PyTorch.
- Explored learning-rate schedules, augmentation, model depth, and classifier design.
- Reached **78.42% validation accuracy** with a custom-designed CNN trained from scratch.
- Reached **91.05% test accuracy** with an ImageNet-pretrained ResNet-18 after fine-tuning its final residual stage.

The from-scratch and pretrained results use different architectures, preprocessing, and pretraining data budgets.

## Repository Overview

| Folder | Focus |
|---|---|
| `micrograd/` | Automatic differentiation and neural-network fundamentals |
| `makemore/` | Character-level language modeling and neural-network training |
| `DeepVisionLab/` | CIFAR-10 experiments, MiniNN, native PyTorch, and transfer learning |
| `DeepVisionLab/mininn/` | Reusable educational neural-network library |

Local dataset directories `data/` and `data_1/` are used by notebooks. Dataset downloads, generated caches, and local checkpoints should be excluded from Git.

### Micrograd

Exploring computation graphs, the chain rule, backpropagation, and neural networks.

Files include:

- `value.py`
- `value.ipynb`
- `neural_net.ipynb`
- `index.ipynb`

### Makemore

Character-level modeling experiments covering embeddings, MLPs, initialization, activations, normalization, and backpropagation.

Notebooks:

- `makemore_1.ipynb`
- `makemore_2.ipynb`
- `makemore_3.ipynb`
- `makemore_4.ipynb`

### DeepVisionLab

| Notebook | Focus |
|---|---|
| `mlp_baseline.ipynb` | MLP baseline for image classification |
| `cnn_baseline.ipynb` | Custom convolution, pooling, and initial CNN experiments |
| `residual_cnn.ipynb` | Residual connections and related experiments |
| `mini_nn.ipynb` | MiniNN development |
| `mininn_test.ipynb` | MiniNN checks and experiments |
| `pytorch_baseline.ipynb` | Native CNNs, schedules, AdamW, augmentation, depth, and pooling |
| `transfer_learning.ipynb` | Frozen pretrained features and final-stage fine-tuning |

Model checkpoints currently live alongside the notebooks.

## What “From Scratch” Means

For MiniNN, I implement layer operations, model composition, and optimizer updates using PyTorch tensors.

**PyTorch provides automatic differentiation and CPU/GPU computation.** MiniNN does not implement an independent tensor engine or autograd system.

Convolution was first implemented with explicit loops, then rewritten using `unfold` and matrix multiplication.

The native PyTorch experiments use `torch.nn` layers and `torch.optim` optimizers. Transfer-learning experiments additionally use pretrained Torchvision weights.

## MiniNN

| Module | Contents |
|---|---|
| `layers.py` | `Linear`, `Conv2D`, `ReLU`, `MaxPool2D`, `Flatten`, `BatchNorm2D`, `GlobalAvgPool2D`, `Dropout` |
| `containers.py` | `Sequential`, `ResidualBlock` |
| `optim.py` | `Adam`, `AdamW` |
| `training.py` | `train_one_epoch`, `evaluate` |
| `checkpoint.py` | `save_checkpoint`, `load_checkpoint` |
| `__init__.py` | Public exports |

### Implemented Features

- Convolution with configurable stride and zero padding.
- Max pooling and global average pooling.
- He initialization for custom convolutional and linear weights.
- Batch normalization with separate training and evaluation behavior.
- Inverted dropout.
- Identity and learned projection shortcuts.
- Recursive train/eval switching through nested containers.
- Explicit device and random-generator arguments for initialization.
- Named model state, including BatchNorm running statistics.
- State-key and tensor-shape validation during loading.
- Adam moment estimates and bias correction.
- AdamW with decoupled weight decay.
- Optimizer-state saving and restoration.
- Sample-weighted loss aggregation.
- Model-only and model-plus-optimizer checkpoint helpers.

MiniNN is an educational implementation with a deliberately small API, not a drop-in replacement for `torch.nn`.

### Example Model

Run this where `mininn` is available on Python’s import path:

```python
import torch
import mininn as mini

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
g = torch.Generator(device=device).manual_seed(42)

model = mini.Sequential([
    mini.Conv2D(
        3, 16, 3, padding=1,
        device=device, generator=g,
    ),
    mini.BatchNorm2D(16, device=device),
    mini.ReLU(),

    mini.ResidualBlock(
        16, device=device, generator=g,
    ),
    mini.ResidualBlock(
        16, 32, stride=2,
        device=device, generator=g,
    ),

    mini.Flatten(),
    mini.Linear(
        8192, 10, device=device, generator=g,
    ),
])

optimizer = mini.AdamW(
    model.parameters(),
    learning_rate=0.001,
    weight_decay=0.01,
)
```

The final layer produces **raw logits**, passed directly into cross-entropy.

### Custom Residual Block

The learned branch consists of:

`Conv → BatchNorm → ReLU → Conv → BatchNorm`

The shortcut is:

- **Identity** when channels match and stride is 1.
- A learned **1 × 1 convolution** when channels or spatial dimensions change.

The outputs are added elementwise. The custom block has **no ReLU after addition** and uses a convolution-only projection.

These custom models are simplified residual CNNs, not standard ResNet-18 implementations.

## Dataset and Evaluation

- **Dataset:** CIFAR-10.
- **Classes:** 10.
- **Training:** 45,000 images.
- **Validation:** 5,000 images.
- **Test:** 10,000 images from the separate test split.
- **Split seed:** 42.
- **Batch size:** 64.
- **Checkpoint selection:** highest validation accuracy.

Training and validation preserve the same split indices. Separate dataset objects keep random training augmentation out of evaluation.

Reported training accuracies are measured in evaluation mode without random augmentation.

### Preprocessing

| Experiment family | Training | Evaluation |
|---|---|---|
| Initial baselines | `ToTensor()` | `ToTensor()` |
| Flip augmentation | Horizontal flip + `ToTensor()` | `ToTensor()` |
| Crop augmentation | `RandomCrop(32, padding=4)` + horizontal flip + `ToTensor()` | `ToTensor()` |
| Pretrained ResNet-18 | Pretrained weights’ transforms | Same pretrained transforms |

Transfer learning uses `ResNet18_Weights.DEFAULT.transforms()`: resize, center crop to 224 × 224, tensor conversion, and normalization using the pretrained weights’ expected statistics.

No random augmentation was used in the recorded transfer-learning runs. Resizing prepares the inputs for the pretrained model; it does not recover missing image detail.

Validation guides model selection. Test results are recorded after selection and are not used to choose subsequent changes.

## Experiment Results

### Initial CNN Baselines

| Experiment | Epochs | Best validation accuracy |
|---|---:|---:|
| One convolution block | 18 | 55.86% |
| Two convolution blocks | 10 | 61.56% |
| Two blocks + horizontal flips | 10 | 62.12% |
| Two blocks + horizontal flips, continued | 50 total | **66.50%** |

The extended augmented run selected **epoch 47**.

**Test accuracy: 66.24% — 6,624 / 10,000 correct.**

This test result belongs to the earlier two-block CNN.

### Identity Shortcut Comparison

A matched experiment used identical starting weights, equal parameter counts of **16,282**, and matched shuffle/augmentation seeds.

| Variant | Best validation accuracy |
|---|---:|
| Plain branch: `F(x)` | **64.08%** |
| Residual branch: `F(x) + x` | 63.16% |

The identity shortcut did not improve this small model’s best validation result in this run.

### BatchNorm and Library Milestones

| Experiment | Epochs | Best validation accuracy |
|---|---:|---:|
| Small residual CNN with BatchNorm in its branch | 10 | 64.94% |
| Fresh run using packaged MiniNN training utilities | 10 | 64.54% |

### Projection, Pooling, and Dropout

These MiniNN models use an identity block followed by a projection block.

| Classifier | Training accuracy | Validation accuracy |
|---|---:|---:|
| Flatten → Linear | 74.88% | **65.24%** |
| GlobalAvgPool → Linear | 63.44% | 62.16% |
| Flatten → Dropout(0.2) → Linear | 71.80% | 64.80% |

Each run selected epoch 10.

The flatten classifier had **81,930 parameters**; the GAP classifier had **330**. Dropout added no trainable parameters.

A smaller train–validation gap did not automatically produce better validation accuracy.

### Native PyTorch, Scheduling, and AdamW

The following models use the projected 16 → 32-channel backbone.

| Experiment | Epochs | Selected epoch | Training accuracy | Validation accuracy |
|---|---:|---:|---:|---:|
| Native PyTorch + Adam | 10 | 9 | 76.01% | 68.02% |
| Adam + learning-rate schedule | 20 | 18 | 84.64% | 72.38% |
| AdamW + learning-rate schedule | 20 | 14 | 83.08% | 72.48% |
| MiniNN AdamW + crops and flips | 20 | 18 | 71.64% | 70.44% |
| Native PyTorch AdamW + crops and flips | 20 | 20 | 74.09% | **72.72%** |

The schedule used:

- Epochs 1–10: learning rate `0.001`.
- Epochs 11–20: learning rate `0.0001`.
- AdamW weight decay: `0.01`.

The paired MiniNN/PyTorch crop experiment matched per-epoch shuffle and augmentation seeds, but initial weights differed.

### Deeper CNN

An additional projection block increased channels from 32 to 64 and reduced spatial dimensions.

| Stage | Output shape |
|---|---|
| Input | `(B, 3, 32, 32)` |
| Conv → BatchNorm → ReLU | `(B, 16, 32, 32)` |
| Identity residual block | `(B, 16, 32, 32)` |
| Projection block, stride 2 | `(B, 32, 16, 16)` |
| Projection block, stride 2 | `(B, 64, 8, 8)` |
| Flatten | `(B, 4096)` |
| Linear | `(B, 10)` |

Both classifier variants ran for 20 epochs with AdamW, crops, flips, and the epoch-11 learning-rate reduction.

| Classifier | Classifier parameters | Training accuracy | Validation accuracy |
|---|---:|---:|---:|
| Flatten → Linear(4096, 10) | 40,970 | 80.74% | **78.42%** |
| GAP → Linear(64, 10) | 650 | 77.61% | 75.92% |

Both selected epoch 20.

GAP reduced classifier parameters by approximately **63×**, with a **2.50-point** validation accuracy decrease. This parameter reduction applies to the classifier, not the entire network.

### Transfer Learning

Used **Torchvision’s standard ResNet-18**, pretrained on ImageNet.

**Stage 1 — frozen features**

- Replaced the original classifier with `Linear(512, 10)`.
- Trained only the classifier’s **5,130 parameters**.
- Kept the backbone in evaluation mode.
- Used AdamW with learning rate `0.001` and weight decay `0.01`.
- Ran five epochs and selected epoch 3.

**Stage 2 — fine-tuning**

- Started from the selected frozen-feature checkpoint.
- Unfroze the final residual stage, `layer4`.
- Used learning rate `1e-5` for that stage and `1e-4` for the classifier.
- Kept earlier stages frozen.
- Kept BatchNorm running statistics fixed.
- Allowed layer 4’s BatchNorm scale and bias to learn.
- Ran five epochs and selected fine-tuning epoch 4.

| Stage | Selected epoch within stage | Training accuracy | Validation accuracy |
|---|---:|---:|---:|
| Frozen backbone + new classifier | 3 | 87.32% | 86.58% |
| Final residual stage fine-tuned | 4 | 97.79% | **91.36%** |

### Selected Transfer Model: Test Results

| Metric | Result |
|---|---:|
| Test images | 10,000 |
| Correct predictions | 9,105 |
| Test accuracy | **91.05%** |
| Test cross-entropy | **0.2755** |

Fine-tuning improved validation accuracy by **4.78 percentage points** over the frozen-feature stage.

### Interpreting These Results

These are learning experiments, mostly individual runs rather than multi-seed estimates.

Initialization, training randomness, architecture, preprocessing, and training duration differ across some comparisons. Small differences should not be treated as reliable gains without further controlled runs.

Transfer learning also introduces ImageNet pretraining, so its results are not directly comparable under an equal training-data budget.

## Error Analysis

For the deeper CNN with **78.42% validation accuracy**:

- 80 cats were predicted as dogs.
- 59 dogs were predicted as cats.
- Cat recall was **55.84%**.
- Cat precision was **65.10%**.

Visual inspection showed low-resolution images and varied framing. These are possible contributors, not proven causes of individual predictions.

Some incorrect predictions had high softmax scores, illustrating that confidence does not guarantee correctness.

## Correctness and Performance Checks

Targeted checks include:

- Known convolution outputs and parameter gradients.
- Max-pooling output and gradient routing.
- Identity-shortcut behavior.
- Projection shapes and weight updates.
- BatchNorm statistics and fixed evaluation buffers.
- Nested train/eval propagation.
- Model-state restoration.
- Adam and AdamW restoration reproducing the next update.
- Global average pooling’s gradient distribution.
- Dropout masking, scaling, and evaluation behavior.

These are targeted checks, not yet a comprehensive automated test suite.

### Convolution Benchmark

For input `(64, 3, 32, 32)`, eight output channels, kernel size 3, stride 1, and padding 1:

- Measured maximum output difference: **0.0** in the tested case.
- Float32 input: **0.75 MiB**.
- Extracted patches: **6.75 MiB**, or **9×** the input size.

| Implementation | Time per forward pass |
|---|---:|
| Custom `unfold` + matrix multiplication | 0.955 ms |
| Native PyTorch convolution | 0.039 ms |

Measured time ratio: **24.53×**.

### Full Training-Step Benchmark

A separate benchmark included forward pass, loss calculation, backward pass, and optimizer update.

| Implementation | Time per training step |
|---|---:|
| MiniNN | 16.160 ms |
| Native PyTorch | 4.766 ms |

Native PyTorch was **3.39× faster** in this measurement.

Both benchmarks included warm-up and GPU synchronization. The training-step benchmark used synthetic inputs already on the device, excluding data loading and host-to-device transfers.

These are local, workload-specific measurements.

After copying weights and BatchNorm buffers between equivalent projected models, one forward check produced a maximum absolute difference of approximately **3.62e-5**, with matching predicted classes.

## Checkpoints

MiniNN separates model state from optimizer state.

**Model state includes:**

- Weights and biases.
- BatchNorm scale and bias.
- BatchNorm running means and variances.

**Optimizer state includes:**

- Learning rate.
- First and second moments.
- Step counter.
- Weight-decay strength for AdamW.

```python
from mininn import save_checkpoint, load_checkpoint

save_checkpoint(
    "training_checkpoint.pt",
    model,
    optimizer=optimizer,
    epoch=10,
)

completed_epochs = load_checkpoint(
    "training_checkpoint.pt",
    model,
    optimizer=optimizer,
    device=device,
)
```

Recreate the matching architecture and optimizer before loading.

### Selected Local Checkpoint Files

| File | Experiment |
|---|---|
| `cnn_aug_best.pt` | Earlier augmented two-block CNN |
| `best_projected_cnn.pt` | MiniNN projected residual CNN |
| `best_gap_cnn.pt` | Earlier MiniNN GAP experiment |
| `best_dropout_cnn.pt` | Earlier MiniNN dropout experiment |
| `best_native_scheduled.pt` | Native Adam with learning-rate schedule |
| `best_mininn_adamw_crop.pt` | MiniNN AdamW with crops |
| `best_native_adamw_crop.pt` | Native AdamW with crops |
| `best_deeper_adamw_crop.pt` | Deeper flatten-based CNN |
| `best_frozen_resnet18.pt` | Frozen-backbone transfer model |
| `best_finetuned_resnet18.pt` | Fine-tuned transfer model |

`best_model.pt` is a generic checkpoint name; inspect its metadata before assuming its architecture.

`checkpoint_helper_test.pt` and `state_roundtrip_test.pt` are check artifacts.

### Restoration Limitations

- Recent native experiment files contain model state and metadata for evaluation, not full optimizer state.
- Restoring best-epoch weights does not restore the optimizer automatically.
- Do not combine best-epoch weights with optimizer moments from a later epoch.
- Exact training replay also requires random-generator and data-pipeline state.
- Architecture settings must be recreated in code.
- MiniNN’s `state_dict()` clones tensors.
- Native PyTorch’s `state_dict()` requires `copy.deepcopy()` for an independent in-memory snapshot.
- The legacy `cnn_aug_best.pt` uses an earlier parameter-list format.

## Current MiniNN Scope

- Square convolution kernels, integer stride, symmetric zero padding, and bias.
- No grouped or dilated convolution.
- Max pooling without padding.
- Adam and AdamW with fixed betas `(0.9, 0.999)` and epsilon `1e-8`.
- Shared optimizer step counter; intermittently updated parameters would need per-parameter counters to match native behavior.
- Dropout supports `0 <= p < 1`.
- Package installation and broader automated testing remain future work.

## Running the Notebooks

Dependencies include:

- Python
- PyTorch
- torchvision
- NumPy
- Matplotlib
- Jupyter or VS Code notebook support

Use a PyTorch installation compatible with your CPU or CUDA environment.

Run cells in order. Dataset and checkpoint paths are relative to the kernel’s working directory.

MiniNN currently lives inside `DeepVisionLab/`, so that directory must be available on Python’s import path.

Suggested learning order:

1. `micrograd/`
2. `makemore/`
3. `DeepVisionLab/mlp_baseline.ipynb`
4. `DeepVisionLab/cnn_baseline.ipynb`
5. `DeepVisionLab/residual_cnn.ipynb`
6. `DeepVisionLab/mini_nn.ipynb`
7. `DeepVisionLab/mininn_test.ipynb`
8. `DeepVisionLab/pytorch_baseline.ipynb`
9. `DeepVisionLab/transfer_learning.ipynb`

During package development:

```python
%load_ext autoreload
%autoreload 2
```

Create fresh model instances after constructor or architecture changes. Always attach the optimizer to the model actually passed into the training loop.

## Planned Repository Cleanup

The paths above describe the current layout. Suggested cleanup:

| Location | Intended contents |
|---|---|
| `DeepVisionLab/mininn/` | Reusable library code |
| `DeepVisionLab/*.ipynb` | Notebooks, retained here for current import compatibility |
| `DeepVisionLab/checkpoints/` | Local model and optimizer artifacts |
| `DeepVisionLab/figures/` | Exported plots and confusion matrices |
| `DeepVisionLab/experiments.md` | Detailed experiment notes as the project grows |

Moving checkpoints requires updating notebook save/load paths.

Suggested `.gitignore` entries:

```gitignore
__pycache__/
*.py[cod]
.ipynb_checkpoints/

/data/
/data_1/

*.pt
*.pth
```

Ignore rules do not untrack files already committed. Generated files must be untracked separately if necessary.

## Key Lessons

- A convolutional filter spans all input channels.
- Shared weights accumulate gradients across their uses.
- Stride affects spatial dimensions and classifier size.
- Identity shortcuts add no parameters; projection shortcuts do.
- BatchNorm parameters and running statistics have different roles.
- `eval()` changes layer behavior; it does not disable gradients.
- GAP discards final spatial arrangement after feature extraction.
- Adam’s moments persist across updates.
- AdamW decay is separate from moment calculations but still scales with learning rate.
- Stronger augmentation can make training harder while improving generalization.
- A smaller train–validation gap does not necessarily mean a better model.
- Accuracy and cross-entropy measure different aspects of predictions.
- Freezing parameters does not freeze BatchNorm running statistics.
- Fine-tuning requires gradient tracking through the unfrozen layers.
- Pretrained features can transfer across related visual tasks.
- Controlled experiments are needed to attribute improvements.

## Roadmap

- [x] Implement custom CNN layers.
- [x] Build identity and projection residual blocks.
- [x] Implement BatchNorm, GAP, and dropout.
- [x] Package reusable components into MiniNN.
- [x] Add Adam and AdamW.
- [x] Add training, evaluation, and checkpoint utilities.
- [x] Perform targeted numerical and state-restoration checks.
- [x] Compare custom and native PyTorch performance.
- [x] Explore schedules, augmentation, depth, and pooling.
- [x] Train a classifier on frozen pretrained features.
- [x] Fine-tune a pretrained residual stage.
- [x] Evaluate the selected transfer model on held-out test data.
- [ ] Consolidate checks into an automated test suite.
- [ ] Add reproducible multi-seed experiment tooling.
- [ ] Improve package installation and documentation.
- [ ] Save RNG and data-pipeline state for more complete resumption.
- [ ] Expand into additional computer-vision tasks.

## Acknowledgments

The micrograd and makemore work follows **Andrej Karpathy’s Neural Networks: Zero to Hero** series.

Computer-vision study also draws on **Michigan Online’s Deep Learning for Computer Vision** lectures.

PyTorch and Torchvision provide tensor computation, automatic differentiation, native layers, datasets, and pretrained models.

---

An ongoing learning repository—built one concept, implementation, and experiment at a time.