# Deep Learning Lab

My hands-on journey through neural networks, deep learning, and computer vision—following Andrej Karpathy’s **Neural Networks: Zero to Hero** series and extending it with my own experiments.

I’m building components, testing my understanding, and documenting what I learn. The long-term goal is to turn these implementations into an educational deep-learning library.

## Repository Overview

| Folder | Focus |
|---|---|
| `micrograd/` | Automatic differentiation and neural-network fundamentals |
| `makemore/` | Character-level language modeling and training neural networks |
| `DeepVisionLab/` | MLP and CNN experiments for image classification |

### Micrograd

Exploring computation graphs, backpropagation, and neural networks through the micrograd material.

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

Applying neural-network fundamentals to CIFAR-10 image classification.

Files:
- `mlp_baseline.ipynb` — MLP baseline.
- `cnn_baseline.ipynb` — custom CNN implementations and experiments.
- `cnn_aug_best.pt` — saved weights and metadata for the selected augmented CNN.

## What “From Scratch” Means

In the vision experiments, I implement layer operations, model composition, and Adam updates using PyTorch tensors.

PyTorch provides automatic differentiation and CPU/GPU computation. This is not an independent tensor engine.

Convolution was first implemented using explicit loops, then rewritten using `unfold` and matrix multiplication.

## Implemented Components

- Custom `Linear`, `Conv2D`, `ReLU`, `MaxPool2D`, and `Flatten` layers.
- A `Sequential` wrapper with parameter collection and train/eval methods.
- Convolution with configurable stride and zero padding.
- Max pooling using patch extraction and maximum selection.
- He initialization for convolutional layers feeding ReLU.
- Manual Adam updates with moment estimates and bias correction.
- Training and evaluation loops.
- Horizontal-flip augmentation.
- Confusion matrices and visual inspection of misclassified images.
- Validation-based checkpoint selection and parameter saving/loading.

## CIFAR-10 CNN

```python
model = Sequential([
    Conv2D(3, 8, 3, padding=1),
    ReLU(),
    MaxPool2D(2, 2),
    Conv2D(8, 16, 3, padding=1),
    ReLU(),
    MaxPool2D(2, 2),
    Flatten(),
    Linear(1024, 10),
])
```

| Stage | Output Shape |
|---|---|
| Input | `(B, 3, 32, 32)` |
| Conv → ReLU | `(B, 8, 32, 32)` |
| MaxPool | `(B, 8, 16, 16)` |
| Conv → ReLU | `(B, 16, 16, 16)` |
| MaxPool | `(B, 16, 8, 8)` |
| Flatten | `(B, 1024)` |
| Linear | `(B, 10)` |

The final layer produces raw class scores—**logits**—which are passed directly into PyTorch’s cross-entropy loss.

## Training Setup

- **Dataset:** CIFAR-10.
- **Training split:** 45,000 images.
- **Validation split:** 5,000 images.
- **Test split:** 10,000 images.
- **Split seed:** 42.
- **Batch size:** 64.
- **Optimizer:** manually implemented Adam.
- **Learning rate for the final experiment:** `0.001`.
- **Training transforms:** `RandomHorizontalFlip(p=0.5)` and `ToTensor()`.
- **Validation/test transforms:** `ToTensor()` only.
- **Checkpoint selection:** highest validation accuracy.

Training and validation use the same fixed split indices throughout the experiments. Separate dataset objects allow augmentation to be applied only to training.

## Results

| Experiment | Epochs Run | Best Validation Accuracy |
|---|---:|---:|
| One convolution block | 18 | 55.86% |
| Two convolution blocks | 10 | 61.56% |
| Two blocks + horizontal flips | 10 | 62.12% |
| Two blocks + horizontal flips, continued | 50 total | **66.50%** |

### Final Model

- **Selected checkpoint:** epoch 47.
- **Validation accuracy:** 66.50%.
- **Test accuracy:** **66.24%**.
- **Correct test predictions:** 6,624 out of 10,000.

These are individual learning experiments, not controlled ablations. Initialization, batch order, training duration, and learning-rate schedules differed. The test set was evaluated after selecting the checkpoint using validation performance.

## Key Learning Takeaways

- Filters combine information across all input channels.
- Shared weights accumulate gradient contributions from every position where they are used.
- Stride and padding determine spatial output dimensions.
- Convolution can be expressed as matrix multiplication over flattened patches.
- Max pooling routes gradients to the selected maximum values.
- Deeper layers have larger receptive fields.
- Cross-entropy expects logits, not softmax probabilities.
- Gradients are cleared between updates, while Adam’s moments persist.
- Augmentation encourages learning across variations of training images.
- Validation guides model selection; test data assesses the selected model.
- Confusion matrices reveal mistakes hidden by overall accuracy.

## Running the Notebooks

Use a Python environment with:

- Jupyter or VS Code notebook support
- PyTorch
- torchvision
- NumPy
- Matplotlib

Individual notebooks may require additional packages.

Run cells in order, starting with imports, class definitions, and dataset setup. Adjust dataset paths and device settings for your machine. For CUDA, use a compatible PyTorch installation and match random generators to the devices where tensors are created.

Suggested order:

1. `micrograd/`
2. `makemore/`
3. `DeepVisionLab/mlp_baseline.ipynb`
4. `DeepVisionLab/cnn_baseline.ipynb`

## Checkpoints

The custom `Sequential` implementation saves parameter tensors directly.

To load `cnn_aug_best.pt`, recreate the same architecture and copy the saved tensors into its parameters in the same order.

The checkpoint contains trained weights and result metadata. It does not include Adam state for exact training resumption.

## Roadmap

- [ ] Package reusable components into an educational library.
- [ ] Remove global device and random-generator dependencies.
- [ ] Standardize layer, optimizer, and checkpoint interfaces.
- [ ] Add numerical checks against equivalent PyTorch operations.
- [ ] Explore normalization, dropout, and residual connections.
- [ ] Expand into further computer-vision tasks.
- [ ] Continue adding explanations and reproducible experiment notes.

## Acknowledgments

The micrograd and makemore work follows **Andrej Karpathy’s Neural Networks: Zero to Hero** teaching series.

DeepVisionLab extends that learning into custom image-classification experiments.

---

An ongoing learning repository—built one concept, implementation, and experiment at a time.
