# Neural Network from Scratch (Pure NumPy)

A minimal 2-layer neural network implemented with **NumPy only** — no autograd, no frameworks. Every gradient in the backward pass is derived by hand via the chain rule, then coded directly.

Trained on a toy binary classification task (two interleaving "moons") that is **not linearly separable**, to demonstrate why hidden layers + nonlinearities are necessary.

## Architecture

```
Input (2) → Linear → ReLU → Linear → Sigmoid → Output (1)
              W1,b1           W2,b2
             [2,4]            [4,1]
```

| Layer | Weights | Output shape | Purpose |
|---|---|---|---|
| Linear 1 | `W1: [2,4]`, `b1: [1,4]` | `[N,4]` | Project input into a higher-dimensional space |
| ReLU | — | `[N,4]` | Nonlinearity — without it, stacked linear layers collapse into one linear layer |
| Linear 2 | `W2: [4,1]`, `b2: [1,1]` | `[N,1]` | Combine features into a single score |
| Sigmoid | — | `[N,1]` | Squash score into a probability in (0, 1) |

## Math

**Forward pass:**

```
Z1 = X @ W1 + b1        A1 = ReLU(Z1)
Z2 = A1 @ W2 + b2        A2 = sigmoid(Z2)
```

**Loss (Binary Cross-Entropy):**

```
L = -mean( y*log(A2) + (1-y)*log(1-A2) )
```

**Backward pass:** pairing sigmoid output with BCE loss collapses the output-layer gradient to a single clean term:

```
dZ2 = A2 - y
dW2 = A1.T @ dZ2 / N          db2 = mean(dZ2)
dA1 = dZ2 @ W2.T
dZ1 = dA1 * (Z1 > 0)          # ReLU derivative
dW1 = X.T @ dZ1 / N           db1 = mean(dZ1)
```

## Results

Trained for 2000 epochs with plain gradient descent (`lr=0.5`):

| Epoch | Loss | Accuracy |
|---|---|---|
| 0 | 1.161 | 0.263 |
| 200 | 0.254 | 0.880 |
| 1000 | 0.028 | 0.997 |
| 2000 | 0.015 | 0.997 |

![Dataset and decision boundary](moons_visualization.png)

The learned decision boundary is **piecewise linear** — a direct fingerprint of ReLU. Each hidden unit contributes one linear "fold"; with only 4 hidden units, the boundary is a few straight segments stitched together rather than a smooth curve. Increasing `hidden_dim` produces a finer piecewise-linear approximation of the true curved boundary.

## Files

- `nn_from_scratch.py` — network definition, training loop, and toy dataset generator
- `visualize.py` — plots the dataset and the learned decision boundary
- `moons_visualization.png` — output of `visualize.py`

## Usage

```bash
pip install numpy matplotlib
python nn_from_scratch.py   # trains and prints loss/accuracy per epoch
python visualize.py         # trains and saves moons_visualization.png
```

## Key implementation notes

- **He initialization** (`* sqrt(2/fan_in)`) is used for weights feeding into ReLU, to keep activation variance stable across layers.
- **`eps=1e-8`** is added inside `log()` in the loss to avoid `NaN` from `log(0)` once predictions become confident.
- The ReLU derivative is computed from the **pre-activation** `Z1` (`Z1 > 0`), not the post-activation `A1` — they're equivalent for ReLU specifically, but using `Z` is the pattern that generalizes to other activations.
