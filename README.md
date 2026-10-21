# Operator Learning

**Status:** In progress. The antiderivative validation case is done. The flow-past-a-body dataset (1200 convex shapes) has been generated, and a DeepONet training script for it is written.

Standard supervised learning methods map vectors to vectors (a feature vector to a target vector), in other words the method learns a function. Operator learning maps *functions* to *functions*; here, an input function $u$ to the output of some operator applied to it.

---

## Problem Setup

### Antiderivative operator (validation case)

$$
G: u \mapsto s, \qquad s(x) = \int_0^x u(t) \, dt, \qquad s(0) = 0
$$

This case is deliberately trivial as an application and quadrature solves it to machine precision. It is used to debug the implementation rather than as a difficult learning problem.

### Steady flow past a convex body

$$
G: \text{shape} \mapsto (u_x, u_y, p)
$$

The operator maps the geometry of a 2D body in a channel to the steady flow field around it at $Re = 20$. The shape is given to the network as its signed distance function (SDF), and the output is the velocity and pressure at arbitrary query points.

---

## Data Generation

### Gaussian random functions

Input functions are drawn from a Gaussian process on $[0, 1]$ with an RBF kernel

$$
k(x, x') = \exp\left(-\frac{(x - x')^2}{2\ell^2}\right)
$$

This uses a Cholesky factorisation to transform a random sample $z$ into $u \sim N(0, K)$.

For the antiderivative, each $u$ is sampled on a fine grid and integrated using the cumulative trapezium rule.

### Convex shapes

Body boundaries are defined in polar form around the centre $(c_x, c_y)$:

$$
r(\theta) = r_0 \left(1 + a_1 \cos(\theta - \phi_1) + a_2 \cos 2(\theta - \phi_2) + \varepsilon\, g(\theta)\right)
$$

Here $g$ is a periodic Gaussian random function and $a_1, a_2, \phi_1, \phi_2$ are drawn uniformly. Shapes are kept only if they are (close to) convex. The check uses the curvature of the boundary, which is computed spectrally from $r$, $r'$ and $r''$.

### Flow solver

Each shape is solved with a D2Q9 lattice Boltzmann solver ([src/solvers/lbm.py](src/solvers/lbm.py)), written in JAX and batched across shapes. The body uses bounce-back walls. The solver runs until the field stops changing (tolerance $10^{-5}$), and shapes that don't converge are dropped.

Generate the dataset with:

```bash
uv run python scripts/generate_dataset.py
```

This writes `data/train_convex.npz` (masks, SDFs, $u_x$, $u_y$, $p$, radii, convexity margins) and `data/train_convex.json` (grid size, $r_0$, centre, $Re$ and the shape-sampling settings). The default grid is $150 \times 600$ with $r_0 = 15$.

---

## Model and Training

[src/model.py](src/model.py) implements a DeepONet with three outputs:

- **Branch net:** an MLP on the SDF, downsampled to $32 \times 64$ and flattened.
- **Trunk net:** an MLP on the query coordinates $(x, y)$, in units of $r_0$ relative to the body centre.

The output for each field is the dot product of the branch and trunk features, plus a bias.

[src/train.py](src/train.py) works as follows:

- Crops the domain to $[-2r_0, 8r_0] \times [-2.5r_0, 2.5r_0]$ around the body.
- Measures pressure relative to the upstream edge.
- Normalises each field with mean and std from the training fluid cells.
- Holds out 200 shapes for validation.
- Trains with AdamW, linear warmup and cosine decay. Each batch uses 32 shapes and 2048 random query points, and the MSE is masked to fluid cells.
- Every 10 epochs, reports the relative $L^2$ error on the validation set for each field.

```bash
uv run python src/train.py
```

The best model is saved to `checkpoints/best.pt`, together with its model config, normalisation stats, data config and train/val split. The loss history goes to `checkpoints/history.json`. To reload the model:

```python
ckpt = torch.load("checkpoints/best.pt")
model = DeepONet(**ckpt["model_cfg"])
model.load_state_dict(ckpt["model"])
```

---

## Layout

```text
scripts/generate_dataset.py   shape sampling + LBM solves -> data/
src/grf.py                    Gaussian random function sampling
src/geometry.py               shape sampling, masks, SDF, convexity test
src/solvers/lbm.py            batched JAX lattice Boltzmann solver
src/model.py                  DeepONet
src/train.py                  training on the convex-shape dataset
src/data_generator.py         antiderivative targets
```
