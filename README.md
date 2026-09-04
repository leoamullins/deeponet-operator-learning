# Operator Learning

**Status:** In progress, data generation and the antiderivative dataset are complete.

Standard supervised learning methods map vectors to vectors (a feature vector to a target vector), in other words the method learns a function. Operator learning maps *functions* to *functions*; here, an input function $u$ to the output of some operator applied to it.

---

## Problem Setup

**Antiderivative operator** (validation case):

$$
G: u \mapsto s, \qquad s(x) = \int_0^x u(t) \, dt, \qquad s(0) = 0
$$

This case is deliberately trivial as an application and quadrature solves it to machine precision. It is used to debug the implementation rather than as a difficult learning problem.

---

## Data Generation

Input functions are drawn from a Guassian distribtuion on $[0, 1]$ with an RBF kernel

$$
k(x, x') = \exp\left(-\frac{(x - x')^2}{2\ell^2}\right)
$$

This uses a Cholesky factorisation, to transform a randomly sample $z$ to $u \sim N(0, K)$

Each $u$ is sampled on a fine grid and integrated using the cumulative trapezium rule.
