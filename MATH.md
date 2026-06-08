# Mathematical Foundations

> Implementation-oriented summary of the mathematics behind
> `gaussian-dag`: the linear Gaussian DAG model, the K-recursion that
> assembles all node-pair covariances in one forward sweep, the
> log-determinant mutual information that follows, and the Wirtinger
> gradient and projected gradient ascent that maximize it under
> network-wide constraints.
>
> The **formal version of record** is the arXiv paper
> *Mutual Information Optimization via K-Recursion and Automatic
> Differentiation for Linear Gaussian Wireless Networks*
> (Wadayama & Na, [arXiv:2606.06982](https://arxiv.org/abs/2606.06982), 2026). This document
> is a complementary implementation-side exposition that uses the
> library's API and 0-based indexing throughout, and points to the
> code paths in `gaussian_dag/`. For full proofs, related work, and
> formal statements, please consult the paper.

## Contents

1. [Purpose and scope](#1-purpose-and-scope)
2. [Linear Gaussian DAG model](#2-linear-gaussian-dag-model)
3. [Edge factorization and the design parameter](#3-edge-factorization-and-the-design-parameter)
4. [The K-recursion](#4-the-k-recursion)
5. [Mutual information from K-blocks](#5-mutual-information-from-k-blocks)
6. [Effective-channel representation](#6-effective-channel-representation)
7. [Wirtinger gradient and projected gradient ascent](#7-wirtinger-gradient-and-projected-gradient-ascent)

---

## 1. Purpose and scope

Modern wireless networks are cascades of stochastic stages with
controllable parameters at every layer (precoders, relay gains, RIS
phase profiles, ...). The Palomar–Verdú gradient for single-link MIMO
established analytic MI-based design for *one* vector Gaussian channel;
what is still missing is a *topology-agnostic* mechanism that

- extends to arbitrary networks built from cascaded linear Gaussian
  stages, and
- is compatible with modern automatic differentiation so that no new
  gradient formula is derived per topology.

`gaussian-dag` provides such a mechanism. The end-to-end mutual
information of a linear Gaussian DAG admits a closed-form
log-determinant expression; all node-pair covariances entering that
expression are produced by a single forward recursion (the
**K-recursion**); and reverse-mode autograd on this recursion returns
the exact Wirtinger gradient at every controllable factor in one
backward sweep. Projected gradient ascent then drives the MI upward
under network-wide constraints (Frobenius / total-power / unit
modulus / ...). The same code path handles single-link MIMO,
branching/merging diamonds, multi-hop AF relays, RIS-aided channels,
and input-covariance shaping.

A note on indexing. The arXiv paper uses 1-based node indices
($V_1, \ldots, V_M$). The library uses **0-based** indices
($V_0, \ldots, V_{M-1}$); node 0 is the unique root, node $M-1$ the
sink. The K-recursion is identical in form under either convention.
This document uses the library's 0-based convention throughout.

---

## 2. Linear Gaussian DAG model

Let $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ be a topologically
ordered DAG with node set $\mathcal{V} = \{0, 1, \ldots, M-1\}$. Each
node $V_j \in \mathbb{C}^{d_j}$ is a complex random vector. The
topological order means $i < j$ for every edge $(i \to j) \in \mathcal{E}$,
equivalently $\mathrm{Pa}(j) \subset \{0, \ldots, j-1\}$ for all
$j \geq 1$.

The source is the unique root

$$X := V_0 \sim \mathcal{CN}(0, \Sigma_X), \qquad \Sigma_X \succ 0,$$

and the sink is $Y := V_{M-1}$. For each non-root node $j \geq 1$,

$$V_j = \sum_{i \in \mathrm{Pa}(j)} A_{ji} V_i + Z_j, \qquad Z_j \sim \mathcal{CN}(0, \Sigma_j), \tag{2.1}$$

with edge transforms $A_{ji} \in \mathbb{C}^{d_j \times d_i}$ and
PSD additive noise $\Sigma_j$. The noises $\{Z_j\}$ are mutually
independent and independent of $X$.

In code, $A_{ji}$ is indexed as `edge_mats[(j, i)]`, $\Sigma_j$ as
`noise_covs[j]`, and $\mathrm{Pa}(j)$ as `parents[j]`. The PSD
assumption $\Sigma_j \succeq 0$ allows noiseless or rank-deficient
intermediate nodes; positive definiteness of $\Sigma_Y$ and
$\Sigma_{Y \mid X}$ is required only when the log-det MI is evaluated
(§5).

---

## 3. Edge factorization and the design parameter

Each edge transform is the product of $K_{ji} \geq 1$ factors,

$$A_{ji} = A_{ji}^{(1)} A_{ji}^{(2)} \cdots A_{ji}^{(K_{ji})}, \tag{3.1}$$

read from the output side ($\ell = 1$) to the input side ($\ell = K_{ji}$).
Each $A_{ji}^{(\ell)}$ is labeled either **controllable** (a precoder,
RIS phase profile, relay gain, ...) or **constant** (a fixed channel
realization). The **design parameter** is the tuple of all controllable
factors,

$$\eta := \{ A_{ji}^{(\ell)} : (j, i, \ell) \in \mathcal{C} \},$$

optimized over a **feasible set** $\mathcal{S}$ encoding the design
constraints — Frobenius per-factor balls, shared total-power balls,
unit-modulus RIS phases, scalar relay gains, etc.

**Note on edge factorization in the library.** `compute_k_blocks`
takes the already-composed $A_{ji}$ in `edge_mats[(j, i)]`. The
multiplicative decomposition (3.1) is handled by the user closure:
construct each `edge_mats[(j, i)]` as `H @ F` (single-link precoder),
`H @ R` (AF relay gain), or `H[(j, i)] @ F_i` (shared relay factor),
and PyTorch autograd accumulates gradient contributions through every
edge that uses a given controllable factor in one backward sweep
(parameter sharing comes for free).

The four canonical topologies covered by the framework — single-link
MIMO, diamond, multi-hop AF relay, and virtual-edge input shaping —
are all special cases of (3.1). See the README's hero figure and
`examples/`.

---

## 4. The K-recursion

### 4.1 Node-pair covariances and canonical storage

For any pair of node indices $(j, k)$ (not necessarily adjacent in
the DAG), define the node-pair covariance

$$K_{jk} := \mathbb{E}[V_j V_k^{\mathsf{H}}] \in \mathbb{C}^{d_j \times d_k}. \tag{4.1}$$

Hermitian symmetry gives $K_{kj} = K_{jk}^{\mathsf{H}}$, so only the
canonical half $\{K_{jk} : j \geq k\}$ is stored. Reads of the
non-canonical half use the **Hermitian flip**

$$K_{ab} := K_{ba}^{\mathsf{H}}, \qquad a < b. \tag{4.2}$$

In code: `compute_k_blocks` returns a dict keyed by canonical
$(j, k)$ with $j \geq k$, and `get_K(K, a, b)` applies the Hermitian
flip on the fly.

### 4.2 The recursion

**Proposition (K-recursion).** *Under the model of §2 processed in
topological order $j = 0, 1, \ldots, M-1$, and with the Hermitian-flip
convention (4.2) for any non-canonical block appearing on the right
below, the canonical K-blocks satisfy*

$$K_{jk} = \begin{cases} \Sigma_X & j = k = 0, \\ \sum_{i \in \mathrm{Pa}(j)} A_{ji} K_{ik} & j \geq 1, \; k < j, \\ \sum_{i, i' \in \mathrm{Pa}(j)} A_{ji} K_{i i'} A_{j i'}^{\mathsf{H}} + \Sigma_j & j \geq 1, \; k = j. \end{cases} \tag{4.3}$$

*Each step uses only matrix products, sums, and Hermitian
transposes, so the full collection $\{K_{jk} : j \geq k\}$ is obtained
in a single forward sweep and is a smooth function of $\eta$.*

The proof is by strong induction in topological order. See the
arXiv paper for the full argument.

### 4.3 Parent cross-covariances are indispensable at merging nodes

The double sum in the self-block branch ($k = j$) forces a merging
node's self-block to depend on the *cross-covariance of its parents*.
Tracking only the auto-covariances $K_{jj}$ is therefore insufficient
in general DAGs. Concretely, for the diamond DAG (nodes
$V_0 \to V_1, V_2$, $V_1, V_2 \to V_3$) the merging self-block reads

$$K_{33} = A_{31} K_{11} A_{31}^{\mathsf{H}} + A_{31} K_{12} A_{32}^{\mathsf{H}} + A_{32} K_{21} A_{31}^{\mathsf{H}} + A_{32} K_{22} A_{32}^{\mathsf{H}} + \Sigma_3, \tag{4.4}$$

with $K_{12} = K_{21}^{\mathsf{H}}$ via the Hermitian flip. The parent
cross-covariance $K_{21}$ explicitly enters $K_{33}$ and hence the
log-det MI.

### 4.4 Computational complexity

Let $d_{\max} = \max_j d_j$ and $|\mathcal{E}|$ the number of DAG
edges. The cross-block and self-block updates of the K-recursion cost
$O(M |\mathcal{E}| d_{\max}^3)$ and $O(\sum_j |\mathrm{Pa}(j)|^2 d_{\max}^3)$
respectively. For sparse DAGs with $|\mathrm{Pa}(j)| = O(1)$, overall
cost is $O(M^2 d_{\max}^3)$ and canonical-block storage is
$O(M^2 d_{\max}^2)$. Reverse-mode AD costs at most a small constant
multiple of the forward sweep.

---

## 5. Mutual information from K-blocks

**Proposition (MI from K-blocks).** *Under the regularity assumption
$\Sigma_Y(\eta), \Sigma_{Y \mid X}(\eta) \succ 0$, the mutual
information of $V_0 = X$ and $V_{M-1} = Y$ is the log-determinant
difference*

$$I(X; Y) = \log\det \Sigma_Y(\eta) - \log\det \Sigma_{Y \mid X}(\eta), \tag{5.1}$$

*where*

$$\Sigma_Y = K_{M-1, M-1}, \qquad \Sigma_{Y \mid X} = K_{M-1, M-1} - K_{M-1, 0} K_{00}^{-1} K_{M-1, 0}^{\mathsf{H}}, \tag{5.2}$$

*read off from the K-blocks. The map $\eta \mapsto I(X; Y)$ is smooth
on the open set where the regularity assumption holds.*

In code, this is evaluated by `mutual_information_from_k(K, output_node=M-1, input_node=0)`.
A small `jitter > 0` keyword stabilizes near-singular Cholesky paths
and matches the standard regularity treatment.

**Numerical implementation.** The library symmetrizes
$\Sigma_Y$ and $\Sigma_{Y \mid X}$ as
$\tfrac{1}{2}(\Sigma + \Sigma^{\mathsf{H}})$ to remove floating-point
asymmetry and evaluates
$\log\det(\Sigma + \varepsilon I) = 2 \sum_i \log L_{ii}$ via Cholesky
to keep the objective smooth in $\eta$ even near the boundary of the
PD cone. Cholesky failure raises a clear `ValueError` (not an opaque
autograd error), with three suggested remedies in the docstring.

All MI values are in **nats**.

---

## 6. Effective-channel representation

The Gaussian-DAG can equivalently be collapsed to a single linear
Gaussian channel

$$Y = G_{M-1} X + R_{M-1}, \qquad R_{M-1} \sim \mathcal{CN}(0, C_{M-1, M-1}), \tag{6.1}$$

with $R_{M-1}$ independent of $X$ and effective channel matrices

$$G_0 = I_{d_X}, \qquad G_j = \sum_{i \in \mathrm{Pa}(j)} A_{ji} G_i \quad (j \geq 1). \tag{6.2}$$

The effective-noise blocks
$C_{jk} := \mathbb{E}[R_j R_k^{\mathsf{H}}]$ obey the K-recursion of §4
with the input covariance set to zero ($C_{00} = 0$), so

$$K_{jk} = G_j \Sigma_X G_k^{\mathsf{H}} + C_{jk}, \tag{6.3}$$

and the MI admits the equivalent representation

$$I(X; Y) = \log\det( G_{M-1} \Sigma_X G_{M-1}^{\mathsf{H}} + C_{M-1, M-1} ) - \log\det C_{M-1, M-1}. \tag{6.4}$$

In code, this is exposed as `compute_effective_channel`, returning
the pair $(G, C)$: a dict `G[j] -> G_j` of effective channel matrices,
and a dict `C[(j, k)] -> C_{jk}` of effective-noise blocks in the same
canonical-storage convention as the K-blocks.

The K-recursion is the more general primitive — it exposes the
intermediate covariances and parent cross-covariances at merging
nodes that internal constraints, multiple sinks, and downstream
Bussgang-type linearizations require. The effective-channel
representation is more compact for a single source–sink MI and is
the natural interface for ISAC FIM construction (see the sister
library [`gaussian-dag-isac`](https://github.com/wadayama/gaussian-dag-isac)).

---

## 7. Wirtinger gradient and projected gradient ascent

### 7.1 Computation graph

By the proposition of §5, the map $\eta \mapsto I(X; Y)$ factors
through a five-stage smooth composition:

$$\eta \;\longrightarrow\; \{A_{ji}\} \;\longrightarrow\; \{K_{jk}\} \;\longrightarrow\; (\Sigma_Y, \Sigma_{Y \mid X}) \;\longrightarrow\; I(X; Y). \tag{7.1}$$

Each arrow is composed of differentiable primitives standard in
modern autograd engines (matrix product, sum, Hermitian transpose,
matrix inverse / solve, log-determinant). Because $\mathcal{G}$ is a
DAG, the forward graph is itself acyclic — the backward sweep is
well-defined in reverse topological order with no unrolling or
fixed-point iteration.

### 7.2 The Wirtinger gradient

For a real-valued $f : \mathbb{C}^{p \times q} \to \mathbb{R}$, the
**conjugate-side Wirtinger gradient** is

$$\nabla_{\Theta^{\ast}} f := ( \partial f / \partial \Theta^{\ast} )^{\mathsf{T}},$$

the steepest-ascent direction in the standard real-Euclidean metric.
A single reverse-mode AD pass on the K-recursion returns the gradient
$\partial I / \partial ( A_{ji}^{(\ell)} )^{\ast}$ at every
controllable factor $(j, i, \ell) \in \mathcal{C}$ simultaneously, by
the cheap-gradient principle. The applicability of the Wirtinger
chain rule to arbitrary complex matrix parameters was formalized by
Schreier & Scharf; the AD machinery here can be viewed as its
automatic, topology-agnostic execution.

**PyTorch-specific note.** PyTorch populates each complex leaf's
`.grad` attribute with $2 \, \nabla_{\Theta^{\ast}} f$ rather than
$\nabla_{\Theta^{\ast}} f$ itself. The factor of two is absorbed into the
step size of any first-order optimizer and does not affect
optimization behavior. The K-recursion + log-det MI pipeline is
otherwise framework-agnostic.

### 7.3 Projected gradient ascent

The exact Wirtinger gradient is precisely the input required by PGA:

$$\eta^{(t+1)} = \mathcal{P}_{\mathcal{S}}( \eta^{(t)} + \alpha_t \, \nabla_{\eta^{\ast}} I(\eta^{(t)}) ), \tag{7.2}$$

with constant step size $\alpha_t > 0$ and Euclidean projection
$\mathcal{P}_{\mathcal{S}}$ onto the feasible set. In code, the loop
is `pga_ascent(compute_mi, params, *, step_size, num_iters, projector)`.
Each iteration costs one K-recursion forward sweep, one reverse-mode
AD backward sweep, and one closed-form projection. The MI objective
is in general non-concave; PGA converges to a stationary point under
standard step-size schedules, and multi-start is recommended for
production use.

### 7.4 Closed-form projections

Three projections cover the constraints used in the bundled examples:

- **Total Frobenius budget** $\sum_{(j, i, \ell) \in \mathcal{C}} \| A_{ji}^{(\ell)} \|_F^2 \leq P$:
  a *single common* scale factor is applied to every controllable
  factor. Implementation: `project_total_power(params, P)`.
- **Per-factor Frobenius budget** $\| A_{ji}^{(\ell)} \|_F^2 \leq P_{ji}^{(\ell)}$:
  each factor is rescaled independently onto its own Frobenius ball.
  Implementation: `project_frobenius_ball(A, P)`.
- **Diagonal / scalar / unit-modulus structural controls**: the
  structural form is enforced by directly parameterizing the
  underlying scalars/vectors; any additional norm or modulus
  constraint is then projected in that lower-dimensional parameter
  space (e.g., $\theta_m \leftarrow \theta_m / | \theta_m |$ for unit
  modulus).

Stiefel / orthogonal / low-rank truncation projections are equally
compatible with the framework but are not used by the bundled
examples.

### 7.5 Input shaping via a virtual source edge

An input covariance $\Sigma_X$ can itself be optimized by introducing
a virtual source $S \sim \mathcal{CN}(0, I)$ and a controllable input
factor $Q$ with $X = Q S$, so that $\Sigma_X = Q Q^{\mathsf{H}}$ and a
trace power constraint $\mathrm{tr}(\Sigma_X) \leq P$ becomes
$\| Q \|_F^2 \leq P$. The augmented DAG is handled by the same
K-recursion, so input-covariance design reduces to optimization of one
additional controllable edge factor. Solving for $Q$ alone (with all
non-source edges frozen) reaches the **classical water-filling
capacity** of the effective channel
$(G_{M-1}, C_{M-1, M-1})$ of §6 — recovered numerically by
`examples/input_covariance.py` to within ~5×10⁻⁴ nats of the
water-filling reference.

---

## Notation summary

| Symbol | Meaning |
| --- | --- |
| $V_j \in \mathbb{C}^{d_j}$ | DAG node ($j = 0, \ldots, M-1$; $V_0 = X$, $V_{M-1} = Y$). |
| $\mathrm{Pa}(j)$ | Parent index set of node $j$. |
| $A_{ji}, A_{ji}^{(\ell)}$ | Edge transform on edge $i \to j$ and its $\ell$-th factor. |
| $\Sigma_X$ | Input covariance at the source $V_0$. |
| $\Sigma_j$ | Independent additive Gaussian noise covariance at node $j$. |
| $K_{jk} = \mathbb{E}[V_j V_k^{\mathsf{H}}]$ | Node-pair covariance block. |
| $G_j, C_{jk}$ | Effective channel matrix and effective-noise covariance block (§6). |
| $\Sigma_Y, \Sigma_{Y \mid X}$ | Output and conditional output covariances. |
| $I(X; Y)$ | Mutual information, in nats. |
| $\eta$ | Tuple of controllable edge factors (the design parameter). |
| $\mathcal{S}, \mathcal{P}_{\mathcal{S}}$ | Feasible set and its Euclidean projection. |

---

## What's next

The implementation conventions, public API, examples, and tutorials
are described in [`README.md`](README.md) and the five-part walkthrough
under [`docs/`](docs/). For the **formal proofs** of the K-recursion
(Proposition 1 in §IV of the paper), the MI representation
(Proposition 2 in §IV), the effective-channel capacity attainment
(Proposition 3 in §V), and the related-work / experimental sections,
consult the arXiv paper
[arXiv:2606.06982](https://arxiv.org/abs/2606.06982).
