# Cross-Initialization Lottery Subspace Transfer Experiment

## Goal

Test how well a **lottery subspace** computed from one training run ("donor") works when reused for other random initializations ("recipients") of the **same architecture and dataset**.

We measure:

- How much performance degrades (or not) when recipients are forced to train in the donor's subspace instead of their own.

---

## Background: Lottery Subspaces (Paper Setup)

The paper *"How Many Degrees of Freedom Do We Need to Train Deep Networks?"* shows that a network with \(D\) parameters can often be trained almost entirely within a low-dimensional subspace of dimension \(d \ll D\).

They construct a **lottery subspace** like this:

1. Train a network normally, storing weights \(w_0, w_1, \dots, w_T\).
2. Form the matrix of trajectory deltas:
   \[
   \Delta w_t = w_t - w_0, \quad t = 1, \dots, T
   \]
3. Compute the SVD/PCA of these deltas and keep the top \(d\) directions:
   \[
   U_d = [u_1 \mid u_2 \mid \dots \mid u_d]
   \]
4. Retrain in the affine subspace
   \[
   w(\theta) = w_{\text{base}} + U_d \theta
   \]
   for some chosen basepoint \(w_{\text{base}}\) along the original trajectory.

They show that training restricted to this \(d\)-dimensional subspace can reach nearly the same accuracy as full-space training.

---

## Question We Are Asking

Given a lottery subspace \(U_{\text{donor}}\) computed from one seed:

> How well does \(U_{\text{donor}}\) work when we train other initializations of the same network, restricted to that subspace?

Concretely:

- Does a network initialized with a different random seed, but trained only along directions in \(U_{\text{donor}}\), reach similar accuracy to:
  - (a) a full-space baseline, or
  - (b) a lottery subspace constructed from its *own* trajectory?

We treat this purely as an empirical question; we do not assume that the subspace is "universal" or "intrinsic."

---

## Experimental Design

### Architectures and Data

- **Architecture:** Tiny CNN (Conv-2 style), ~20k parameters (same structure as in the paper).
- **Dataset:** MNIST.
- **Optimizer:** Adam, learning rate \(0.05\).
- **Training schedule:** 3 epochs for all runs (full-space and subspace).
- **Subspace dimensions tested:**
  \[
  d \in \{8, 32, 128, 512, 1024, 2048\}
  \]

### Donor / Recipient Setup

We fix:

- **Donor seed:** 0
- **Recipient seeds:** 100, 101, 102, 103, 104

#### Step 1: Donor Lottery Subspace

For the donor network:

1. Initialize network with seed 0 → weights \(w^{(d)}_0\).
2. Train for 3 epochs and record weights \(w^{(d)}_t\).
3. Form trajectory deltas \(\Delta w^{(d)}_t = w^{(d)}_t - w^{(d)}_0\).
4. Compute SVD and extract the top-\(d\) directions:
   \[
   U_{\text{donor}, d}
   \]

This gives us, for each \(d\), a donor lottery subspace \(U_{\text{donor}, d}\).

#### Step 2: Recipient Baseline (Own Subspace)

For each recipient seed \(s \in \{100, \dots, 104\}\):

1. Initialize network with seed \(s\): \(w^{(s)}_0\).
2. Train in full parameter space for 3 epochs, record trajectory.
3. Construct that recipient's own lottery subspace \(U^{(s)}_d\) using the same SVD procedure.
4. Retrain in that subspace using the affine parameterization
   \[
   w^{(s)}(\theta) = w^{(s)}_0 + U^{(s)}_d \theta
   \]

This gives a **per-recipient baseline**: "How well can seed \(s\) do when confined to its own lottery subspace?"

#### Step 3: Recipient Transfer (Donor Subspace)

For the **transfer condition**, we keep the recipient's initialization but use the donor's directions:

For each recipient seed \(s\) and each dimension \(d\):

1. Initialize with seed \(s\): \(w^{(s)}_0\).
2. Train restricted to the **donor** subspace:
   \[
   w^{(s)}_{\text{transfer}}(\theta) = w^{(s)}_0 + U_{\text{donor}, d} \theta
   \]

So:

- **Basepoint** is recipient's own init \(w^{(s)}_0\).
- **Directions** are donor's PCs \(U_{\text{donor}, d}\).

This is the core "cross-initialization transfer" experiment.

---

## Conditions Summary

For each recipient seed \(s\) and each subspace dimension \(d\):

| Condition       | Basepoint         | Directions used         | What it measures                              |
|-----------------|-------------------|-------------------------|-----------------------------------------------|
| Full-space      | — (no subspace)   | —                       | Unconstrained baseline (from the paper setup) |
| Own-subspace    | \(w^{(s)}_0\)     | \(U^{(s)}_d\)           | Standard lottery subspace performance         |
| Donor-transfer  | \(w^{(s)}_0\)     | \(U_{\text{donor}, d}\) | Cross-seed subspace transfer                  |

The main quantity of interest is the **gap** between:

- Own-subspace accuracy vs. Donor-transfer accuracy, as a function of \(d\).

---

## Metrics and Plots

For each \(d\):

- Compute test accuracy for:
  - Full-space baseline (per seed).
  - Own-subspace (per seed).
  - Donor-transfer (per seed).

Then:

1. **Accuracy vs. \(d\):**
   Plot mean accuracy over recipient seeds for own-subspace and donor-transfer.

2. **Gap vs. \(d\):**
   For each \(d\),
   \[
   \text{Gap}(d) = \mathbb{E}_s[\text{Acc}^{(s)}_{\text{own}}(d)] - \mathbb{E}_s[\text{Acc}^{(s)}_{\text{transfer}}(d)]
   \]
   Plot this to show how much performance is lost when using donor directions.

3. Optionally, show per-seed points (scatter) to visualize variance across seeds.

We treat the numeric gap and its trend over \(d\) as the main empirical result. We do not interpret small differences as "proof" of anything beyond this particular setup.

---

## How to Run

```bash
cd degrees-of-freedom/

# Full cross-init transfer experiment (runs all 4 steps automatically)
./run_transfer_experiment.sh

# Quick debug / sanity run (fewer dimensions)
./run_transfer_experiment.sh --quick

# Plot and inspect results
python plot_transfer_experiment.py --show
```

### What the Script Does

The `run_transfer_experiment.sh` script runs four steps:

1. **Generate donor lottery subspace** — Trains seed 0 and extracts principal directions.
2. **Run transfer experiment** — Trains seeds 100–104 using the *donor's* subspace.
3. **Run baseline experiment** — Trains seeds 100–104 using their *own* subspaces.
4. **Generate comparison plots** — Creates figures comparing transfer vs. baseline.

### Output Files

Results are saved to `../cross-init-lottery-subspace-data/`:

- `artifact_lottery_subspace_*_grad00.pkl` — Donor subspace data
- `artifact_lottery_subspace_transfer_*_transfer_*_results.pkl` — Transfer experiment results
- `artifact_lottery_subspace_transfer_*_baseline_*_results.pkl` — Baseline experiment results

Figures are saved to `figures/`:

- `fig_transfer_comparison.png` — Accuracy vs. dimension for transfer vs. baseline
- `fig_transfer_gap.png` — Accuracy gap (baseline − transfer) vs. dimension
- `fig_transfer_individual.png` — Scatter plot of individual runs

---

## Interpreting Results

- **Small gap (< 2%)**: Transfer works well. Lottery subspaces may capture architecture-intrinsic structure.
- **Moderate gap (2–5%)**: Partial transfer. Subspaces have some shared structure but are partially init-specific.
- **Large gap (> 5%)**: Subspaces appear init-specific. The donor's directions don't generalize well.

The gap typically decreases as \(d\) increases, since larger subspaces contain more of the full parameter space.
