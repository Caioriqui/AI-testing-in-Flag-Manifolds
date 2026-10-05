# AI Testing in Flag Manifolds

Final project for MM845 (*Topics in Geometry III: AI for Geometry*) at the University of Campinas (UNICAMP).

The project studies when a labelled Coxeter diagram is **spherical** (equivalently, when its Cartan/Gram matrix is positive definite, i.e. it is a Dynkin diagram) and asks whether a reinforcement-learning agent can turn a random diagram into a spherical one by local edge edits, using only the definition of sphericity and none of Dynkin's necessary conditions (no cycles, bounded degree, arm lengths, ...).

It combines:

1. an **exhaustive search** over all labelled diagrams up to N = 6, exploiting permutation symmetry;
2. a **PPO agent with a hand-written graph neural network** (PyTorch only, no GNN library), trained either directly at N = 8 or with a curriculum over N;
3. **baselines** (uniform-random and greedy one-step margin ascent) and notebooks to analyse everything.

## Contents

- [Problem setup](#problem-setup)
- [Repository layout](#repository-layout)
- [Branches](#branches)
- [Environment setup](#environment-setup)
- [Quick sanity checks](#quick-sanity-checks)
- [Training PPO](#training-ppo)
- [Evaluating a checkpoint](#evaluating-a-checkpoint)
- [Notebooks](#notebooks)
- [Reproducibility notes](#reproducibility-notes)

## Problem setup

A diagram on N vertices is a symmetric integer matrix `M`. Off-diagonal entries (labels) lie in `{2, 3, 4, 6}`, where `2` means *no edge*; the diagonal stores `0` and is unused. The Cartan matrix is `A_ii = 2`, `A_ij = -2 cos(pi / m_ij)`, and the diagram is spherical iff `A` is positive definite.

- **State:** the label matrix, padded to a fixed `N_MAX = 8` with isolated vertices when an episode uses fewer vertices.
- **Action:** one discrete choice of (unordered vertex pair, label), giving C(8,2) · 4 = 112 actions; actions touching padding vertices are masked.
- **Reward:** `tanh` of the smallest eigenvalue of `A` (the *spherical margin*) after each edit. An episode ends on success (spherical diagram) or after a fixed budget of edits.
- **Datasets of initial diagrams:** every pair gets an i.i.d. label. **Dataset A** is uniform over `{none, 3, 4, 6}` (about 21 edges at N = 8). **Dataset B** has P(none) = 3/4 (about N − 1 = 7 edges, close to the tree regime); it is used only as a diagnostic.

## Repository layout

| Path | Contents |
| --- | --- |
| `environment.yml` | Conda environment specification (see [Environment setup](#environment-setup)). |
| `src/coxeter.py` | Diagram representation, Cartan matrix, spherical test and margin, graph utilities, permutation-symmetry tools. |
| `src/exhaustive.py` | Exhaustive search up to N = 6 over isomorphism classes, weighted by orbit size. |
| `src/datagen.py` | i.i.d. samplers for Datasets A and B and their structural statistics. |
| `src/env.py` | RL environment (single and vectorised), with padding for variable N. |
| `src/policy.py` | Actor-critic graph neural network. |
| `src/train_ppo.py` | PPO training script, run from the terminal. |
| `src/analyze.py` | Checkpoint evaluation, Coxeter-type classification, training curves. |
| `notebooks/` | Benchmarks, structural analysis, training-history comparison, inference. |
| `data/` | Representative diagrams produced by notebook 01 (input of notebook 02). |
| `results/` | Checkpoints, training logs, baselines and figures. Generated; mostly ignored by Git. |

## Branches

| Branch | Focus |
| --- | --- |
| `main` | Fixed-N PPO (the script defaults to N = 8 and accepts `--n`) and the current versions of notebooks 03 and 04. |
| `variable-N` | Curriculum training over N (stages `{3,4,5}` → `{3,…,6}` → `{3,…,7}` → `{3,…,8}`, promoted by rolling success rate) and its own versions of notebooks 03 and 04. |

The training options and saved outputs differ between the branches. Before running an experiment, read that branch's `src/train_ppo.py` and the setup cells of its notebooks.

```bash
git switch variable-N   # curriculum experiments
git switch main         # fixed-N experiments
```

Files under `results/` come from different branches and runs; identify an experiment by its run directory name, the notebook's input paths and the `config` stored in each checkpoint, not by assuming everything belongs to one branch.

## Environment setup

The environment is specified in `environment.yml` and managed with [conda](https://docs.conda.io) (we recommend [Miniforge](https://github.com/conda-forge/miniforge#download)). Setup has two steps: create the conda environment, then install PyTorch.

### 1. Create the conda environment

From the repository root:

```bash
conda env create -f environment.yml
conda activate iaflag
```

- **Windows:** run these commands in the **Miniforge Prompt**, not in plain PowerShell.
- Run `conda activate iaflag` again in every new terminal before working on the project.

### 2. Install PyTorch (separately)

PyTorch is **not** installed by `environment.yml`: the right build depends on your hardware, so you must install it yourself, with the `iaflag` environment active.

**CPU only** (laptops; enough to run everything in this project):

```bash
pip install torch
```

**NVIDIA GPU (Linux/Windows):** use the selector at <https://pytorch.org/get-started/locally/> (choose *Pip*) to get the command matching your CUDA version, for example:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

**Apple Silicon (macOS):** the plain `pip install torch` above is enough.

`src/train_ppo.py` uses a CUDA GPU automatically when one is available and runs on CPU otherwise. The model is small, so CPU training is fine.

Verify the installation:

```bash
python -c "import numpy, torch, pandas, matplotlib; print(numpy.__version__, torch.__version__)"

You should see two version numbers and no error. If `import torch` fails with `ModuleNotFoundError`, you are probably in the wrong environment: check that your prompt starts with `(iaflag)`.

On Windows, `train_ppo.py` sets `KMP_DUPLICATE_LIB_OK=TRUE` to avoid a common OpenMP library conflict.

### Notebook kernel

If the notebooks do not see the environment, register it once:

```bash
python -m ipykernel install --user --name iaflag --display-name "Python (iaflag)"
```

### Updating the environment file

After installing a new package with `conda`, update the file with:

```bash
conda env export --from-history > environment.yml
```

`--from-history` lists only packages installed with `conda`, so anything installed with `pip` (such as PyTorch) is left out on purpose.
```

## Quick sanity checks

Run all commands from the repository root with the environment active. Each module has self-tests in its `__main__` block:

```bash
python -m src.coxeter      # known diagrams: A_3, D_4, B_2, G_2, affine A~_2, symmetry counts
python -m src.datagen      # sampler statistics for Datasets A and B
python -m src.env          # environment, padding and action encoding checks
python -m src.policy       # network shapes, action masking, padding isolation
python -m src.analyze --self-test   # Coxeter-type classifier (A, B/C, D, E, F, G)
python -m src.exhaustive   # exhaustive search N = 1..6 plus a cross-check against forest pruning
```

`src.exhaustive` reproduces the N ≤ 6 counts and is the slowest (N = 6 takes a few minutes).

## Training PPO

Training is a terminal script, never run from a notebook. Always use `python -m src.<module>` from the repository root.

### Fixed N (`main`)

Each command writes to its own directory, so the six comparison runs do not overwrite one another. Run them one at a time:

```bash
python -m src.train_ppo --dataset A --n 8 --max-episode-steps 30 --seed 0 --total-timesteps 2000000 --checkpoint-dir results/A_seed0_Nfixed8
python -m src.train_ppo --dataset A --n 8 --max-episode-steps 45 --seed 0 --total-timesteps 2000000 --checkpoint-dir results/A_seed0_steps45_Nfixed8
python -m src.train_ppo --dataset A --n 8 --max-episode-steps 60 --seed 0 --total-timesteps 2000000 --checkpoint-dir results/A_seed0_steps60_Nfixed8
python -m src.train_ppo --dataset B --n 8 --max-episode-steps 30 --seed 0 --total-timesteps 2000000 --checkpoint-dir results/B_seed0_Nfixed8
python -m src.train_ppo --dataset B --n 8 --max-episode-steps 45 --seed 0 --total-timesteps 2000000 --checkpoint-dir results/B_seed0_steps45_Nfixed8
python -m src.train_ppo --dataset B --n 8 --max-episode-steps 60 --seed 0 --total-timesteps 2000000 --checkpoint-dir results/B_seed0_steps60_Nfixed8
```

### Curriculum over N (`variable-N`)

```bash
git switch variable-N
python -m src.train_ppo --dataset A --seed 0 --total-timesteps 2000000 --checkpoint-dir results/A_seed0
python -m src.train_ppo --dataset B --seed 0 --total-timesteps 2000000 --checkpoint-dir results/B_seed0
```

The curriculum options are `--curriculum-success-threshold` (default 0.7), `--curriculum-window-updates` (20) and `--curriculum-min-updates-per-stage` (20). The vertex count is sampled uniformly from the current stage, and earlier stages are never dropped.

### Notes on both

- For a short smoke test use `--total-timesteps 100000`. It checks that everything runs but does **not** reproduce the reported training horizon.
- The defaults use 16 environments and 64 rollout steps per update, so each update advances 1,024 steps; a request for 2,000,000 timesteps therefore stops at 1,999,872.
- Each run directory receives `checkpoint_latest.pt`, periodic `checkpoint_step_<steps>.pt` files (recovery checkpoints only) and `train_log.npz`. The log holds `timesteps`, `reward`, `margin` and `success_rate` (and `stage` on the curriculum branch).
- Change `--seed` for other seeds, and `--n` (minimum 2) on `main` for other vertex counts.

## Evaluating a checkpoint

```bash
python -m src.analyze --checkpoint results/A_seed0_Nfixed8/checkpoint_latest.pt --n-episodes 500 --dataset A
python -m src.analyze --checkpoint results/B_seed0_Nfixed8/checkpoint_latest.pt --n-episodes 500 --dataset B
```

Add `--deterministic` for argmax actions. On the curriculum branch, `--active-ns 3 4 5 6 7 8` evaluates several vertex counts at once. The command prints the success rate, the mean number of edits to success and the Coxeter type of some successful diagrams, and, if `train_log.npz` sits beside the checkpoint, saves the training curves there.

## Notebooks

Start Jupyter from the repository root so relative paths resolve, select the project kernel, and run the cells from top to bottom:

```bash
jupyter lab
```

| Notebook | What it does | Inputs |
| --- | --- | --- |
| `01_exhaustive_search_benchmark` | Runs the exhaustive search, writes a summary CSV and optionally saves representative diagrams. N = 6 is heavy. | none |
| `02_spherical_structure_analysis` | Structural summaries of spherical diagrams. | Pickles in `data/` produced by notebook 01. |
| `03_training_history_comparison_Nfixed8_simple` (`main`) | Compares six fixed-N runs and generates random-action baselines. Does not load checkpoints. | The six `results/*/train_log.npz` files listed above. |
| `04_inference_A_B_timesteps30` (`main`) | Evaluates the A/A and B/B policies on 500 fresh episodes against greedy and uniform-random baselines. Frozen checkpoints, no training. | `results/A_seed0_Nfixed8/checkpoint_latest.pt` and `results/B_seed0_Nfixed8/checkpoint_latest.pt`. Outputs go to `results/inference_A_B_timesteps30/`. |

The curriculum versions of notebooks 03 and 04 live on the `variable-N` branch.

## Running the notebooks without retraining
 
`results/` is ignored by Git, but this branch tracks the eight files that notebooks 03 and 04 need, so they run on a fresh clone without retraining:
 
```
results/A_seed0_Nfixed8/checkpoint_latest.pt
results/A_seed0_Nfixed8/train_log.npz
results/A_seed0_steps45_Nfixed8/train_log.npz
results/A_seed0_steps60_Nfixed8/train_log.npz
results/B_seed0_Nfixed8/checkpoint_latest.pt
results/B_seed0_Nfixed8/train_log.npz
results/B_seed0_steps45_Nfixed8/train_log.npz
results/B_seed0_steps60_Nfixed8/train_log.npz
```
 
Everything else under `results/` is regenerated by the notebooks and stays untracked. If you retrain into the same directories, these files are overwritten; use another `--checkpoint-dir` to keep them.
 
Notebook 02 is optional and is not runnable on a fresh clone: it reads the representatives that notebook 01 saves in `data/`, so run notebook 01 first (`data/` is not tracked; notebook 01 creates the folder if it does not exist).

## Reproducibility notes

- Code, environment and randomness are the three controlled ingredients: the code is versioned in Git, the environment is pinned in `environment.yml`, and PPO and environment randomness flow from explicit seeds.
- GPU and library versions can still affect bit-for-bit reproducibility, and all reported results are single-seed stochastic estimates. Compare curves and confidence intervals rather than expecting identical numbers.
- To reproduce the curriculum experiments, switch to `variable-N` and follow that branch's own scripts and notebooks; do not substitute the fixed-N commands.
