# AI Testing in Flag Manifolds — `variable-N` branch (curriculum over N)

Final project for MM845 (*Topics in Geometry III: AI for Geometry*) at the University of Campinas (UNICAMP).

The project studies when a labelled Coxeter diagram is **spherical** (equivalently, when its Cartan/Gram matrix is positive definite, i.e. it is a Dynkin diagram) and asks whether a reinforcement-learning agent can turn a random diagram into a spherical one by local edge edits, using only the definition of sphericity and none of Dynkin's necessary conditions (no cycles, bounded degree, arm lengths, ...).

**This is the `variable-N` branch: PPO trained with a curriculum over the number of vertices N.** The fixed-N experiments (training directly at N = 8, with no curriculum) live on `main`; see [Branches](#branches).

It combines:

1. an **exhaustive search** over all labelled diagrams up to N = 6, exploiting permutation symmetry;
2. a **PPO agent with a hand-written graph neural network** (PyTorch only, no GNN library), trained with a curriculum over N (stages `{3,4,5}` → `{3,…,6}` → `{3,…,7}` → `{3,…,8}`);
3. **baselines** (uniform-random, and greedy decoding of the trained policy) and notebooks to analyse everything.

## Contents

- [Problem setup](#problem-setup)
- [Curriculum over N](#curriculum-over-n)
- [Repository layout](#repository-layout)
- [Branches](#branches)
- [Environment setup](#environment-setup)
- [Quick sanity checks](#quick-sanity-checks)
- [Training PPO](#training-ppo)
- [Evaluating a checkpoint](#evaluating-a-checkpoint)
- [Notebooks](#notebooks)
- [Running the notebooks without retraining](#running-the-notebooks-without-retraining)
- [Reproducibility notes](#reproducibility-notes)

## Problem setup

A diagram on N vertices is a symmetric integer matrix `M`. Off-diagonal entries (labels) lie in `{2, 3, 4, 6}`, where `2` means *no edge*; the diagonal stores `0` and is unused. The Cartan matrix is `A_ii = 2`, `A_ij = -2 cos(pi / m_ij)`, and the diagram is spherical iff `A` is positive definite.

- **State:** the label matrix, padded to a fixed `N_MAX = 8` with isolated vertices when an episode uses fewer vertices.
- **Action:** one discrete choice of (unordered vertex pair, label), giving C(8,2) · 4 = 112 actions; actions touching padding vertices are masked.
- **Reward:** `tanh` of the smallest eigenvalue of `A` (the *spherical margin*) after each edit. An episode ends on success (spherical diagram) or after a fixed budget of edits (30 by default).
- **Datasets of initial diagrams:** every pair gets an i.i.d. label. **Dataset A** is uniform over `{none, 3, 4, 6}` (about 21 edges at N = 8). **Dataset B** has P(none) = 3/4 (about N − 1 = 7 edges, close to the tree regime); it is used only as a diagnostic, since it encodes the tree structure of the target.

## Curriculum over N

Each episode uses a number of real vertices `n_active ≤ 8`, sampled uniformly from the current curriculum stage; the remaining vertices are isolated padding. Padding does not change the spherical margin, so the same network and the same 112-action space serve every N.

| Stage | Sampled `n_active` |
| --- | --- |
| 0 | 3, 4, 5 |
| 1 | 3, 4, 5, 6 |
| 2 | 3, 4, 5, 6, 7 |
| 3 | 3, 4, 5, 6, 7, 8 |

The agent is promoted to the next stage when the rolling mean success rate over the last `--curriculum-window-updates` updates (default 20) reaches `--curriculum-success-threshold` (default 0.7), after at least `--curriculum-min-updates-per-stage` updates (default 20) in the current stage. Earlier sizes are never dropped. The stage reached is logged in `train_log.npz` and stored in each checkpoint.

## Repository layout

| Path | Contents |
| --- | --- |
| `environment.yml` | Conda environment specification (see [Environment setup](#environment-setup)). |
| `src/coxeter.py` | Diagram representation, Cartan matrix, spherical test and margin, graph utilities, permutation-symmetry tools. |
| `src/exhaustive.py` | Exhaustive search up to N = 6 over isomorphism classes, weighted by orbit size. |
| `src/datagen.py` | i.i.d. samplers for Datasets A and B and their structural statistics. |
| `src/env.py` | RL environment (single and vectorised), with padding for variable N. |
| `src/policy.py` | Actor-critic graph neural network, with masking of padding vertices and actions. |
| `src/train_ppo.py` | PPO training script with the curriculum, run from the terminal. |
| `src/analyze.py` | Checkpoint evaluation, Coxeter-type classification, training curves. |
| `notebooks/` | Benchmarks, structural analysis, training-history comparison, inference comparison. |
| `data/` | Representative diagrams produced by notebook 01 (input of notebook 02). |
| `results/` | Checkpoints, training logs and evaluation tables. Generated; mostly ignored by Git (see [below](#running-the-notebooks-without-retraining) for the files that are tracked). |

## Branches

| Branch | Focus |
| --- | --- |
| `main` | Fixed-N PPO (the script defaults to N = 8 and accepts `--n`) and its versions of notebooks 03 and 04. |
| `variable-N` (this branch) | Curriculum training over N and its own versions of notebooks 03 and 04. |

The training options and saved outputs differ between the branches. Before running an experiment, read that branch's `src/train_ppo.py` and the setup cells of its notebooks.

```bash
git switch variable-N   # curriculum experiments (this README)
git switch main         # fixed-N experiments
```

Files under `results/` come from different branches and runs; identify an experiment by its run directory name, the notebook's input paths and the `config` stored in each checkpoint, not by assuming everything belongs to one branch. On this branch the relevant run directories are `results/A_seed0/` and `results/B_seed0/`, whereas the fixed-N runs of `main` use names ending in `_Nfixed8`.

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

`src/train_ppo.py` uses a CUDA GPU automatically when one is available and runs on CPU otherwise. The model is small, so CPU training is fine. The inference notebook (04) runs on CPU (`DEVICE = "cpu"`).

Verify the installation:

```bash
python -c "import numpy, torch, pandas, matplotlib; print(numpy.__version__, torch.__version__)"
```

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

Training is a terminal script, never run from a notebook. Always use `python -m src.<module>` from the repository root, with this branch checked out. The two runs used by the notebooks write to separate directories:

```bash
git switch variable-N
python -m src.train_ppo --dataset A --seed 0 --total-timesteps 2000000 --checkpoint-dir results/A_seed0
python -m src.train_ppo --dataset B --seed 0 --total-timesteps 2000000 --checkpoint-dir results/B_seed0
```

The curriculum options are `--curriculum-success-threshold` (default 0.7), `--curriculum-window-updates` (20) and `--curriculum-min-updates-per-stage` (20). The vertex count is sampled uniformly from the current stage, and earlier stages are never dropped. Promotions are printed during training as `>>> curriculum: promoted to stage ...`.

### Notes

- For a short smoke test use `--total-timesteps 100000`. It checks that everything runs but does **not** reproduce the reported training horizon.
- The defaults use 16 environments and 64 rollout steps per update, so each update advances 1,024 steps; a request for 2,000,000 timesteps therefore stops at 1,999,872.
- Each run directory receives `checkpoint_latest.pt`, periodic `checkpoint_step_<steps>.pt` files (recovery checkpoints only) and `train_log.npz`. The log holds `timesteps`, `reward`, `margin`, `success_rate` and `stage` (the curriculum stage in force during each update).
- Change `--seed` for other seeds, and keep the run directory name in sync with the seed (for example `results/A_seed1`).

## Evaluating a checkpoint

```bash
python -m src.analyze --checkpoint results/A_seed0/checkpoint_latest.pt --n-episodes 500 --dataset A
python -m src.analyze --checkpoint results/B_seed0/checkpoint_latest.pt --n-episodes 500 --dataset B
```

By default only N = 8 is evaluated. Add `--active-ns 3 4 5 6 7 8` to evaluate several vertex counts at once, and `--deterministic` for argmax actions. The command prints the success rate, the mean number of edits to success and the Coxeter type of some successful diagrams, and, if `train_log.npz` sits beside the checkpoint, saves the training curves there.

## Notebooks

Start Jupyter from the repository root so relative paths resolve, select the project kernel, and run the cells from top to bottom:

```bash
jupyter lab
```

| Notebook | What it does | Inputs | Outputs |
| --- | --- | --- | --- |
| `01_exhaustive_search_benchmark` | Runs the exhaustive search, writes a summary CSV and optionally saves representative diagrams. N = 6 is heavy. | none | CSV and pickles in `data/` |
| `02_spherical_structure_analysis` | Structural summaries of spherical diagrams. | Pickles in `data/` produced by notebook 01. | figures |
| `03_training_history_comparison` | Compares the curriculum training histories of Datasets A and B: summary table of first, final and best reward, margin and success rate; training curves with stage changes marked; the early promotions on Dataset A; and the smoothed PPO margin against a uniform-random baseline computed separately for each curriculum stage. Does not load checkpoints or run the policy. | `results/A_seed0/train_log.npz`, `results/B_seed0/train_log.npz` | figures only |
| `04_model_inference_comparison` | Evaluates the frozen A- and B-trained models on fresh episodes for every N from 2 to 8, with stochastic and greedy decoding, on both datasets (conditions A/A, B/B, A/B, B/A; the first letter is the training dataset, the second the evaluation dataset). Reports success rate with 95% Wilson intervals, steps to success, connected success rate, number of distinct canonical classes up to vertex permutation, Coxeter types found, a drawing of every connected A/A stochastic class, and a model ranking per metric. No training. | `results/A_seed0/checkpoint_latest.pt`, `results/B_seed0/checkpoint_latest.pt` | CSV tables in `results/inference_evaluation_comparison/` |

Notes on notebook 04:

- The full protocol uses 100 episodes for each model × evaluation dataset × N × decoding mode, i.e. 2 × 2 × 7 × 2 × 100 = 5,600 rollouts, on CPU. For a quick check set `RUN_FULL_EVALUATION = False` in the configuration cell, which uses 2 episodes per combination instead.
- Episode seeds are derived deterministically from `BASE_SEED`, so reruns reproduce the same episodes.
- It writes `episode_metrics.csv`, `aggregate_metrics.csv`, `comparison_metrics.csv`, `model_ranking.csv`, `checkpoint_metadata.csv` and `aa_diagram_types.csv` to `results/inference_evaluation_comparison/`. These are regenerated by the notebook and are not tracked.

Notes on notebook 03:

- The random baseline is simulated on the fly (a uniform policy over valid actions, once per curriculum stage), with a seed different from the training seed on purpose; it needs no files.
- The training logs are plotted as logged; only the last figure uses a trailing moving average of 50 updates.

The fixed-N versions of notebooks 03 and 04 live on `main`.

## Running the notebooks without retraining

`results/` is ignored by Git, but this branch tracks the four files that notebooks 03 and 04 need, so they run on a fresh clone without retraining:

```
results/A_seed0/checkpoint_latest.pt    results/A_seed0/train_log.npz
results/B_seed0/checkpoint_latest.pt    results/B_seed0/train_log.npz
```

Everything else under `results/` is regenerated by the notebooks and stays untracked. If you retrain into the same directories, these files are overwritten; use another `--checkpoint-dir` to keep them.

Notebook 02 is optional and is not runnable on a fresh clone: it reads the representatives that notebook 01 saves in `data/`, so run notebook 01 first (`data/` is not tracked; notebook 01 creates the folder if it does not exist).

## Reproducibility notes

- Code, environment and randomness are the three controlled ingredients: the code is versioned in Git, the environment is pinned in `environment.yml`, and PPO and environment randomness flow from explicit seeds.
- GPU and library versions can still affect bit-for-bit reproducibility, and all reported results are single-seed stochastic estimates. Compare curves and confidence intervals rather than expecting identical numbers.
- The tracked checkpoints are the frozen models behind the reported curriculum results. To reproduce the fixed-N experiments instead, switch to `main` and follow its README; do not substitute these commands.
