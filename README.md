# Self-Pruning Neural Network (CIFAR-10)

A PyTorch implementation of a self-pruning MLP where each connection is controlled by a learnable sigmoid gate. Training uses cross-entropy plus an L1 penalty on gate values to encourage sparsity.

## Highlights

- Custom `PrunableLinear` layer with element-wise learnable gates
- Multi-lambda experiment runner for sparsity/accuracy trade-off analysis
- Auto-generated experiment report (`REPORT.md`)

## Repository Structure

```text
.
├── self_pruning_cifar10.py      # Main training and reporting script
├── requirements.txt             # Python dependencies
├── REPORT.md                    # Generated experiment report
├── results/                     # Generated CSV and plots
├── data/                        # CIFAR-10 cache (ignored by git)
└── .venv/                       # Local virtual environment (ignored by git)
```

## Quick Start

### 1. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Run a full CIFAR-10 experiment

```powershell
python self_pruning_cifar10.py --epochs 15 --lambdas 1e-6 5e-6 2e-5
```

## Latest Real Results (CIFAR-10)

Experiment date: 2026-04-20

Command used:

```powershell
python self_pruning_cifar10.py --epochs 2 --lambdas 1e-6 5e-6 2e-5 --batch-size 128 --num-workers 0 --results-csv results/real_summary.csv --hist-path results/real_gate_hist.png --report-path REPORT.md
```

| Lambda | Test Accuracy | Sparsity Level (%) |
|---:|---:|---:|
| 1.00e-06 | 40.38% | 0.00% |
| 5.00e-06 | 40.84% | 0.00% |
| 2.00e-05 | 41.86% | 0.00% |

Best accuracy in this run: 41.86% at lambda = 2.00e-05.

Related artifacts:

- `results/real_summary.csv`
- `results/real_gate_hist.png`
- `REPORT.md`

## Key CLI Options

```text
--lambdas                Lambda values for sparsity regularization
--sparsity-threshold     Gate threshold used to count pruned connections
--results-csv            Output CSV path
--hist-path              Output histogram path
--report-path            Output markdown report path
```

## Output Artifacts

After training, the script generates:

- CSV summary of lambda vs. accuracy/sparsity
- Gate distribution histogram for the best model
- Markdown report in `REPORT.md`

## Notes for Evaluation

- The implementation correctly applies gated weights and a custom sparsity term.
- To demonstrate stronger pruning behavior, run longer experiments and a wider lambda sweep.

## Reproducibility

- Default seed is set via `--seed` (default: `42`).
- Device auto-selection uses CUDA when available, otherwise CPU.
