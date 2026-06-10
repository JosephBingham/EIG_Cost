# EIG-Cost: Cost-Aware Active Feature Acquisition for Clinical Differential Diagnosis

Code and benchmark for the paper:

> **EIG-Cost: Cost-Aware Active Feature Acquisition for Clinical Differential Diagnosis**  
> Joseph Bingham - Faculty of Biology, Technion – Israel Institute of Technology
> Netanel Arussy — Faculty of Computer Science, Technion – Israel Institute of Technology  
> *Artificial Intelligence in Medicine* (under review)

EIG-Cost selects the next diagnostic test panel by maximising Expected Information Gain penalised by real 2026 Medicare CPT costs:

```
score(j) = EIG(j | x_obs) - λ · cost(j)
```

It is evaluated against nine AFA baselines on a 202,971-patient MIMIC-IV benchmark spanning 21 acute conditions.

---

## Requirements

- Python 3.12 (managed via [uv](https://github.com/astral-sh/uv))
- MIMIC-IV v2.2 access ([PhysioNet credentialing required](https://physionet.org/content/mimiciv/))

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create environment and install dependencies
uv sync
```

All pipeline commands use `uv run` — do not use the system Python.

---

## Data Setup

**1. Obtain MIMIC-IV access** at [PhysioNet](https://physionet.org/content/mimiciv/).

**2. Build the 21-condition cohort:**
```bash
python3 scripts/cohort/build_cohort.py \
    --mimic-dir /path/to/mimic-iv \
    --out extra/data/mimic_iv
```

**3. Build the 5-condition subset** (DKA, MI, pancreatitis, sepsis, stroke):
```bash
python3 scripts/cohort/build_cohort.py \
    --mimic-dir /path/to/mimic-iv \
    --out extra/data/mimic_iv_5class \
    --conditions diabetic_emergency myocardial_infarction pancreatitis sepsis stroke
```

Expected outputs: `X_train/val/test.npy`, `y_train/val/test.npy`, `costs.npy`, `groups.json`, `metadata.json` in each data directory.

---

## Running the Pipeline

The full pipeline (train all methods → evaluate → aggregate → plot) is orchestrated with Snakemake + Hydra:

```bash
nohup ./run.sh > pipeline.log 2>&1 &
echo "PID: $!"
```

Monitor progress:
```bash
grep "steps.*done" pipeline.log | tail -5
```

Training EIG-Cost alone (one bundle, one budget):
```bash
uv run python scripts/train/eig_cost.py \
    train_dataset_bundle_path=extra/output/datasets/mimic_iv/0/train.bundle \
    val_dataset_bundle_path=extra/output/datasets/mimic_iv/0/val.bundle \
    classifier_bundle_path=extra/output/trained_classifiers/initializer-cold/dataset-mimic_iv.bundle \
    save_path=extra/output/trained_methods/.../method.bundle \
    hard_budget=30 device=cpu seed=0 \
    experiment@_global_=mimic_iv
```

---

## Key Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `λ` (cost penalty) | 0.005 | Penalises \$1 cost ≈ 5×10⁻³ nats EIG |
| `n_mc_samples` | 50 | Monte Carlo samples per candidate panel |
| Classifier hidden dims | [256, 128, 64] | MaskableClassifier MLP |
| CVAE latent dim | 64 | TabularMAE latent space |
| Classifier epochs | 50 | Adam, lr=1e-3, batch=256 |
| CVAE epochs | 30 | Adam, lr=5e-4, batch=256 |

---

## Results Summary

**21-condition cohort, budget \$30** (mean ± SD, 3 seeds):

| Method | Macro-F1 | Accuracy | Vol. Stop |
|--------|----------|----------|-----------|
| **EIG-Cost (ours)** | **0.180 ± 0.012** | **0.265 ± 0.010** | 0.0% |
| AACO-NN | 0.169 ± 0.011 | 0.244 ± 0.009 | 0.0% |
| AACO | 0.162 ± 0.009 | 0.233 ± 0.008 | 0.0% |
| DIME | 0.027 ± 0.002 | 0.103 ± 0.002 | 96.9% |
| OL (w/ mask) | 0.018 ± 0.001 | 0.099 ± 0.001 | 100.0% |

Seven of nine baselines exhibit ≥87% voluntary stop rates (*baseline collapse*) — they predict via class priors without acquiring any features.

Full results in `extra/output/analysis/`.

---

## Repository Structure

```
AFA-Benchmark/
├── afabench/
│   ├── eig_cost/           # EIG-Cost method (afa_method.py, models.py)
│   └── common/             # Shared utilities, registry, bundle I/O
├── scripts/
│   ├── train/eig_cost.py   # Training entry point (Hydra)
│   └── eval/               # Evaluation scripts
├── extra/
│   ├── conf/               # Hydra configs (EIG-Cost + all baselines)
│   ├── data/               # Cohort data (not tracked — requires MIMIC-IV)
│   ├── output/             # Pipeline outputs (not tracked)
│   └── workflow/           # Snakemake orchestration
├── run.sh                  # Pipeline entry point
└── paper/                  # LaTeX source, figures, references
```

---

## Costs

Acquisition costs are derived from the 2026 Medicare Physician Fee Schedule (CPT codes). The cost vector is stored in `extra/data/mimic_iv/costs.npy` (shape: `[55]`, units: USD). Total full-panel cost: \$408.80.

---

## Citation

```bibtex
@article{Bingham2025eigcost,
  author  = {Bingham, Joseph},
  title   = {{EIG-Cost}: Cost-Aware Active Feature Acquisition for
             Clinical Differential Diagnosis},
  journal = {--},
  year    = {2026},
  note    = {Under review}
}
```

---

## License

Code: MIT. Data: subject to [MIMIC-IV Data Use Agreement](https://physionet.org/content/mimiciv/view-license/).
