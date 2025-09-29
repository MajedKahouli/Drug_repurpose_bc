# Drug_repurpose_bc

Patient-specific drug repurposing for breast cancer. Reproduces per-patient drug rankings by integrating:
- LINCS L1000 signature reversal (cosine)
- STRING v11 network proximity (z-score)
- Target–signature overlap (Jaccard)

Validated with strict leave-one-patient-out (LOPO). Includes scripts to build features, run baselines, re-rank with Elastic Net, and generate figures/tables.

## Quick start
```bash
conda env create -f env.yml
conda activate repurpose-gpu
python scripts/04_build_features_gpu.py --input data/raw --out work/features_gpu
python scripts/04a_baseline_blend.py --features work/features_gpu --out work/baseline
python scripts/06_validate_rankings.py --rankings work/baseline --approved data/validation/breast_cancer_drugs.tsv
