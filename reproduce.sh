# Usage: bash reproduce.sh
set -e

# 0) create env (first time only)
#   conda env create -f environment.yml
#   conda activate repurpose-bc

# 1) run the numbered scripts (adjust paths if needed)
python scripts/01_make_patient_degs.py        --config data/config_example.yaml
python scripts/02_make_reversal_scores.py     --config data/config_example.yaml
python scripts/03_make_network_proximity.py   --config data/config_example.yaml
python scripts/04_make_overlap_features.py    --config data/config_example.yaml
python scripts/05_make_baseline_prior.py      --config data/config_example.yaml
python scripts/06_eval_lopo_baseline.py       --config data/config_example.yaml
python scripts/07_rerank_elasticnet_lopo.py   --config data/config_example.yaml
python scripts/08_blend_sweep.py              --config data/config_example.yaml
python scripts/09_weight_sweep_donor.py       --config data/config_example.yaml
python scripts/10_make_figures.py             --config data/config_example.yaml
