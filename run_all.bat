@echo off
REM run_all.bat
REM Create/activate env manually the first time:
REM   conda env create -f environment.yml
REM   conda activate repurpose-bc

for %%S in (
  scripts\01_make_patient_degs.py
  scripts\02_make_reversal_scores.py
  scripts\03_make_network_proximity.py
  scripts\04_make_overlap_features.py
  scripts\05_make_baseline_prior.py
  scripts\06_eval_lopo_baseline.py
  scripts\07_rerank_elasticnet_lopo.py
  scripts\08_blend_sweep.py
  scripts\09_weight_sweep_donor.py
  scripts\10_make_figures.py
) do (
  echo Running %%S ...
  python %%S --config data\config_example.yaml || goto :eof
)
