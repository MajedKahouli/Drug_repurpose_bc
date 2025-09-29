#!/usr/bin/env python3
import argparse
import os
import glob
import re
import numpy as np
import pandas as pd
from scipy.stats import zscore

FEATURE_CHOICES = {'overlap_count','jaccard','weighted_overlap'}


def z_per_patient(df, cols):
    out = df.copy()
    for c in cols:
        if c in out.columns:
            vals = out[c].astype(float).to_numpy()
            if np.all(np.isnan(vals)):
                out[c + '_z'] = np.nan
            else:
                mu = np.nanmean(vals)
                sd = np.nanstd(vals)
                if sd == 0 or not np.isfinite(sd):
                    out[c + '_z'] = 0.0
                else:
                    out[c + '_z'] = (vals - mu) / sd
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features-dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--alpha', type=float, default=0.6)
    ap.add_argument('--beta', type=float, default=0.2)
    ap.add_argument('--gamma', type=float, default=0.2)
    ap.add_argument('--overlap-feature', default='jaccard', choices=list(FEATURE_CHOICES))
    ap.add_argument('--tsv', action='store_true', help='Read features as TSV instead of Parquet')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    feat_files = sorted(glob.glob(os.path.join(args.features_dir, 'features_*.parquet')))

    if args.tsv:
        feat_files = sorted(glob.glob(os.path.join(args.features-dir, 'features_*.tsv')))

    for ff in feat_files:
        pid = re.search(r'features_(.+)\.(parquet|tsv)$', os.path.basename(ff)).group(1)
        if ff.endswith('.parquet'):
            df = pd.read_parquet(ff)
        else:
            df = pd.read_csv(ff, sep='\t')

        
        # The above line is awkward due to hyphen; do explicitly:
        df = z_per_patient(df, ['reversal_score'])
        df = z_per_patient(df, [args.overlap_feature])

        # Proximity: convert to higher-is-better, then z
        df['prox_score'] = -df['prox_z']
        df = z_per_patient(df, ['prox_score'])

        # Blend
        a, b, g = args.alpha, args.beta, args.gamma
        total = a + b + g
        if total == 0:
            a = 0.6; b = 0.2; g = 0.2; total = 1.0
        a, b, g = a/total, b/total, g/total
        df['baseline_score'] = (
            a * df['reversal_score_z'].fillna(0) +
            b * df[f'{args.overlap_feature}_z'].fillna(0) +
            g * df['prox_score_z'].fillna(0)
        )

        out_cols = ['drug', 'reversal_score', args.overlap_feature, 'prox_z', 'baseline_score']
        out_df = df[out_cols].copy()
        out_df = out_df.sort_values('baseline_score', ascending=False)
        out_path = os.path.join(args.out, f'baseline_{pid}.tsv')
        out_df.to_csv(out_path, sep='\t', index=False)
        print(f'Wrote {out_path} ({len(out_df)} rows)')

if __name__ == '__main__':
    main()
