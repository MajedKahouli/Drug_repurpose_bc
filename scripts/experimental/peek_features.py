import sys, pandas as pd
p = sys.argv[1]
df = pd.read_parquet(p) if p.endswith('.parquet') else pd.read_csv(p, sep='\t')
print(df.describe(include='all').T)
print(df.head(10))
