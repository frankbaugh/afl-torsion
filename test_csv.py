import pickle
import pandas as pd

CSV_PATH = '/rds/project/rds-a1NGKrlJtrw/dyna_mol/pdb_1024_processed_trunc.csv'

df = pd.read_csv(CSV_PATH)

for column in df.columns:
    print(f"Column: {column}")
    print(df[column].head(10))
    print("\n")
