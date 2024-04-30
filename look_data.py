import pandas as pd
import pprint
pp = pprint.PrettyPrinter(indent=4)

df = pd.read_parquet('../raw_data/training.parquet.snappy')

print(df.shape)
pp.pprint(df.head)
pp.pprint(df[0:10]['ecg_path'])

with open('output.txt', 'w') as f:
    f.write(df.head().to_string())