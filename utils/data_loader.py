import os
import pandas as pd
import pyarrow.parquet as pq


def load_data(datasets, table):
    loaded_data = []
    for dataset in datasets:
        table_path = os.path.join('data', dataset, f'{table}.parquet')
        loaded_data.append(pq.read_pandas(table_path).to_pandas())
    return pd.concat(loaded_data, ignore_index=True)
