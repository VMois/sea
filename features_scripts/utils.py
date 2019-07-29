import os
import logging
import librosa
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def extract_mfcc_features(filename_path: str, offset: float, duration: float, sample_rate: int):
    x, sample_rate = librosa.load(filename_path,
                                  res_type='kaiser_fast',
                                  duration=duration,
                                  sr=sample_rate,
                                  offset=offset)
    sample_rate = np.array(sample_rate)

    # Extract audio features by taking an arithmetic mean of 13 identified MFCCs.
    # https://librosa.github.io/librosa/generated/librosa.feature.mfcc.html
    return np.mean(librosa.feature.mfcc(y=x, sr=sample_rate, n_mfcc=13), axis=0)


def save_dataframe(df: pd.DataFrame, dataset_name: str, purpose: str):
    '''
    :param df: Dataframe to save
    :param dataset_name: unique name of the saved dataset
    :param purpose: purpose of file
    :return: None
    '''
    root = os.path.join('data', dataset_name)
    os.makedirs(root, exist_ok=True)
    dest_path = os.path.join(root, '{0}.parquet'.format(purpose))
    if os.path.exists(dest_path):
        os.remove(dest_path)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, dest_path)
    logging.info(f'Saved {len(df)} records for {dataset_name} in {purpose}')
    logging.info(df.head())
