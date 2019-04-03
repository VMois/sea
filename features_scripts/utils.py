import os
import librosa
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.utils import shuffle


def extract_features(filename_path: str, offset: float, duration: float, sample_rate: int):
    x, sample_rate = librosa.load(filename_path,
                                  res_type='kaiser_fast',
                                  duration=duration,
                                  sr=sample_rate,
                                  offset=offset)
    sample_rate = np.array(sample_rate)

    # Extract audio features by taking an arithmetic mean of 13 identified MFCCs.
    # https://librosa.github.io/librosa/generated/librosa.feature.mfcc.html
    result = np.mean(librosa.feature.mfcc(y=x, sr=sample_rate, n_mfcc=13), axis=0)
    if len(result) != 216:
        return None
    return result


def save_dataframe(df, purpose: str, dataset_name: str):
    '''
    :param df: Dataframe to save
    :param purpose: train or test
    :param dataset_name: unique name of the saved dataset
    :return: None
    '''
    dest_path = 'data/{0}/{1}.parquet'.format(purpose, dataset_name)
    if os.path.exists(dest_path):
        os.remove(dest_path)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, dest_path)


def separate_dataframe_on_train_and_test(df):
    # TODO: Improve separation by label tag
    # https://stackoverflow.com/questions/24147278/how-do-i-create-test-and-train-samples-from-one-dataframe-with-pandas
    df = shuffle(df)
    msk = np.random.rand(len(df)) < 0.8
    train_data = df[msk]
    test_data = df[~msk]
    return train_data, test_data
