'''
BME AI - Speech Emotion Analysis - RAVDESS Extraction script
------------------------------------------------------------
'''
import os.path
import zipfile
import shutil
import logging
import pandas as pd
from .utils import save_dataframe, extract_mfcc_features

logging.basicConfig(level=logging.INFO)


def label_to_emotion(label: int):
    emotions = {
        1: 'neutral',
        2: 'calm',
        3: 'happy',
        4: 'sad',
        5: 'angry',
        6: 'fearful',
        7: 'disgust',
        8: 'surprise',
    }
    return emotions[label]


def ravdess_extract(dataset_id: int):
    required_zip_filenames = ['Audio_Speech_Actors_01-24.zip', 'Audio_Song_Actors_01-24.zip']
    allowed_emotions = [2, 3, 4, 5, 6]

    for filename in required_zip_filenames:
        if not os.path.isfile('raw-data/{0}'.format(filename)):
            print(
                'Please download Audio_Speech_Actors_01-24.zip '
                'and Audio_Song_Actors_01-24.zip from https://zenodo.org/record/1188976'
            )
            print('Place these files in a folder called raw-data/ in the main directory.')
            return

    dest_dir = 'raw-data/ravdess'
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    else:
        shutil.rmtree(dest_dir)
        os.makedirs(dest_dir)

    # Unzip the files above into raw-data/ravdess
    for zip_filename in required_zip_filenames:
        with zipfile.ZipFile(os.path.join('raw-data/', zip_filename)) as zip_file:
            for member in zip_file.namelist():
                filename = os.path.basename(member)
                if not filename:
                    continue

                # copy file (taken from zipfile's extract)
                source = zip_file.open(member)
                target = open(os.path.join(dest_dir, filename), 'wb')
                with source, target:
                    shutil.copyfileobj(source, target)

    columns_list = ['id', 'filename', 'gender', 'emotion', 'features', 'actor_id']
    features_df = pd.DataFrame(columns=columns_list)
    for index, filename in enumerate(os.listdir(dest_dir)):
        if not filename.endswith('.wav'):
            continue

        filename_no_ext = filename.split('.')[0]
        identifiers = filename_no_ext.split('-')
        emotion = int(identifiers[2])
        actor_id = int(identifiers[6])
        gender = 'male' if actor_id % 2 == 1 else 'female'

        if emotion not in allowed_emotions:
            continue

        feature = extract_mfcc_features(os.path.join(dest_dir, filename),
                                        offset=0.5,
                                        duration=2,
                                        sample_rate=22050 * 2)

        features_df = features_df.append({
            'id': int(f'{dataset_id}{index}'),
            'filename': filename,
            'emotion': label_to_emotion(emotion),
            'gender': gender,
            'features': feature,
            'actor_id': int(f'{dataset_id}{actor_id}'),
        }, ignore_index=True)

    save_dataframe(features_df, 'ravdess', 'mfcc')
    logging.info(features_df.head())
    logging.info(f'Successfully saved {len(features_df)} audio files for RAVDESS')
    shutil.rmtree(dest_dir)
