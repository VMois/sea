'''
BME AI - Speech Emotion Analysis - SAVEE Extraction script
------------------------------------------------------------
'''
import os.path
import pandas as pd
import logging
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


def savee_extract(dataset_id: int):
    allowed_emotions = [1, 3, 4, 5, 6]
    actors = ['DC', 'JE', 'JK', 'KL']

    columns_list = ['id', 'filename', 'gender', 'emotion', 'features', 'actor_id']
    features_df = pd.DataFrame(columns=columns_list)
    map_emo = {
        'n': 1,
        'h': 3,
        'sa': 4,
        'a': 5,
        'f': 6,
        'd': 7,
        'su': 8,
    }
    audio_id = 0
    for index, actor in enumerate(actors):
        root = os.path.join('raw-data/AudioData', actor)
        actor_id = int(f'{dataset_id}{index + 1}')
        files = os.listdir(root)
        for filename in files:
            if not filename.endswith('.wav'):
                continue
            filename_no_ext = filename.split('.')[0]

            identifiers = filename_no_ext[0:len(filename_no_ext) - 2]
            emotion = int(map_emo[str(identifiers)])
            gender = 'male'

            if emotion not in allowed_emotions:
                continue

            feature = extract_mfcc_features(os.path.join(root, filename),
                                            offset=0.5,
                                            duration=2,
                                            sample_rate=22050 * 2)

            features_df = features_df.append({
                'id': int(f'{dataset_id}{audio_id}'),
                'filename': filename,
                'emotion': label_to_emotion(emotion),
                'gender': gender,
                'actor_id': actor_id,
                'features': feature,
            }, ignore_index=True)

            audio_id += 1
    save_dataframe(features_df, 'savee', 'mfcc')
    logging.info(features_df.head())
    logging.info(f'Successfully saved {len(features_df)} audio files for SAVEE')
