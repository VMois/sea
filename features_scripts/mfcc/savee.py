'''
BME AI - Speech Emotion Analysis - SAVEE Extraction script
------------------------------------------------------------
'''
import os.path
import pandas as pd
from features_scripts.utils import save_dataframe, extract_mfcc_features
from features_scripts.mfcc.config import COLUMNS


def savee_extract(dataset_id: int):
    actors = ['DC', 'JE', 'JK', 'KL']

    features_df = pd.DataFrame(columns=COLUMNS)
    map_emo = {
        'n': 'neutral',
        'h': 'happy',
        'sa': 'sad',
        'a': 'angry',
        'f': 'fear',
        'd': 'disgust',
        'su': 'surprise',
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
            emotion = map_emo[str(identifiers)]
            gender = 'male'

            feature = extract_mfcc_features(os.path.join(root, filename),
                                            offset=0.5,
                                            duration=2,
                                            sample_rate=22050 * 2)

            features_df = features_df.append({
                'id': int(f'{dataset_id}{audio_id}'),
                'filename': filename,
                'emotion': emotion,
                'gender': gender,
                'actor_id': actor_id,
                'features': feature,
            }, ignore_index=True)

            audio_id += 1
    save_dataframe(features_df, 'savee', 'mfcc')
