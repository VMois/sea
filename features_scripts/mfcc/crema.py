import os
import pandas as pd
from features_scripts.utils import save_dataframe, extract_mfcc_features
from features_scripts.mfcc.config import COLUMNS


def label_to_emotion(label: str):
    emotions = {
        'NEU': 'neutral',
        'HAP': 'happy',
        'SAD': 'sad',
        'ANG': 'angry',
        'FEA': 'fear',
        'DIS': 'disgust',
    }
    return emotions[label]


def crema_extract(dataset_id: int):
    features_df = pd.DataFrame(columns=COLUMNS)
    source_dir = 'raw-data/AudioMP3'
    actors_df = pd.read_csv('raw-data/crema_actors.csv')
    for index, filename in enumerate(os.listdir(source_dir)):
        if not filename.endswith('.mp3'):
            continue
        filename_no_ext = filename.split('.')[0]

        filename_split = filename_no_ext.split('_')
        emotion = label_to_emotion(filename_split[2])
        actor_id = int(filename_split[0])

        feature = extract_mfcc_features(os.path.join(source_dir, filename),
                                        offset=0,
                                        duration=2,
                                        sample_rate=22050 * 2)
        gender = str(actors_df[actors_df['ActorID'] == actor_id].iloc[0]['Sex']).lower()

        features_df = features_df.append({
            'id': int(f'{dataset_id}{index}'),
            'filename': filename,
            'emotion': emotion,
            'gender': gender,
            'features': feature,
            'actor_id': int(f'{dataset_id}{actor_id}'),
        }, ignore_index=True)

    save_dataframe(features_df, 'crema', 'mfcc')
