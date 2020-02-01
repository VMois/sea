import os
import pandas as pd
from features_scripts.utils import extract_mel_spec_as_image


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
    source_dir = 'raw-data/AudioMP3'
    actors_df = pd.read_csv('raw-data/crema_actors.csv')
    for index, filename in enumerate(os.listdir(source_dir)):
        if not filename.endswith('.mp3'):
            continue
        filename_no_ext = filename.split('.')[0]

        filename_split = filename_no_ext.split('_')
        emotion = label_to_emotion(filename_split[2])
        actor_id = int(filename_split[0])
        gender = str(actors_df[actors_df['ActorID'] == actor_id].iloc[0]['Sex']).lower()
        filepath = os.path.join(source_dir, filename)
        dst_path = os.path.join('img', f'{gender}_{emotion}_{dataset_id}{index}.png')
        extract_mel_spec_as_image(filepath, dst_path)
