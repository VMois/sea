import os
import matplotlib.pyplot as plt
from features_scripts.utils import extract_mel_spec_as_image

def savee_extract(dataset_id: int):
    actors = ['DC', 'JE', 'JK', 'KL']
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
        files = os.listdir(root)
        for filename in files:
            if not filename.endswith('.wav'):
                continue
            filename_no_ext = filename.split('.')[0]

            identifiers = filename_no_ext[0:len(filename_no_ext) - 2]
            emotion = map_emo[str(identifiers)]
            gender = 'male'
            filepath = os.path.join(root, filename)
            dest_path = os.path.join('img', f'{gender}_{emotion}_{dataset_id}{audio_id}.png')
            extract_mel_spec_as_image(filepath, dest_path)
            audio_id += 1
