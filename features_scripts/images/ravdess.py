import os.path
import zipfile
import shutil
from features_scripts.utils import extract_mel_spec_as_image


def label_to_emotion(label: int):
    emotions = {
        1: 'neutral',
        2: 'neutral',  # originally "calm"
        3: 'happy',
        4: 'sad',
        5: 'angry',
        6: 'fear',
        7: 'disgust',
        8: 'surprise',
    }
    return emotions[label]


def ravdess_extract(dataset_id: int):
    required_zip_filenames = ['Audio_Speech_Actors_01-24.zip', 'Audio_Song_Actors_01-24.zip']

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

    for index, filename in enumerate(os.listdir(dest_dir)):
        if not filename.endswith('.wav'):
            continue

        filename_no_ext = filename.split('.')[0]
        identifiers = filename_no_ext.split('-')
        emotion = int(identifiers[2])
        emotion_label = label_to_emotion(emotion)
        actor_id = int(identifiers[6])
        gender = 'male' if actor_id % 2 == 1 else 'female'
        filepath = os.path.join(dest_dir, filename)
        dst_path = os.path.join('img', f'{gender}_{emotion_label}_{dataset_id}{index}.png')
        extract_mel_spec_as_image(filepath, dst_path)
    shutil.rmtree(dest_dir)
