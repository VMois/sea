'''
BME AI - Speech Emotion Analysis - SAVEE Extraction script
------------------------------------------------------------
'''
import os.path
import zipfile
import shutil
import pandas as pd
from .utils import save_dataframe, \
    separate_dataframe_on_train_and_test, extract_features


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


def savee_extract():
    required_zip_filenames = ['AudioData.zip']
    allowed_emotions = [2, 3, 4, 5, 6]

    for filename in required_zip_filenames:
        if not os.path.isfile('raw-data/{0}'.format(filename)):
            print(
                'Please download AudioData.zip '
            )
            print('Place these files in a folder called raw-data/ in the main directory.')
            return

    if not os.path.exists('raw-data/savee'):
        os.makedirs('raw-data/savee')
    else:
        shutil.rmtree('raw-data/savee')
        os.makedirs('raw-data/savee')

    # Unzip the files above into raw-data/ravdess
    for zip_filename in required_zip_filenames:
        zip_ref = zipfile.ZipFile('raw-data/{0}'.format(zip_filename), 'r')
        zip_ref.extractall('raw-data/savee')
        zip_ref.close()

    columns_list = ['filename', 'gender', 'emotion', 'features']
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
    for root, dirs, files in os.walk('raw-data/savee'):
        for filename in files:
            if not filename.endswith('.wav'):
                continue
            filename_no_ext = filename.split('.')[0]

            identifiers = filename_no_ext[0:len(filename_no_ext)-2]
            emotion = int(map_emo[str(identifiers)])
            gender = 'male'

            if emotion not in allowed_emotions:
                continue

            # Sample rate: 44,100 Hz
            # Duration: 2.5 seconds
            # Skip time: 0.5 seconds from the beginning
            feature = extract_features(os.path.join(root, filename),
                                       offset=0.5,
                                       duration=2.5,
                                       sample_rate=22050 * 2)

            features_df = features_df.append({
                'filename': filename,
                'emotion': label_to_emotion(emotion),
                'gender': gender,
                'features': feature,
            }, ignore_index=True)

    train_df, test_df = separate_dataframe_on_train_and_test(features_df)

    save_dataframe(train_df, 'train', 'savee')
    save_dataframe(test_df, 'test', 'savee')
    print('Successfully saved', len(features_df), 'audio files for savee')
    print('- train data: ', len(train_df))
    print('- test data: ', len(test_df))

    shutil.rmtree('raw-data/savee')
