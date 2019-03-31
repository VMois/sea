'''
BME AI - Speech Emotion Analysis - RAVDESS Extraction script
------------------------------------------------------------

RAVDESS filename identifiers:
  
  Modality (01 = full-AV, 02 = video-only, 03 = audio-only).
  Vocal channel (01 = speech, 02 = song).
  Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
  Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the 'neutral' emotion.
  Statement (01 = 'Kids are talking by the door', 02 = 'Dogs are sitting by the door').
  Repetition (01 = 1st repetition, 02 = 2nd repetition).
  Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).

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
    allowed_emotions = [1, 3, 4, 5, 6, 7, 8]

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

    for root, dirs, files in os.walk('raw-data/savee'):
        for file in files:
            if not file.endswith('.wav'):
                continue
            filename_no_ext = file.split('.')[0]
            map_emo = {
                    'n': 1,
                    'h': 3,
                    'sa': 4,
                    'a': 5,
                    'f': 6,
                    'd': 7,
                    'su': 8,
                }
            identifiers=filename_no_ext[0:len(filename_no_ext)-2]
            emotion = int(map_emo[str(identifiers)])
            gender = 'male'

            if emotion not in allowed_emotions:
                continue

            # Sample rate: 44,100 Hz
            # Duration: 2.5 seconds
            # Skip time: 0.5 seconds from the beginning
            feature = extract_features(os.path.join(root, file),
                                       offset=0.5,
                                       duration=2.5,
                                       sample_rate=22050 * 2)

            features_df = features_df.append({
                'filename': file,
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
