'''

BME AI - Speech Emotion Analysis - Feature Extraction
-----------------------------------------------------
This script will automatically unzip, analyze, and save a pandas data frame containing features and metadata of the audio files in RAVDESS.

********* Instructions ************
1) Download Audio_Speech_Actors_01-24.zip and Audio_Song_Actors_01-24.zip from https://zenodo.org/record/1188976
2) Place these zip files in a folder called raw-data in the main directory.
3) The results are stored in the Apache Parquet format with gzip compression in a file called 'audio-features.parquet.gzip' located in the main directory.
***********************************

RAVDESS filename identifiers:
  
  Modality (01 = full-AV, 02 = video-only, 03 = audio-only).
  Vocal channel (01 = speech, 02 = song).
  Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
  Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the 'neutral' emotion.
  Statement (01 = 'Kids are talking by the door', 02 = 'Dogs are sitting by the door').
  Repetition (01 = 1st repetition, 02 = 2nd repetition).
  Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).

'''

# Import libraries.
import os.path
import zipfile
import shutil
import pandas as pd
import librosa
import numpy as np
import time

print(' ___ __  __ ___     _   ___ ')
print('| _ )  \/  | __|   /_\ |_ _|')
print('| _ \ |\/| | _|   / _ \ | | ')
print('|___/_|  |_|___| /_/ \_\___|')
print('                            ')
  
# Time script execution for performance evaluation.
start = time.time()

# Check if raw-data folder exists. If not, create it.
if not os.path.exists('raw-data'):
  os.makedirs('raw-data')

# Check if raw-data/audio-files folder exists. If not, create it.
if not os.path.exists('raw-data/audio-files'):
  os.makedirs('raw-data/audio-files')

# Check if RAVDESS zip files exist.
if os.path.isfile('raw-data/Audio_Speech_Actors_01-24.zip') and os.path.isfile('raw-data/Audio_Speech_Actors_01-24.zip'):
  
  # Unzip the files above into raw-data/audio-files
  print('Unzipping Audio_Speech_Actors_01-24.zip...')
  zip_ref = zipfile.ZipFile('raw-data/Audio_Speech_Actors_01-24.zip', 'r')
  zip_ref.extractall('raw-data/audio-files')
  zip_ref.close()
  print('Unzipping Audio_Speech_Actors_01-24.zip...')
  zip_ref = zipfile.ZipFile('raw-data/Audio_Speech_Actors_01-24.zip', 'r')
  zip_ref.extractall('raw-data/audio-files')
  zip_ref.close()
  print('Unzipping complete.')
  
  # Initiate a data frame to be filled with the following columns: filename, gender, emotion, and features.
  # The order of these columns is very important since we will be loading the data in this specific order.
  columnlist = ['filename','gender','emotion','features']
  features_dataframe = pd.DataFrame(columns=columnlist)
  index = 0
  
  # Loop through the extracted .wav audio files.
  for root, dirs, files in os.walk('raw-data/audio-files'):
    for file in files:
        if file.endswith('.wav'):
          print('Analyzing', file)
          
          # Extract audio features using LibROSA.
          features = []
          
          # Load two copies of the current audio file into the variables X and sample_rate.
          # Sample rate: 44,100 Hz
          # Duration: 2.5 seconds
          # Skip time: 0.5 seconds from the beginning
          # Window function: Kaiser https://en.wikipedia.org/wiki/Kaiser_window
          X, sample_rate = librosa.load(os.path.join(root, file), res_type='kaiser_fast', duration=2.5, sr=22050*2, offset=0.5)
          
          # Convert sample_rate into a numpy array.
          sample_rate = np.array(sample_rate)
          
          # Extract audio features by taking an arithmetic mean of 13 identified Mel-Frequency Cepstral Coefficients (MFCC).
          # https://librosa.github.io/librosa/generated/librosa.feature.mfcc.html
          features = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13), axis=0)
          
          # Identify emotion types and gender according to the RAVDESS filename standard.
          # Note that we are ignoring the neutral (01), disgust (07), and surprised (08) emotions.
          if file[6:-16]=='02' and int(file[18:-4])%2==0:
            features_dataframe.loc[index] = [file, 'female', 'calm', features]
            # Increment the data frame index for the next row only if a matching file was found.
            index += 1
          elif file[6:-16]=='02' and int(file[18:-4])%2==1:
            features_dataframe.loc[index] = [file, 'male', 'calm', features]
            index += 1
          elif file[6:-16]=='03' and int(file[18:-4])%2==0:
            features_dataframe.loc[index] = [file, 'female', 'happy', features]
            index += 1
          elif file[6:-16]=='03' and int(file[18:-4])%2==1:
            features_dataframe.loc[index] = [file, 'male', 'happy', features]
            index += 1
          elif file[6:-16]=='04' and int(file[18:-4])%2==0:
            features_dataframe.loc[index] = [file, 'female', 'sad', features]
            index += 1
          elif file[6:-16]=='04' and int(file[18:-4])%2==1:
            features_dataframe.loc[index] = [file, 'male', 'sad', features]
            index += 1
          elif file[6:-16]=='05' and int(file[18:-4])%2==0:
            features_dataframe.loc[index] = [file, 'female', 'angry', features]
            index += 1
          elif file[6:-16]=='05' and int(file[18:-4])%2==1:
            features_dataframe.loc[index] = [file, 'male', 'angry', features]
            index += 1
          elif file[6:-16]=='06' and int(file[18:-4])%2==0:
            features_dataframe.loc[index] = [file, 'female', 'fearful', features]
            index += 1
          elif file[6:-16]=='06' and int(file[18:-4])%2==1:
            features_dataframe.loc[index] = [file, 'male', 'fearful', features]
            index += 1

  print('Analysis complete.')
  
  # Delete the extracted audio files.
  shutil.rmtree('raw-data/audio-files')  

  # Save the data frame.
  if os.path.exists('audio-features.parquet.gzip'):
    os.remove('audio-features.parquet.gzip')
  features_dataframe.to_parquet('audio-features.parquet.gzip', compression='gzip')
  pd.set_option('display.expand_frame_repr', False)
  end = time.time()
  print('Successfully analyzed', len(features_dataframe.index), 'audio files.')
  print('This script took ', str(round(end - start, 2)), 'seconds to execute.')
  print('The data frame file has been saved to audio-features.parquet.gzip.')
  print('Data frame preview of the first 5 rows:')
  print(features_dataframe[:5])
    
else:
  # Zip files are necessary to extract features
  print('Please download Audio_Speech_Actors_01-24.zip and Audio_Song_Actors_01-24.zip from https://zenodo.org/record/1188976')
  print('Place these files in a folder called raw-data in the main directory.')
