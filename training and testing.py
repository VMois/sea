'''

BME AI - Speech Emotion Analysis - Feature Extraction
-----------------------------------------------------
This script will automatically train and test the model

********* Instructions ************


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
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
import time

print(' ___ __  __ ___     _   ___ ')
print('| _ )  \/  | __|   /_\ |_ _|')
print('| _ \ |\/| | _|   / _ \ | | ')
print('|___/_|  |_|___| /_/ \_\___|')
print('                            ')

# Time script execution for performance evaluation.
start = time.time()

Hdata=pq.read_pandas('audio-features.parquet').to_pandas()
Hdata=shuffle(Hdata)
i = np.random.rand(len(Hdata)) < 0.8
train = Hdata[i]
test = Hdata[~i]
'''
trainfeatures = train.iloc[:,[-1]]
trainlabel = train.iloc[:,[-2]]
testfeatures = test.iloc[:,[-1]]
testlabel = test.iloc[:, [-2]]

X_train = np.array(trainfeatures)
y_train = np.array(trainlabel)
X_test = np.array(testfeatures)
y_test = np.array(testlabel)


x_traincnn =np.expand_dims(X_train, axis=2)
x_testcnn= np.expand_dims(X_test, axis=2)


y_train = np_utils.to_categorical(lb.fit_transform(y_train))
y_test = np_utils.to_categorical(lb.fit_transform(y_test))
'''
end = time.time()
print('Data Division is done')
print(len(train.index), 'training sample')
print(len(test.index), 'testing sample')
print('This script took ', str(round(end - start, 2)), 'seconds to execute.')
