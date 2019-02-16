
'''
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from sklearn.utils import shuffle


import keras
from keras.utils import np_utils

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Input, Flatten, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder



Hdata=pq.read_pandas('audio-features.parquet').to_pandas()
Hdata=shuffle(Hdata)
i = np.random.rand(len(Hdata)) < 0.8
train = Hdata[i]
test = Hdata[~i]


print(Hdata)

print('============')
print(train)
print('============')
print(test)
print('============')


trainfeatures = train.iloc[:,[-1]]
trainlabel = train.iloc[:,[-2]]
testfeatures = test.iloc[:,[-1]]
testlabel = test.iloc[:, [-2]]

X_train = np.array(np.ravel(trainfeatures))
y_train = np.array(trainlabel)
X_test = np.array(testfeatures)
y_test = np.array(testlabel)

lb = LabelEncoder()
x_traincnn =np.expand_dims(X_train, axis=2)
x_testcnn= np.expand_dims(X_test, axis=2)

y_train = np_utils.to_categorical(lb.fit_transform(y_train))
y_test = np_utils.to_categorical(lb.fit_transform(y_test))
print(y_train[1])
print('*************')
print(X_train[1])

print(trainfeatures)
print('====================')
print(trainlabel)


model = Sequential()

model.add(Conv1D(256, 5,padding='same',
                 input_shape=(216,1)))
model.add(Activation('relu'))
model.add(Conv1D(128, 5,padding='same'))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(MaxPooling1D(pool_size=(8)))
model.add(Conv1D(128, 5,padding='same',))
model.add(Activation('relu'))
#model.add(Conv1D(128, 5,padding='same',))
#model.add(Activation('relu'))
#model.add(Conv1D(128, 5,padding='same',))
#model.add(Activation('relu'))
#model.add(Dropout(0.2))
model.add(Conv1D(128, 5,padding='same',))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(10))
model.add(Activation('softmax'))
opt = keras.optimizers.rmsprop(lr=0.00001, decay=1e-6)
print(
model.summary())
model.compile(loss='categorical_crossentropy', optimizer=opt,metrics=['accuracy'])

cnnhistory=model.fit(x_traincnn, y_train, batch_size=16, epochs=700, validation_data=(x_testcnn, y_test))
'''

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
import keras
from keras.utils import np_utils
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Input, Flatten, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder


data = pq.read_pandas('audio-features.parquet').to_pandas()
features = data['features'].values
emotions = data['emotions'].values

# https://stackoverflow.com/questions/24147278/how-do-i-create-test-and-train-samples-from-one-dataframe-with-pandas
msk = np.random.rand(len(data)) < 0.8

#Hdata=shuffle(Hdata)
#i = np.random.rand(len(Hdata)) < 0.8
#train = Hdata[i]
#test = Hdata[~i]

#trainfeatures = train.iloc[:,[-1]]
#trainlabel = train.iloc[:,[-2]]
#testfeatures = test.iloc[:,[-1]]
#testlabel = test.iloc[:, [-2]]

#X_train = np.concatenate(np.array(trainfeatures))
#y_train = np.array(trainlabel)
#X_test = np.array(testfeatures)
#y_test = np.array(testlabel)
#print(X_train.shape)


'''
lb = LabelEncoder()

x_traincnn =np.expand_dims(X_train, axis=2)
x_testcnn= np.expand_dims(X_test, axis=2)

y_train = np_utils.to_categorical(lb.fit_transform(y_train))
y_test = np_utils.to_categorical(lb.fit_transform(y_test))

model = Sequential()

model.add(Conv1D(256, 5,padding='same',
                 input_shape=(216,1)))
model.add(Activation('relu'))
model.add(Conv1D(128, 5,padding='same'))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(MaxPooling1D(pool_size=(8)))
model.add(Conv1D(128, 5,padding='same',))
model.add(Activation('relu'))
#model.add(Conv1D(128, 5,padding='same',))
#model.add(Activation('relu'))
#model.add(Conv1D(128, 5,padding='same',))
#model.add(Activation('relu'))
#model.add(Dropout(0.2))
model.add(Conv1D(128, 5,padding='same',))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(10))
model.add(Activation('softmax'))
opt = keras.optimizers.rmsprop(lr=0.00001, decay=1e-6)
print(
model.summary())
model.compile(loss='categorical_crossentropy', optimizer=opt,metrics=['accuracy'])

cnnhistory=model.fit(x_traincnn, y_train, batch_size=16, epochs=700, validation_data=(x_testcnn, y_test))'''

'''
end = time.time()
print('Data Division is done')
print(len(train.index), 'training sample')
print(len(test.index), 'testing sample')
print('This script took ', str(round(end - start, 2)), 'seconds to execute.')'''