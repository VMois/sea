import pandas as pd
import numpy as np
import tensorflow as tf
import pyarrow.parquet as pq
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
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
import time
import matplotlib.pyplot as plt

data = pq.read_pandas('audio-features.parquet').to_pandas()
data = shuffle(data)
print(data.columns)

# https://stackoverflow.com/questions/24147278/how-do-i-create-test-and-train-samples-from-one-dataframe-with-pandas
msk = np.random.rand(len(data)) < 0.8

train_data = data[msk]
test_data = data[~msk]

train_features =list(train_data['features'].values)

train_emotions = np.array(train_data['emotion'].values)
test_features = np.array(test_data['features'].values)
test_emotions = np.array(test_data['emotion'].values)
lb = LabelEncoder()

print(train_features.shape)

x_traincnn =np.expand_dims(train_features, axis=2)
x_testcnn= np.expand_dims(test_features, axis=2)

print (x_traincnn.shape)

'''

model = Sequential()

model.add(Conv1D(256, 5,padding='same',input_shape=(216,1)))
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



print (model.summary())

model.compile(loss='categorical_crossentropy', optimizer=opt,metrics=['accuracy'])

cnnhistory=model.fit(x_traincnn, train_emotions, batch_size=16, epochs=700, validation_data=(x_testcnn, test_emotions))

'''