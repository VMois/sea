import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
from sklearn.preprocessing import LabelEncoder
from utils.data_loader import load_data

train_data = load_data(['crema', 'savee'], 'mfcc').dropna()
test_data = load_data(['ravdess'], 'mfcc').dropna()

# choose only emotions from the list
allowed_emotions = ['happy', 'sad', 'angry', 'disgust', 'neutral']
train_data = train_data[train_data['emotion'].isin(allowed_emotions)]
test_data = test_data[test_data['emotion'].isin(allowed_emotions)]

train_features = np.array(pd.DataFrame(train_data['features'].values.tolist()).fillna(0))
train_labels = (train_data['gender'].map(str) + '_' + train_data['emotion']).values

test_features = np.array(pd.DataFrame(test_data['features'].values.tolist()).fillna(0))
test_labels = (test_data['gender'].map(str) + '_' + test_data['emotion']).values
lb = LabelEncoder()

train_labels = tf.keras.utils.to_categorical(lb.fit_transform(train_labels))
test_labels = tf.keras.utils.to_categorical(lb.fit_transform(test_labels))

train_features_cnn = np.expand_dims(train_features, axis=2)
test_features_cnn = np.expand_dims(test_features, axis=2)


model = tf.keras.Sequential()

model.add(layers.Conv1D(256, 3, padding='same', input_shape=(173, 1)))
model.add(layers.Activation('relu'))
model.add(layers.Conv1D(128, 5, padding='same'))
model.add(layers.Activation('relu'))
model.add(layers.Dropout(0.2))
model.add(layers.MaxPooling1D(pool_size=(8)))
model.add(layers.Conv1D(128, 5, padding='same',))
model.add(layers.Activation('relu'))
model.add(layers.Conv1D(128, 9, padding='same',))
model.add(layers.Activation('relu'))
model.add(layers.Flatten())
model.add(layers.Dense(128))
model.add(layers.Activation('relu'))
model.add(layers.Dense(10))
model.add(layers.Activation('softmax'))
opt = tf.keras.optimizers.Adam(lr=0.00001, decay=1e-6)

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

cnnhistory = model.fit(train_features_cnn,
                       train_labels,
                       batch_size=16,
                       epochs=2,
                       validation_data=(test_features_cnn, test_labels))
