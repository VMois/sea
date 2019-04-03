import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

train_data = pq.read_pandas('data/train/').to_pandas().dropna()
test_data = pq.read_pandas('data/test/').to_pandas().dropna()

train_features = np.array(pd.DataFrame(train_data['features'].values.tolist()).fillna(0))
train_emotions = (train_data['gender'].map(str) + '_' + train_data['emotion']).values

test_features = np.array(pd.DataFrame(test_data['features'].values.tolist()).fillna(0))
test_emotions = (test_data['gender'].map(str) + '_' + test_data['emotion']).values
lb = LabelEncoder()

train_emotions = tf.keras.utils.to_categorical(lb.fit_transform(train_emotions))
test_emotions = tf.keras.utils.to_categorical(lb.fit_transform(test_emotions))

train_features_cnn = np.expand_dims(train_features, axis=2)
test_features_cnn = np.expand_dims(test_features, axis=2)

model = tf.keras.Sequential()

model.add(layers.Conv1D(256, 5, padding='same', input_shape=(216, 1)))
model.add(layers.Activation('relu'))
model.add(layers.Conv1D(128, 5, padding='same'))
model.add(layers.Activation('relu'))
model.add(layers.Dropout(0.1))
model.add(layers.MaxPooling1D(pool_size=(8)))
model.add(layers.Conv1D(128, 5, padding='same',))
model.add(layers.Activation('relu'))
model.add(layers.Conv1D(128, 5, padding='same',))
model.add(layers.Activation('relu'))
model.add(layers.Flatten())
model.add(layers.Dense(10))
model.add(layers.Activation('softmax'))
opt = tf.keras.optimizers.RMSprop(lr=0.00001, decay=1e-6)

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

cnnhistory = model.fit(train_features_cnn,
                       train_emotions,
                       batch_size=16,
                       epochs=100,
                       validation_data=(test_features_cnn, test_emotions))

plt.plot(cnnhistory.history['loss'])
plt.plot(cnnhistory.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
