import pandas as pd
import numpy as np
import tensorflow as tf
import pyarrow.parquet as pq
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
import time

data = pq.read_pandas('audio-features.parquet').to_pandas()
data = shuffle(data)
print(data.columns)

# https://stackoverflow.com/questions/24147278/how-do-i-create-test-and-train-samples-from-one-dataframe-with-pandas
msk = np.random.rand(len(data)) < 0.8

train_data = data[msk]
test_data = data[~msk]

train_features = train_data['features'].values
train_emotions = train_data['emotion'].values
test_features = test_data['features'].values
test_emotions = test_data['emotion'].values

#print(train_features[0])

'''
model = tf.keras.Sequential()

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
opt = keras.optimizers.rmsprop(lr=0.00001, decay=1e-6)<





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

end = time.time()
print('Data Division is done')
print(len(train.index), 'training sample')
print(len(test.index), 'testing sample')
print('This script took ', str(round(end - start, 2)), 'seconds to execute.')
'''