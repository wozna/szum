from pathlib import Path
import time

from scipy.io import wavfile
import numpy as np
import pandas as pd
from scipy import signal
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from utils_simple import get_data, prepare_data, get_specgrams, batch_generator, get_model

import keras
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint


train = prepare_data(get_data('/macierz/home/s165554/pg/szum/data/training'))
shape = (129, 124, 1)
model = get_model(shape)
model.compile(loss='binary_crossentropy',
              optimizer=Adam(), metrics=['accuracy'])


# create training and test data.

labelbinarizer = LabelBinarizer()
X = train.path
y = labelbinarizer.fit_transform(train.word)
X, Xt, y, yt = train_test_split(X, y, test_size=0.3, stratify=y)

tensorboard = TensorBoard(
    log_dir='./logs/{}'.format(time.time()), batch_size=32)
check_pointer = ModelCheckpoint(filepath="models/" + 'model.{epoch:02d}-{val_loss:.2f}.hdf5',
                                verbose=1,
                                save_best_only=False)


train_gen = batch_generator(X.values, y, batch_size=32)
valid_gen = batch_generator(Xt.values, yt, batch_size=32)

model.fit_generator(
    generator=train_gen,
    epochs=20,
    steps_per_epoch=X.shape[0] // 32,
    validation_data=valid_gen,
    validation_steps=Xt.shape[0] // 32,
    callbacks=[check_pointer])


# Create a submission

# test = prepare_data(get_data('/macierz/home/s165554/pg/szum/data/evaluation'))

# predictions = []
# paths = test.path.tolist()

# for path in paths:
#     specgram = get_specgrams([path])
#     pred = model.predict(np.array(specgram))
#     predictions.extend(pred)

# labels = [labelbinarizer.inverse_transform(
#     p.reshape(1, -1), threshold=0.5)[0] for p in predictions]
# test['labels'] = labels
# test.path = test.path.apply(lambda x: str(x))
# submission = pd.DataFrame({'fname': test.path.tolist(), 'label': labels})
# print(submission)
# #submission.to_csv('submission.csv', index=False)
