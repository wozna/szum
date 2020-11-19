from pathlib import Path
import time

from scipy.io import wavfile
import numpy as np
import pandas as pd
from scipy import signal
from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

import keras
import os
import sys
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dense, Input, Dropout, Flatten
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint
from utils_simple import get_data, prepare_data, get_specgrams, batch_generator, get_model

# import model from file 




def predict(args):
    labelbinarizer = LabelBinarizer()
    model_path = args[0]
    dataset_path = args[1]
    test = prepare_data(get_data(dataset_path))
    y = labelbinarizer.fit_transform(test.word)
    predictions = []
    paths = test.path.tolist()
    label = test.word.tolist()
    model = load_model(model_path)

    for path in paths:
        specgram = get_specgrams([path])
        pred = model.predict(np.array(specgram))
        predictions.extend(pred)
    
    labels = [labelbinarizer.inverse_transform(
    p.reshape(1, -1), threshold=0.5)[0] for p in predictions]
    test['labels'] = labels
    answers = test.path.tolist()
    correct = 0
    # for i in range(10):
    #     print(labels[i])
    #     print(os.path.basename(os.path.dirname(answers[i])))

    for i in range(len(labels)):
        if labels[i] == os.path.basename(os.path.dirname(answers[i])):
            correct=correct+1
    
    print(correct / len(labels))

    test.path = test.path.apply(lambda x: str(x))
    submission = pd.DataFrame({'fname': test.path.tolist(), 'label': labels})
    submission.to_csv('submission.csv', index=False)

# args = [ file_name, model_path, dataset_path ]
predict(sys.argv[1:])
