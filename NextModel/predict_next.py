import numpy as np
from tensorflow import keras
from utils_next import wav2feature, prepare_data, get_data, word2index, get_audio

index2word = [word for word in word2index]
print(index2word)
num_classes = len(word2index)
num_samples_per_class = 2000

print("loading dataset...")
#train = prepare_data(get_data('/macierz/home/s165554/pg/szum/data/evaluation/'))
train = prepare_data(get_data('data_new/evaluation'))
model = keras.models.load_model("models/NextModel/model_January10_.11.hdf5")

samples = []
classes = []

for class_name in train.word:
    classes.append(word2index[class_name])
classes = np.array(classes, dtype=np.int)

for path in train.path:
    samples.append(get_audio(path))

samples = np.array(samples)
classes = np.array(classes, dtype=np.int)

correct = 0

for id, rec in enumerate(samples):
    recorded_feature = wav2feature(rec)

    recorded_feature = np.expand_dims(recorded_feature, 0)  # add "fake" batch dimension 1
    prediction = model.predict(recorded_feature).reshape((11,))
    # normalize prediction output to get "probabilities"
    prediction /= prediction.sum()

    # print the 3 candidates with highest probability
    prediction_sorted_indices = prediction.argsort()
    print("candidates:\n------------  " + index2word[classes[id]])
    for k in range(3):
        i = int(prediction_sorted_indices[-1 - k])
        print("%d.)\t%s\t:\t%2.1f%%" % (k + 1, index2word[i], prediction[i] * 100))
        if k == 0 and index2word[i] == index2word[classes[id]]:
            correct += 1

    print("-----------------------------")

acc = correct / len(samples)
print("TOP1 acc = " + str(acc))

