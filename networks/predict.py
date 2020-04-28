from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import sys


def predict(args):
    model_path = args[0]
    photo_path = args[1]
    model = load_model(model_path)
    img = image.load_img(photo_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = np.divide(x, 223)

    scores = model.predict(x)
    r = np.argmax(scores)
    return r


# args = [ file_name, model_path, photo_model ]
result = predict(sys.argv[1:])
print((result), end='')
