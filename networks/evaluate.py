from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.xception import preprocess_input

model = load_model('models/NetworkInceptionV3_18_October/NetworkInceptionV3_18_October.hdf5')

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

test_generator = test_datagen.flow_from_directory(
    'data/validation',
    target_size=(299, 299),
    batch_size=32,
    class_mode='categorical',
    shuffle=False)  # dont shuffle data

STEP_SIZE_EVALUATE = test_generator.n // test_generator.batch_size
label_map = test_generator.class_indices
print(label_map)
scores = model.evaluate_generator(test_generator, steps=STEP_SIZE_EVALUATE, verbose=1)

print("Accuracy = ", scores[1])
print("Loss = ", scores[0])
print(model.metrics_names)
