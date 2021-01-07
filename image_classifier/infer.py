from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img

import numpy as np
import os
import random


def preprocess(image_path: str, image_shape: tuple = None):
    image_dimensions = (100, 100, 3)
    if image_shape:
        pass
    else:
        image_shape = image_dimensions
    test_inp = image_path
    test_img = np.asarray(load_img(test_inp, target_size=image_shape))
    test_img = np.divide(test_img, 255.0)
    test_img = np.asarray([test_img]).astype("float32")
    return test_img


def predict(folder_path: str, model_path: str):
    image_dimensions = (100, 100, 3)
    full_image_path = os.path.join(folder_path, random.choice(os.listdir(folder_path)))
    model = load_model(model_path)
    image = preprocess(image_path=full_image_path, image_shape=image_dimensions)
    results = model.predict(image)
    print(full_image_path)
    print(np.argmax(results[0]))


model_path = "/Users/vsatpathy/Desktop/docs/training_data/intel/image_classifier.h5"
folder_path = (
    "/Users/vsatpathy/Desktop/off_POCs/intel-image-classification/seg_train/buildings"
)
predict(folder_path=folder_path, model_path=model_path)
