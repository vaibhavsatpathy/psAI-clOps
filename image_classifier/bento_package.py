from bento_predictor import ImageClassifier
from tensorflow.keras.models import load_model
import tensorflow as tf
import json


def classifier_models(model_service, model_path: str, labels_path: str):
    model_cnn = load_model(model_path)
    tf.saved_model.save(model_cnn, "artifacts/")
    model_cnn = tf.saved_model.load("artifacts/")
    model_service.pack("model", model_cnn)

    with open(labels_path, "r") as f:
        labels = json.load(f)
    model_service.pack("labels", labels)


def main():
    model_service = ImageClassifier()
    classifier_models(
        model_service=model_service, model_path=model_path, labels_path=labels_path
    )
    saved_path = model_service.save()


model_path = "/Users/vsatpathy/Desktop/docs/training_data/intel/image_classifier.h5"
labels_path = "/Users/vsatpathy/Desktop/docs/training_data/intel/rev_labels_intel.json"
main()