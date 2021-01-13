from bento_predictor import DigitGenerator
from tensorflow.keras.models import load_model
import tensorflow as tf


def classifier_models(model_service, model_path: str):
    model_gen = load_model(model_path)
    tf.saved_model.save(model_gen, "artifacts/")
    model_gen = tf.saved_model.load("artifacts/")
    model_service.pack("model", model_gen)


def main():
    model_service = DigitGenerator()
    classifier_models(model_service=model_service, model_path=generator_model_path)
    saved_path = model_service.save()


generator_model_path = (
    "/Users/vsatpathy/Desktop/docs/training_data/van_gan/generator.h5"
)
main()