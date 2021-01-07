from bento_predictor import ModelZoo
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
import spacy
import json


def mcr_models(
    model_zoo, tf_model_path_mcr: str, text_file_path_mcr: str, labels_path_mcr: str
):
    model_cnn = load_model(tf_model_path_mcr)
    tf.saved_model.save(model_cnn, "artifacts/")
    model_cnn = tf.saved_model.load("artifacts/")
    model_zoo.pack("mcr_model", model_cnn)

    text_model = spacy.load("en_core_web_sm")
    model_zoo.pack("mcr_spacy_model", text_model)

    tokenizer = Tokenizer()
    with open(text_file_path_mcr, "r") as f:
        bow = f.read()
    tokenizer.fit_on_texts(bow.split("####"))
    model_zoo.pack("mcr_tokenizer", tokenizer)

    with open(labels_path_mcr, "r") as f:
        labels_mcr = json.load(f)
    model_zoo.pack("mcr_labels", labels_mcr)


def main():
    model_zoo = ModelZoo()
    mcr_models(
        model_zoo=model_zoo,
        tf_model_path_mcr=tf_model_path_mcr,
        text_file_path_mcr=text_file_path_mcr,
        labels_path_mcr=labels_path_mcr,
    )
    saved_path = model_zoo.save()


tf_model_path_mcr = (
    "/Users/vsatpathy/Desktop/docs/training_data/mcr/document_classifier.h5"
)
text_file_path_mcr = (
    "/Users/vsatpathy/Desktop/docs/training_data/mcr/file_and_text_mcr.txt"
)
labels_path_mcr = "/Users/vsatpathy/Desktop/docs/training_data/mcr/rev_labels_mcr.json"
main()
