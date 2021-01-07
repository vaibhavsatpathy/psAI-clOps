from bento_predictor import ModelZoo
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
import spacy
import json


def som_models(
    model_zoo,
    tf_model_path_som: str,
    text_file_path_som: str,
    master_labels_path: str,
    sub_labels_path: str,
):
    model_cnn = load_model(tf_model_path_som)
    tf.saved_model.save(model_cnn, "artifacts/")
    model_cnn = tf.saved_model.load("artifacts/")
    model_zoo.pack("som_model", model_cnn)

    text_model = spacy.load("en_core_web_sm")
    model_zoo.pack("som_spacy_model", text_model)

    tokenizer = Tokenizer()
    with open(text_file_path_som, "r") as f:
        bow = f.read()
    tokenizer.fit_on_texts(bow.split("####"))
    model_zoo.pack("som_tokenizer", tokenizer)

    with open(master_labels_path, "r") as f:
        labels_som = json.load(f)
    model_zoo.pack("som_master_labels", labels_som)

    with open(sub_labels_path, "r") as g:
        sub_labels_som = json.load(g)
    model_zoo.pack("som_sub_labels", sub_labels_som)


def main():
    model_zoo = ModelZoo()
    som_models(
        model_zoo=model_zoo,
        tf_model_path_som=tf_model_path_som,
        text_file_path_som=text_file_path_som,
        master_labels_path=master_labels_path,
        sub_labels_path=sub_labels_path,
    )
    saved_path = model_zoo.save()


tf_model_path_som = (
    "/Users/vsatpathy/Desktop/docs/training_data/som/document_classifier.h5"
)
text_file_path_som = (
    "/Users/vsatpathy/Desktop/docs/training_data/som/file_and_text_som.txt"
)
master_labels_path = (
    "/Users/vsatpathy/Desktop/docs/training_data/som/rev_labels_master_som.json"
)
sub_labels_path = "/Users/vsatpathy/Desktop/docs/training_data/som/rev_labels_som.json"
main()
