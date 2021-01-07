import bentoml
from bentoml.types import FileLike
from bentoml.adapters import JsonInput, FileInput, MultiFileInput
from bentoml.frameworks.spacy import SpacyModelArtifact
from bentoml.frameworks.tensorflow import TensorflowSavedModelArtifact
from bentoml.service.artifacts.common import (
    JSONArtifact,
    PickleArtifact,
)

import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
import numpy as np
from pytesseract import image_to_string
import re
from PIL import Image
from typing import List


@bentoml.env(infer_pip_packages=True)
@bentoml.artifacts(
    [
        TensorflowSavedModelArtifact("mcr_model"),
        SpacyModelArtifact("mcr_spacy_model"),
        PickleArtifact("mcr_tokenizer"),
        JSONArtifact("mcr_labels"),
        TensorflowSavedModelArtifact("som_model"),
        SpacyModelArtifact("som_spacy_model"),
        PickleArtifact("som_tokenizer"),
        JSONArtifact("som_master_labels"),
        JSONArtifact("som_sub_labels"),
    ]
)
class ModelZoo(bentoml.BentoService):
    def helper(self, text):
        dummy = []
        for word in text:
            dummy.append(str(word))
        final = " ".join(dummy)
        return final

    def preprocess_spacy(self, spacy_model, text, num_of_words: int):
        text = str(text)
        text = text.split(" ")
        text = self.helper(text)
        text = str(text.lower())
        # Remove all the special characters
        text = re.sub(r"\W", " ", text)
        text = re.sub(r"[^a-zA-Z ]+", "", text)
        # remove all single characters
        text = re.sub(r"\s+[a-zA-Z]\s+", " ", text)
        # Remove single characters from the start
        text = re.sub(r"\^[a-zA-Z]\s+", " ", text)
        # Substituting multiple spaces with single space
        text = re.sub(r"\s+", " ", text, flags=re.I)
        # text = self.artifacts.mcr_spacy_model(text)
        text = spacy_model(text)
        filtered = [token.lemma_ for token in text if token.is_stop == False]
        text = " ".join(filtered[: num_of_words * 2])
        text = text.strip().split(" ")
        text = " ".join(text[:num_of_words])
        return text

    def tokenize_sentence(self, sentence, tokenizer, maximum_word_length):
        updated_sentence = sentence.split(" ")
        tok_sent = []
        for word in updated_sentence:
            if word in tokenizer.word_index:
                tok_sent.append(tokenizer.word_index[word])
            else:
                tok_sent.append(0)
        if len(tok_sent) != maximum_word_length:
            delta = maximum_word_length - len(tok_sent)
            for i in range(delta):
                tok_sent.append(0)
        return tok_sent

    def pre_process_image(self, image_file):
        ocr_image = np.asarray(Image.open(image_file))
        image = np.asarray(
            Image.open(image_file).convert(mode="RGB").resize((100, 100))
        )
        image = np.divide(image, 255.0)
        image = np.asarray([image]).astype("float32")
        return ocr_image, image

    def pre_process_mcr(self, file):
        ocr_image, image = self.pre_process_image(image_file=file)
        doc_text = image_to_string(ocr_image)
        doc_text_processed = self.preprocess_spacy(
            spacy_model=self.artifacts.mcr_spacy_model, text=doc_text, num_of_words=10
        )
        fin_text = self.tokenize_sentence(
            sentence=doc_text_processed,
            tokenizer=self.artifacts.mcr_tokenizer,
            maximum_word_length=10,
        )
        return image, np.asarray([fin_text]).astype("float32")

    def pre_process_som(self, file):
        ocr_image, image = self.pre_process_image(image_file=file)
        doc_text = image_to_string(ocr_image)
        doc_text_processed = self.preprocess_spacy(
            spacy_model=self.artifacts.som_spacy_model, text=doc_text, num_of_words=10
        )
        fin_text = self.tokenize_sentence(
            sentence=doc_text_processed,
            tokenizer=self.artifacts.som_tokenizer,
            maximum_word_length=10,
        )
        return image, np.asarray([fin_text]).astype("float32")

    @bentoml.api(input=FileInput())
    def predict_document_labels_mcr(self, file_stream):
        image, text = self.pre_process_mcr(file=file_stream)
        model = self.artifacts.mcr_model.signatures["serving_default"]
        model._num_positional_args = 2
        results = model(tf.constant(text), tf.constant(image))
        conv_results = results.get("dense_1")[0].numpy()
        document_label = self.artifacts.mcr_labels[str(np.argmax(conv_results))]
        return {"document_type": document_label}

    @bentoml.api(input=FileInput())
    def predict_document_labels_som(self, file_stream):
        image, text = self.pre_process_som(file=file_stream)
        model = self.artifacts.som_model.signatures["serving_default"]
        model._num_positional_args = 2
        results = model(tf.constant(text), tf.constant(image))
        mas_results = results.get("dense_1")[0].numpy()
        sub_results = results.get("dense_4")[0].numpy()
        master_label = self.artifacts.som_master_labels[str(np.argmax(mas_results))]
        sub_label = self.artifacts.som_sub_labels[str(np.argmax(sub_results))]
        return {"master document type": master_label, "sub document type": sub_label}
