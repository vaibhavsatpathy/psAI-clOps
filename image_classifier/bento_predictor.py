import bentoml
from bentoml.adapters import FileInput
from bentoml.frameworks.tensorflow import TensorflowSavedModelArtifact
from bentoml.service.artifacts.common import JSONArtifact

import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
import importlib.util
import numpy as np
from PIL import Image


@bentoml.env(infer_pip_packages=True)
@bentoml.artifacts([TensorflowSavedModelArtifact("model"), JSONArtifact("labels")])
class ImageClassifier(bentoml.BentoService):
    def pre_process_image(self, image_file):
        image = np.asarray(
            Image.open(image_file).convert(mode="RGB").resize((100, 100))
        )
        image = np.divide(image, 255.0)
        image = np.asarray([image]).astype("float32")
        return image

    @bentoml.api(input=FileInput())
    def predict_image(self, file_stream):
        image = self.pre_process_image(image_file=file_stream)
        model = self.artifacts.model.signatures["serving_default"]
        model._num_positional_args = 1
        results = model(tf.constant(image))
        print(results)
        conv_results = results.get("dense")[0].numpy()
        label = self.artifacts.labels[str(np.argmax(conv_results))]
        return {"label": label}
