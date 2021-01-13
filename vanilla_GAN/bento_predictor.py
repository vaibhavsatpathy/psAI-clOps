import bentoml
from bentoml.adapters import JsonInput
from bentoml.frameworks.tensorflow import TensorflowSavedModelArtifact

import tensorflow as tf
import importlib.util
import numpy as np
from PIL import Image


@bentoml.env(infer_pip_packages=True)
@bentoml.artifacts([TensorflowSavedModelArtifact("model")])
class DigitGenerator(bentoml.BentoService):
    @bentoml.api(input=JsonInput())
    def generate_image(self, file_stream):
        model = self.artifacts.model.signatures["serving_default"]
        model._num_positional_args = 1
        noise = np.random.normal(0, 1, (1, 100))
        noise = tf.convert_to_tensor(noise, dtype=tf.float32)
        results = model(noise)
        generated_image = results.get("dense_3")[0].numpy().reshape(28, 28)
        return {"digit_generated": generated_image}
