import bentoml
from bentoml.adapters import JsonInput
from bentoml.frameworks.tensorflow import TensorflowSavedModelArtifact

import tensorflow as tf
import importlib.util
import numpy as np
from PIL import Image


@bentoml.env(infer_pip_packages=True)
@bentoml.artifacts([TensorflowSavedModelArtifact("model")])
class ConditionalDigitGenerator(bentoml.BentoService):
    @bentoml.api(input=JsonInput())
    def generate_conditional_image(self, parsed_json):
        model = self.artifacts.model.signatures["serving_default"]
        model._num_positional_args = 2
        noise = np.random.normal(0, 1, (1, 100))
        noise = tf.convert_to_tensor(noise, dtype=tf.float32)
        label = np.asarray(int(parsed_json.get("number"))).reshape(-1, 1)
        label = tf.convert_to_tensor(label, dtype=tf.int32)
        results = model(noise, label)
        generated_image = results.get("sequential")[0].numpy().reshape(28, 28)
        return {"digit_generated": generated_image}
