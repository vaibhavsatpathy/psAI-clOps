from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def test(gen_model_path: str, i: int):
    gen = load_model(gen_model_path)
    noise = np.random.normal(0, 1, (1, 100))
    image = np.squeeze(gen.predict(noise), axis=0)
    plt.imsave(
        "/Users/vsatpathy/Desktop/off_POCs/cycle_gan/epoch_%d" % i,
        image.reshape(28, 28),
        format="jpg",
        cmap="gray",
    )


generator_model_path = (
    "/Users/vsatpathy/Desktop/docs/training_data/van_gan/generator.h5"
)
test(gen_model_path=generator_model_path, i=0)
