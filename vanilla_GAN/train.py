import os
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import argparse

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras import initializers
from tensorflow.keras.layers import (
    Reshape,
    Dense,
    Dropout,
    Flatten,
    BatchNormalization,
    Convolution2D,
    UpSampling2D,
    Input,
    LeakyReLU,
)


tracking_uri = (
    "http://testuser:password@ec2-18-218-100-222.us-east-2.compute.amazonaws.com"
)
# tracking_uri = "postgresql://postgres:postgres@localhost:5432/"
s3_bucket = "s3://docuedge-mlflow-bucket"  # replace this value


def generator():
    gen = Sequential()
    gen.add(Dense(256, input_dim=100))
    gen.add(LeakyReLU(0.2))
    gen.add(Dense(512))
    gen.add(LeakyReLU(0.2))
    gen.add(Dense(1024))
    gen.add(LeakyReLU(0.2))
    gen.add(Dense(784, activation="tanh"))
    gen.compile(loss="binary_crossentropy", optimizer=Adam(lr=0.0002, beta_1=0.5))
    return gen


def discriminator():
    disc = Sequential()
    disc.add(Dense(1024, input_dim=784))
    disc.add(LeakyReLU(0.2))
    disc.add(Dropout(0.2))
    disc.add(Dense(512))
    disc.add(LeakyReLU(0.2))
    disc.add(Dropout(0.2))
    disc.add(Dense(256))
    disc.add(LeakyReLU(0.2))
    disc.add(Dropout(0.2))
    disc.add(Dense(1, activation="sigmoid"))
    disc.compile(loss="binary_crossentropy", optimizer=Adam(lr=0.0002, beta_1=0.5))
    return disc


def stacked_GAN(gen, disc):
    disc.trainable = False
    gan_input = Input(shape=(100,))
    x = gen(gan_input)
    gan_out = disc(x)
    gan_stack = Model(inputs=gan_input, outputs=gan_out)
    gan_stack.compile(loss="binary_crossentropy", optimizer=Adam(lr=0.0002, beta_1=0.5))
    return gan_stack


def train(
    gen,
    disc,
    gan_stack,
    max_iter: int,
    batch_size: int,
    img_shape: int,
    file_path: str,
    artifact_name: str,
    exp_name: str,
):

    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    try:
        expr_name = exp_name  # create a new experiment (do not replace)
        mlflow.create_experiment(expr_name, s3_bucket)
        mlflow.set_experiment(expr_name)
        experiment = mlflow.get_experiment_by_name(exp_name)
    except:
        experiment = mlflow.get_experiment_by_name(exp_name)

    os.makedirs(os.path.join(file_path, artifact_name), exist_ok=True)
    mlflow.tensorflow.autolog(every_n_iter=1)
    with mlflow.start_run(experiment_id=experiment.experiment_id) as run:

        mlflow.log_metrics(
            {
                "batch_size": batch_size,
                "epochs": max_iter,
                "image_shape": img_shape,
            }
        )

        (X_train, _), (_, _) = mnist.load_data()
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = X_train.reshape(60000, img_shape)

        for i in range(0, max_iter):
            noise = np.random.normal(0, 1, (batch_size, 100))
            image_batch = X_train[
                np.random.randint(0, X_train.shape[0], size=batch_size)
            ]

            fake_images = gen.predict(noise)

            final_images = np.concatenate([image_batch, fake_images])
            final_labels = np.concatenate(
                (
                    np.ones((np.int64(batch_size), 1)),
                    np.zeros((np.int64(batch_size), 1)),
                )
            )

            disc.trainable = True
            disc_loss = disc.train_on_batch(final_images, final_labels)

            disc.trainable = False
            y_mis_labels = np.ones(batch_size)
            gen_loss = gan_stack.train_on_batch(noise, y_mis_labels)

            mlflow.log_metrics(
                {"generator_loss": gen_loss, "discriminator_loss": disc_loss}
            )

            print(
                "epoch_%d---->gen_loss:[%f]---->disc_loss:[%f]"
                % (i, gen_loss, disc_loss)
            )
            # if i % 1000 == 0:
            #     test(gen, i)

        gen.save(os.path.join(file_path, artifact_name, "generator.h5"))
        # disc.save(os.path.join(file_path, artifact_name, "discriminator.h5"))

        meta_data_path = os.path.join(file_path, artifact_name)
        for artifact in sorted(os.listdir(meta_data_path)):
            if artifact != ".DS_Store":
                artifact_path = os.path.join(meta_data_path, artifact)
                if (
                    os.path.isfile(artifact_path)
                    and artifact_path.split(".")[-1] != "h5"
                ):
                    print(f"artifact to be uploaded is: {artifact}")
                    mlflow.log_artifact(local_path=artifact_path)

        artifact_uri = mlflow.get_artifact_uri()
        print(artifact_uri)
        mlflow.end_run()


parser = argparse.ArgumentParser()
parser.add_argument("-fp", "--file_path", help="Directory path to save artifacts")
parser.add_argument("-a", "--art_name", help="Artifacts name")
parser.add_argument("-b", "--batch_size", default=32, help="Batch size for training")
parser.add_argument("-e", "--epochs", default=20000, help="Number of epochs")
parser.add_argument("-is", "--img_shape", default=784, help="One dimension of image")
parser.add_argument(
    "-exp",
    "--experiment_name",
    default="vanilla_gan",
    help="Name of the experiment for tracking",
)
args = parser.parse_args()
train(
    gen=generator(),
    disc=discriminator(),
    gan_stack=stacked_GAN(gen=generator(), disc=discriminator()),
    max_iter=int(args.epochs),
    batch_size=int(args.batch_size),
    img_shape=int(args.img_shape),
    file_path=args.file_path,
    artifact_name=args.art_name,
    exp_name=args.experiment_name,
)
