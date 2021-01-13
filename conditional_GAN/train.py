import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import mlflow

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras import initializers
from tensorflow.keras.layers import (
    Input,
    multiply,
    Embedding,
    LeakyReLU,
    Reshape,
    Dense,
    Dropout,
    Flatten,
    Convolution2D,
    UpSampling2D,
    BatchNormalization,
)


tracking_uri = (
    "http://testuser:password@ec2-18-218-100-222.us-east-2.compute.amazonaws.com"
)
s3_bucket = "s3://docuedge-mlflow-bucket"  # replace this value


def generator():
    gen = Sequential()
    gen.add(Dense(256, input_dim=100))
    gen.add(LeakyReLU(0.2))
    gen.add(BatchNormalization(momentum=0.8))
    gen.add(Dense(512))
    gen.add(LeakyReLU(0.2))
    gen.add(BatchNormalization(momentum=0.8))
    gen.add(Dense(1024))
    gen.add(LeakyReLU(0.2))
    gen.add(BatchNormalization(momentum=0.8))
    gen.add(Dense(784, activation="tanh"))
    # gen.summary()

    noise = Input(shape=(100,))
    label = Input(shape=(1,), dtype="int32")
    label_embedding = Flatten()(Embedding(10, 100)(label))
    model_input = multiply([noise, label_embedding])
    image = gen(model_input)

    gen = Model([noise, label], image)
    gen.compile(loss="binary_crossentropy", optimizer=Adam(lr=0.0002, beta_1=0.5))
    return gen


def discriminator():
    disc = Sequential()
    disc.add(Dense(512, input_dim=784))
    disc.add(LeakyReLU(0.2))
    disc.add(Dropout(0.4))
    disc.add(Dense(512))
    disc.add(LeakyReLU(0.2))
    disc.add(Dropout(0.4))
    disc.add(Dense(512))
    disc.add(LeakyReLU(0.2))
    disc.add(Dropout(0.4))
    disc.add(Dense(1, activation="sigmoid"))
    # disc.summary()

    image = Input(shape=(784,))
    label = Input(shape=(1,), dtype="int32")
    label_embedding = Flatten()(Embedding(10, 784)(label))
    model_input = multiply([image, label_embedding])
    prediction = disc(model_input)

    disc = Model([image, label], prediction)
    disc.compile(
        loss="binary_crossentropy",
        optimizer=Adam(lr=0.0002, beta_1=0.5),
        metrics=["accuracy"],
    )
    return disc


def stacked_GAN(gen, disc):
    gan_input = Input(shape=(100,))
    label = Input(shape=(1,))
    x = gen([gan_input, label])
    disc.trainable = False
    gan_out = disc([x, label])
    gan_stack = Model([gan_input, label], gan_out)
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
    with mlflow.start_run(experiment_id=experiment.experiment_id):

        mlflow.log_metrics(
            {
                "batch_size": batch_size,
                "epochs": max_iter,
                "image_shape": img_shape,
            }
        )

        (X_train, y_train), (_, _) = mnist.load_data()
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = X_train.reshape(60000, 784)
        y_train = y_train.reshape(-1, 1)

        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        for i in range(max_iter):
            noise = np.random.normal(0, 1, (batch_size, 100))
            index = np.random.randint(0, X_train.shape[0], size=batch_size)
            image_batch = X_train[index]
            label_batch = y_train[index]

            fake_images = gen.predict([noise, label_batch])

            disc.trainable = True
            disc_loss_real = disc.train_on_batch([image_batch, label_batch], valid)
            disc_loss_fake = disc.train_on_batch([fake_images, label_batch], fake)
            disc_loss_final = 0.5 * np.add(disc_loss_real, disc_loss_fake)

            fake_labels = np.random.randint(0, 10, batch_size).reshape(-1, 1)
            disc.trainable = False
            gen_loss = gan_stack.train_on_batch([noise, fake_labels], valid)

            mlflow.log_metrics(
                {"generator_loss": gen_loss, "discriminator_loss": disc_loss_final[0]}
            )

            print(
                "epoch_%d---->gen_loss:[%f]---->disc_loss:[%f]---->acc:[%f]"
                % (i, gen_loss, disc_loss_final[0], disc_loss_final[1] * 100)
            )
            # if i % 100 == 0:
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
    default="conditional_gan",
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
