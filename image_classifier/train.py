import os
import random
import numpy as np
import mlflow
from mlflow import pyfunc
import argparse
import json

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    Dense,
    Flatten,
)

tracking_uri = "http://testuser:test@ec2-18-220-228-243.us-east-2.compute.amazonaws.com"
s3_bucket = "s3://docuedge-mlflow-bucket"  # replace this value


def model_arc(y_labels: dict, image_inp_shape: tuple):
    inp_layer_images = Input(shape=image_inp_shape)

    conv_layer = Conv2D(filters=64, kernel_size=(2, 2), activation="relu")(
        inp_layer_images
    )
    flatten_layer = Flatten()(conv_layer)

    out_layer = Dense(len(y_labels), activation="softmax")(flatten_layer)

    model = Model(inp_layer_images, out_layer)
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def data_loader(gt_data_path: list, gt_labels: dict, bs: int, image_shape: tuple):
    while True:
        images = []
        labels = []
        while len(images) < bs:
            indice = random.randint(0, len(gt_data_path) - 1)
            image_path = gt_data_path[indice]

            label = gt_labels.get(image_path.split("/")[-2])
            labels.append(label)

            test_img = np.asarray(load_img(image_path, target_size=image_shape))
            img = np.divide(test_img, 255.0)
            images.append(img)
        yield np.asarray(images), np.asarray(labels)


def read_data(data_path: str):
    folders = os.listdir(data_path)

    all_images_paths = []
    all_labels = {}
    for label in folders:
        if label != ".DS_Store":
            images = os.path.join(data_path, label)
            for image in os.listdir(images):
                full_image_path = os.path.join(images, image)
                all_images_paths.append(full_image_path)
            if label not in all_labels:
                all_labels[label] = len(all_labels)
    rev_labels = {}
    for key, val in all_labels.items():
        rev_labels[val] = key
    return all_images_paths, all_labels, rev_labels


def train(
    image_shape: int,
    epochs: int,
    batch_size: int,
    data_path: str,
    save_dir_path: str,
    art_name: str,
    exp_name: str,
    trained_model_path: str,
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

    mlflow.tensorflow.autolog(every_n_iter=1)
    with mlflow.start_run(experiment_id=experiment.experiment_id):
        image_dimensions = (image_shape, image_shape, 3)
        no_of_epochs = epochs
        batch_size = batch_size
        dataset_path = data_path
        gt_image_paths, gt_labels, gt_rev_labels = read_data(data_path=dataset_path)
        os.makedirs(os.path.join(save_dir_path, art_name), exist_ok=True)

        mlflow.log_metrics(
            {
                "batch_size": batch_size,
                "epochs": epochs,
                "image_shape": image_shape,
            }
        )

        with open(
            os.path.join(save_dir_path, art_name, f"rev_labels_{art_name}.json"),
            "w+",
        ) as tar:
            json.dump(gt_rev_labels, tar)

        print("target_encodings: ", gt_labels)
        print("Number of training images: ", len(gt_image_paths))

        train_gen = data_loader(
            gt_data_path=gt_image_paths,
            gt_labels=gt_labels,
            bs=batch_size,
            image_shape=image_dimensions,
        )

        if os.path.isfile(trained_model_path):
            model = load_model(trained_model_path)
        else:
            model = model_arc(y_labels=gt_labels, image_inp_shape=image_dimensions)
        model.fit(
            x=train_gen,
            steps_per_epoch=len(gt_image_paths) // batch_size,
            epochs=no_of_epochs,
        )
        model.save(
            filepath=os.path.join(save_dir_path, art_name, f"image_classifier.h5")
        )

        meta_data_path = os.path.join(save_dir_path, art_name)
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
parser.add_argument("-dp", "--data_path", help="File path of the dataset")
parser.add_argument("-fp", "--file_path", help="Directory path to save artifacts")
parser.add_argument("-a", "--art_name", help="Artifacts name")
parser.add_argument("-b", "--batch_size", default=8, help="Batch size for training")
parser.add_argument("-e", "--epochs", default=3, help="Number of epochs")
parser.add_argument("-is", "--img_shape", default=100, help="One dimension of image")
parser.add_argument("-mp", "--model_path", default="NULL", help="Path to trained model")
parser.add_argument(
    "-exp",
    "--experiment_name",
    default="test_experiment",
    help="Name of the experiment for tracking",
)
args = parser.parse_args()
train(
    image_shape=args.img_shape,
    epochs=args.epochs,
    batch_size=args.batch_size,
    data_path=args.data_path,
    save_dir_path=args.file_path,
    art_name=args.art_name,
    exp_name=args.experiment_name,
    trained_model_path=args.model_path,
)
