import argparse
import os
from hybrid_v1 import train_hybrid_v1
from pre_process_text import (
    pdf_to_images,
    read_text_from_pages,
)


def process(
    dataset_path: str,
    save_dir: str,
    pdf_check: bool,
    artifact_name: str,
    num_words_to_read: int,
):
    updated_dataset_path = os.path.join(save_dir, artifact_name, "dataset")
    os.makedirs(updated_dataset_path, exist_ok=True)
    for document_type in sorted(os.listdir(dataset_path)):
        if document_type != ".DS_Store":
            folder_path = os.path.join(dataset_path, document_type)
            updated_document_type_folder_path = os.path.join(
                updated_dataset_path, document_type
            )
            os.makedirs(updated_document_type_folder_path, exist_ok=True)
            for documents in sorted(os.listdir(folder_path)):
                if documents != ".DS_Store":
                    document_path = os.path.join(folder_path, documents)
                    if pdf_check:
                        # Perform conversion and store the images in a temp folder
                        pdf_to_images(
                            full_path_pdf=document_path,
                            converted_images_path=updated_document_type_folder_path,
                            meta_name=artifact_name,
                        )
    if pdf_check:
        images_data_path = os.path.join(save_dir, artifact_name)
    else:
        images_data_path = dataset_path
    master_data_path = read_text_from_pages(
        complete_folder_path=images_data_path,
        path_to_save_essential_data=save_dir,
        meta_name=artifact_name,
        num_of_words=num_words_to_read,
    )
    return master_data_path


def single_level(args):
    all_data_path = process(
        dataset_path=args.data_path,
        save_dir=args.file_path,
        pdf_check=bool(args.pdfs),
        artifact_name=args.art_name,
        num_words_to_read=int(args.num_of_words),
    )
    train_hybrid_v1(
        text_plus_file_path=all_data_path,
        batch_size=int(args.batch_size),
        epochs=int(args.epochs),
        image_shape=int(args.img_shape),
        max_words=int(args.num_of_words),
        artifact_name=args.art_name,
        save_dir_path=args.file_path,
        trained_model_path=args.model_path,
        experiment_name=args.experiment_name,
    )


parser = argparse.ArgumentParser()
parser.add_argument("-dp", "--data_path", help="File path of the dataset")
parser.add_argument("-fp", "--file_path", help="Directory path to save artifacts")
parser.add_argument("-a", "--art_name", help="Artifacts name")
parser.add_argument("-p", "--pdfs", default=False, help="Dataset type")
parser.add_argument("-n", "--num_of_words", default=10, help="No of words to read")
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
single_level(args=args)  # For single level document classification
