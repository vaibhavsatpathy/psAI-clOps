import spacy
import re
from pdf2image import convert_from_path
import os
from tqdm import tqdm
import pre_processing
from PIL import Image
import pytesseract

nlp = spacy.load("en_core_web_sm")


def helper(text):
    dummy = []
    for word in text:
        dummy.append(str(word))
    final = " ".join(dummy)
    return final


def preprocess_spacy(text, num_of_words: int):
    text = str(text)
    text = text.split(" ")
    text = helper(text)
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
    text = nlp(text)
    filtered = [token.lemma_ for token in text if token.is_stop == False]
    text = " ".join(filtered[: num_of_words * 2])
    text = text.strip().split(" ")
    text = " ".join(text[:num_of_words])
    return text


def read_text_from_pages_v2(
    complete_folder_path: str,
    path_to_save_essential_data: str,
    meta_name: str,
    num_of_words: int,
):
    final_path_for_data = os.path.join(
        path_to_save_essential_data, meta_name, f"file_and_text_{meta_name}.txt"
    )
    if os.path.isfile(final_path_for_data):
        data = open(final_path_for_data, "r").read()
    else:
        data = "null"
    print("####  Reading pages ####")
    document_folders_path = os.path.join(complete_folder_path)
    master_doc_types = sorted(os.listdir(document_folders_path))
    text_of_all_pages = []
    for master_doc_type in master_doc_types:
        if master_doc_type != ".DS_Store":
            print("MASTER DOCUMENT TYPE: ", master_doc_type)
            sub_doc_type_path = os.path.join(document_folders_path, master_doc_type)
            for doc_image_type in sorted(os.listdir(sub_doc_type_path)):
                if doc_image_type != ".DS_Store":
                    print("DOCUMENT TYPE: ", doc_image_type)
                    complete_doc_image_path = os.path.join(
                        sub_doc_type_path, doc_image_type
                    )
                    pages = sorted(os.listdir(complete_doc_image_path))
                    for page in tqdm(pages):
                        if page != ".DS_Store":
                            page_path = os.path.join(complete_doc_image_path, page)
                            if page_path not in data:
                                document_page = Image.open(page_path)
                                document_text = pytesseract.image_to_string(
                                    document_page
                                )
                                document_page.close()
                                essential_file_path_and_text = (
                                    page_path
                                    + "####"
                                    + preprocess_spacy(
                                        document_text, num_of_words=num_of_words
                                    )
                                    + "\n"
                                )
                                text_of_all_pages.append(essential_file_path_and_text)

    if os.path.isfile(final_path_for_data):
        all_essential_data = open(final_path_for_data, "a+")
        all_essential_data.writelines(text_of_all_pages)
    else:
        all_essential_data = open(final_path_for_data, "w")
        all_essential_data.writelines(text_of_all_pages)
    return final_path_for_data


def pdf_to_images(full_path_pdf: str, converted_images_path: str, meta_name: str):
    doc = full_path_pdf.split("/")[-1]
    index = 0
    OUTPUT_PATH = converted_images_path
    os.makedirs(name=OUTPUT_PATH, exist_ok=True)

    print("Document name: ", doc)
    if str(doc.split(".pdf")[-2]) + "_" + str(index) + ".jpg" not in os.listdir(
        converted_images_path
    ):
        pil_images = convert_from_path(full_path_pdf, dpi=300)

        for image in tqdm(pil_images):
            processed_image = pre_processing.preprocess_image_file(image)
            try:
                processed_image = Image.fromarray(processed_image)
                processed_image.save(
                    os.path.join(OUTPUT_PATH, str(doc.split(".pdf")[-2]))
                    + "_"
                    + str(index)
                    + ".jpg",
                    format="JPEG",
                    subsampling=0,
                    quality=100,
                )
                index += 1
                processed_image.close()
            except:
                index += 1
    else:
        pass