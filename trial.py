import requests

image_path = "/Users/vsatpathy/Desktop/off_POCs/intel-image-classification/seg_train/buildings/0.jpg"

with open(image_path, "rb") as f:
    image_bytes = f.read()

files = {
    "image": ("test_image", image_bytes),
}
url = "http://127.0.0.1:5000/predict_image"
# url = "https://bentoml.smartbox-capture.com/predict_document_labels_som"

response = requests.post(url, files=files)
print(response.text)
