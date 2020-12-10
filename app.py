"""
    Referensi:
    - https://pytorch.org/tutorials/intermediate/flask_rest_api_tutorial.html
    - https://github.com/avinassh/pytorch-flask-api
"""

import io
import json

import torch as pt
import torchvision.transforms as transforms

from PIL import Image

from flask import Flask, jsonify, request
from flask_cors import CORS

class_index = {
    0: "Angka 1",
    1: "Angka 2",
    2: "Angka 3",
    3: "Angka 4",
    4: "Angka 5"
}
model = pt.load("model.pth", map_location=pt.device("cpu"))
model.eval()


def transform_image(image_bytes):
    test_transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize([0.3875, 0.3815, 0.3621],[0.2459, 0.2397, 0.2395])
    ])
    image = Image.open(io.BytesIO(image_bytes))
    return test_transform(image)


def get_prediction(image_bytes):
    tensor_img = transform_image(image_bytes=image_bytes)
    outputs = model(tensor_img.unsqueeze(0))
    predicted_idx = outputs.max(1).indices

    return class_index[predicted_idx.item()]


app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        class_name = get_prediction(image_bytes=img_bytes)
        return jsonify({'class_name': class_name})


@app.route('/test', methods=['GET'])
def test():
    return jsonify({'hello':'test success'})

if __name__ == '__main__':
    app.run()