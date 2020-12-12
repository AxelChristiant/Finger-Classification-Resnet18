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
import math

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
    preds = outputs.squeeze().tolist()
    preds = get_scaled(preds)
    preds = dict(sorted(preds.items(), key=lambda x: x[1], reverse=True))

    return preds




def min_max_scale(x,xmax,xmin) :
    x_scaled = (x - xmin)/(xmax-xmin)
    return x_scaled


def get_scaled(preds) :
    preds_max = max(preds)
    preds_min = min(preds)
    scaled_preds={}
    for i,pred in enumerate(preds) :
        result = min_max_scale(pred,preds_max,preds_min) * 100
        scaled_preds["angka_"+str(i+1)] = result
    return scaled_preds



app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        class_name = get_prediction(image_bytes=img_bytes)
        return jsonify(class_name)


@app.route('/test', methods=['GET'])
def test():
    return jsonify({'hello':'test success'})

if __name__ == '__main__':
    app.run()