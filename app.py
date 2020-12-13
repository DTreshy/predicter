from flask import Flask, request, jsonify
from io import BytesIO
from keras.models import load_model
from PIL import Image
from flask_cors import CORS
import tensorflow as tf
import keras
import numpy as np

app = Flask(__name__)
CORS(app)
model = load_model('model', compile=True)
labels = ['triangle', 'rectangle', 'circle']
#label = [76]


@app.route('/', methods=["POST"])
def predict():

    image = request.files['image']
    image = Image.open(image)
    image.convert('P')
    image = image.resize((50, 50))
    #image.save('rectangle' + str(label[0]) + '.png')
    #label[0] += 1
    #return '0'
    image = np.asarray(image)
    image = image / 255
    img = np.zeros((1, 50, 50))
    for i in range(0, 50):
        for j in range(0, 50):
            img[0][i][j] = image[i][j][3]

    predictions = model(img)
    prediction = np.argmax(predictions, axis=1)
    propability = np.max(predictions)
    print(prediction)
    print(propability)
    return jsonify(prediction=labels[int(str(prediction)[1])], propability=round(float(propability*100), 1))


if __name__ == '__main__':
    app.run(host='0.0.0.0')
