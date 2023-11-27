from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from flask_cors import CORS
import numpy as np
import requests
from io import BytesIO

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
api = Api(app)

# Load the pre-trained model
model = load_model('keras_model.h5')


class ImageClassifier(Resource):
    @staticmethod
    def post():
        try:
            data = request.get_json()
            img_url = data.get('image_url')

            if not img_url:
                return {'error': 'Missing image_url parameter'}, 400

            # Load the image from the URL
            response = requests.get(img_url)
            img = image.load_img(BytesIO(response.content), target_size=(224, 224))

            # Preprocess the image
            img_array = image.img_to_array(img)
            img_array = preprocess_input(np.expand_dims(img_array, axis=0))

            # Make prediction
            predictions = model.predict(img_array)
            predicted_label = int(np.argmax(predictions))

            # Map numerical label to class name
            class_mapping = {0: 'equitation', 1: 'autre'}
            result_class = class_mapping.get(predicted_label, 'autre')

            return {'class': result_class}

        except Exception as e:
            return {'error': str(e)}


api.add_resource(ImageClassifier, '/predict')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
