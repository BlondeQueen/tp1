from flask import Flask, request, jsonify
import tensorflow as tf  # noqa: F401 (tensorflow importé si besoin backend)
from tensorflow import keras
import numpy as np

app = Flask(__name__)

# Chargement du modele Keras
model = keras.models.load_model('mnist_model.h5')

@app.route('/predict', methods=['POST'])
def predict():
	data = request.json
	# Verification des donnees
	if 'image' not in data:
		return jsonify({'error': 'No image provided'}), 400
	image_data = np.array(data['image'])
	# Assurer le format (1, 784) et normalisation
	image_data = image_data.reshape(1, 784)
	image_data = image_data.astype('float32') / 255.0
	prediction = model.predict(image_data)
	predicted_class = np.argmax(prediction, axis=1)[0]
	return jsonify({
		'prediction': int(predicted_class),
		'probabilities': prediction.tolist()
	})

if __name__ == '__main__':
	app.run(host='0.0.0.0', port=5000)
