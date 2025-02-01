from flask import Flask, render_template, request

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
#from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
import os 
from tensorflow.keras.models import load_model
import numpy as np


from flask import Flask, render_template, request
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input  # Assuming ResNet50

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = load_model(os.path.join(BASE_DIR, 'finnalproject_Apps.h5'))
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'} 

app.config['UPLOAD_FOLDER'] = 'uploads' 

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
def hello_word():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    if 'imagefile' not in request.files:
        return render_template('index.html', prediction='No image uploaded!')

    imageFile = request.files['imagefile']

    if imageFile.filename == '':
        return render_template('index.html', prediction='No selected file')

    if imageFile and allowed_file(imageFile.filename):  # Assuming you have an 'allowed_file' function for validation
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], imageFile.filename)  # Assuming you have an 'UPLOAD_FOLDER' config variable
        imageFile.save(image_path)

        try:
            image = load_img(image_path, target_size=(224, 224))
            image = img_to_array(image)
            image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))  # Handle potential grayscale images
            image = preprocess_input(image)
            yhat = model.predict(image)

            # Handle model output format (assuming single class label)
            predicted_class_index = np.argmax(yhat)
            predicted_class = "Predicted Class: " + str(predicted_class_index)  # Replace with your class labels if needed

            return render_template('index.html', prediction=predicted_class)

        except Exception as e:
            return render_template('index.html', prediction=f"Error processing image: {str(e)}")

    else:
        return render_template('index.html', prediction='Invalid image format')

if __name__ == '__main__':
    app.config['UPLOAD_FOLDER'] = 'uploads'  # Assuming you have an 'uploads' directory for storing images
    app.run(port=3000, debug=True)
