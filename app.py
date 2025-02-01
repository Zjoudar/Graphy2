from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import cv2
import os
import time
import tensorflow as tf
from keras.applications.vgg16 import preprocess_input
import concurrent.futures
from flask import Flask, request,render_template, redirect,session
from flask_sqlalchemy import SQLAlchemy
import bcrypt


app = Flask(__name__)
global model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = load_model(os.path.join(BASE_DIR, 'finnalprject_Apps.h5))
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
db = SQLAlchemy(app)
app.secret_key = 'secret_key'

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(100))

    def __init__(self,email,password,name):
        self.name = name
        self.email = email
        self.password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    def check_password(self,password):
        return bcrypt.checkpw(password.encode('utf-8'),self.password.encode('utf-8'))

with app.app_context():
    db.create_all()

@app.route('/')
def index():
    return render_template('edit2.html')


@app.route('/register',methods=['GET','POST'])
def register():
    if request.method == 'POST':
        # handle request
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']

        new_user = User(name=name,email=email,password=password)
        db.session.add(new_user)
        db.session.commit()
        return redirect('/welcome')



    return render_template('register.html')

@app.route('/login',methods=['GET','POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        user = User.query.filter_by(email=email).first()
        
        if user and user.check_password(password):
            session['email'] = user.email
            return redirect('/welcome')
        else:
            return render_template('login.html',error='Invalid user')

    return render_template('login.html')


@app.route('/welcome')
def welcome():
    return render_template('welcome.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')






@app.route('/success', methods=['GET', 'POST'])
def success():
    error = ''

    if request.method == 'POST':
        if 'imagefile' not in request.files:
            return render_template('app.html', prediction='No image uploaded!')

        file = request.files['imagefile']

        if file.filename == '':
            return render_template('app.html', prediction='No selected file')

        if file and allowed_file(file.filename):
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(image_path) 

            try:
                start_time = time.time()
                image = cv2.imread(image_path)
                if image is None: 
                    return render_template('app.html', prediction='Error loading image') 
                img = cv2.resize(image, (128, 128)) 
                img = img_to_array(img)
                img = np.expand_dims(image, axis=0) 
                img = img / 255.0  
                image = preprocess_input(image)
                yhat = model.predict(image)
                predicted_class_index = np.argmax(yhat)
                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                    future = executor.submit(model.predict, image)
                    yhat = future.result()
                    predicted_class_index = np.argmax(yhat)
                predicted_class = "Predicted Class: " + str(predicted_class_index)  
                end_time = time.time()
                processing_time = end_time - start_time
                print(f"Image processing time: {processing_time:.4f} seconds") 
                return render_template('IndexApp.html', prediction=predicted_class)

            except (IOError, OSError) as e:
                print(f"Error processing image: {str(e)}")
                return render_template('app.html', prediction="Error processing image")

        else:
            error = "Please upload images of jpg, jpeg, and png extension only"

    else:
        error = "No file uploaded."

    return render_template('app.html', error=error)

if __name__ == "__main__":
    app.run()
