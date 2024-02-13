from flask import Flask, request, render_template, redirect, url_for
import tensorflow as tf
from werkzeug.utils import secure_filename
import os
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Assuming your models are in a folder named 'models'
MODEL_FOLDER = '../models'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model(model_name):
    model_path = os.path.join(MODEL_FOLDER, model_name)
    model = tf.keras.models.load_model(model_path)
    return model

def preprocess_image(image_path, target_size):
    """
    Preprocess the image: resize and scale pixel values.

    Args:
    - image_path (str): Path to the image to preprocess.
    - target_size (tuple): The target image size (width, height).

    Returns:
    - img_array: Preprocessed image array.
    """
    # Load the image
    img = image.load_img(image_path, target_size=target_size)
    
    # Convert the image to a numpy array
    img_array = image.img_to_array(img)
    
    # Scale pixel values to [0, 1]
    img_array /= 255.0

    # Expand dimensions to match the model's input format (add batch dimension)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        model_choice = request.form.get('model_selection')
        if file.filename == '' or not allowed_file(file.filename):
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            image_path = os.path.join('uploads', filename)
            file.save(image_path)
            
            # Load the selected model
            model = load_model(model_choice + '.h5')
            
            # Preprocess and classify the image
            # Implement preprocessing to fit your model's input requirements
            processed_image = preprocess_image(image_path, model.input_shape[1:3])
            prediction = model.predict(processed_image)
            
            # Map prediction to class names, if necessary
            # result = map_prediction_to_class(prediction)
            
            return render_template('result.html', result=prediction)
    
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)