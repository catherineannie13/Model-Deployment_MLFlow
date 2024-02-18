from flask import Flask, request, render_template, redirect, url_for
import tensorflow as tf
from werkzeug.utils import secure_filename
import os
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model(model_name):
    """
    Loads a pre-trained model.

    Parameters:
    - model_name (str): The name of the model to load.

    Returns:
    - model (tf.keras.Model): The loaded model.
    """
    full_model_name = model_name + '_ROC_AUC_ES_RLRP_saved'
    base_dir = 'C:/Users/cathe/model_deployment_DL/serving/models'
    model_path = os.path.join(base_dir, full_model_name)
    model = tf.keras.models.load_model(model_path)
    return model

def preprocess_image(image_path, target_size):
    """
    Preprocesses an image for model input.

    Parameters:
    - image_path (str): The path to the image file.
    - target_size (tuple): The desired size of the image after resizing.

    Returns:
    - img_array (numpy.ndarray): The preprocessed image as a numpy array.
    """
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array /= 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    """
    Handles the file upload and classification.

    Returns:
    - If the request method is POST and the file is successfully uploaded and classified, it returns the rendered template with the predicted fungus name.
    - If the request method is GET or the file upload fails, it returns the rendered upload template.
    """
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        model_choice = request.form.get('model_selection')
        if file.filename == '' or not allowed_file(file.filename):
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            
            # Ensure the uploads directory exists
            uploads_dir = os.path.join(app.root_path, 'uploads')
            os.makedirs(uploads_dir, exist_ok=True) 
            
            image_path = os.path.join(uploads_dir, filename)
            file.save(image_path)
            
            # Load the selected model
            model = load_model(model_choice)
            
            # Preprocess and classify the image
            # Implement preprocessing to fit your model's input requirements
            processed_image = preprocess_image(image_path, model.input_shape[1:3])
            prediction = model.predict(processed_image)
            
            # Map prediction to fungus name
            mapping = {0: 'Tortuous septate hyaline hyphae (TSH)', 
                       1: 'Beaded arthroconidial septate hyaline hyphae (BASH)', 
                       2: 'Groups or mosaics of arthroconidia (GMA)', 
                       3: 'Septate hyaline hyphae with chlamydioconidia (SHC)', 
                       4: 'Broad brown hyphae (BBH)'}
            predicted_class_id = np.argmax(prediction, axis=1)[0]  
            fungus = mapping[predicted_class_id]
            
            return render_template('result.html', result=fungus)
    
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)