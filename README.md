### Model Comparison and Deployment
This project is a Flask-based web application for image classification. Users can upload an image and select a pre-trained model to classify the image into predefined categories. The project demonstrates the integration of machine learning models into a web application, allowing for practical use of deep learning models.

#### Running the App Locally
1. cd serving
2. python3 -m venv venv
3. source venv/bin/activate (mac) OR ./venv/Scripts/activate (windows)
4. pip3 install -r requirements.txt
5. flask run

#### Usage

- **Uploading an Image**: On the home page, click "Choose File" to select an image file from your computer, select the model you wish to use for classification, and then click "Upload" to submit.
- **Viewing Results**: After processing, the classification results will be displayed on a new page. You can upload another image by clicking the provided link.

#### Models

The application currently supports the following models for image classification:

- **ResNet**: Short for Residual Networks, a classic architecture that utilizes skip connections or shortcuts to jump over some layers. These connections help solve the vanishing gradient problem and allow the model to be very deep, significantly increasing its performance on complex image classification tasks. Specifically, the version implemented is ResNet50, which has 50 layers deep.

- **DenseNet**: Stands for Densely Connected Convolutional Networks. Unlike ResNet, DenseNet connects each layer to every other layer in a feed-forward fashion. For each layer, the feature maps of all preceding layers are used as inputs, and its own feature maps are used as inputs into all subsequent layers. This connectivity pattern leads to improved efficiency in gradient propagation and reduces the number of parameters. The version used here is DenseNet121 with 121 layers.

- **EfficientNet**: This family of models uses a compound scaling method to uniformly scale network width, depth, and resolution with a set of fixed scaling coefficients. This allows the model to achieve higher accuracy with efficient use of computational resources. EfficientNetB0, the baseline model of this family, is included in the app, offering a good balance between speed and accuracy.

- **InceptionV3**: Part of the Inception family, this model introduces modules with parallel convolutions of different sizes, allowing it to capture information at various scales. The architecture also includes factorized 7x7 convolutions and an auxiliary classifier to propagate label information lower down the network, improving training speed and accuracy.
