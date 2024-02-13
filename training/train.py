import mlflow
import mlflow.tensorflow
from data_preparation.preprocess import split_data, create_data_generators
from models.densenet import DenseNetModel
import tensorflow as tf

def train_model(model, model_name):
    # Initialize MLFlow experiment
    mlflow.set_experiment(model_name)

    with mlflow.start_run():
        # Split dataset into training, validation, and test sets
        # data_path = '../dataset/defungi'
        # split_data(source_dir=data_path, target_dir=data_path)

        # Define directories for train, val, and test datasets
        train_dir = '../dataset/defungi/train'
        val_dir = '../dataset/defungi/val'
        test_dir = '../dataset/defungi/test'

        # Model and training parameters
        image_size = (224, 224)
        batch_size = 32
        num_classes = 5 
        learning_rate = 1e-4
        epochs = 10

        # Log parameters
        mlflow.log_param("image_size", image_size)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("num_classes", num_classes)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("epochs", epochs)

        # Create data generators
        train_generator, val_generator, test_generator = create_data_generators(
            train_dir=train_dir,
            val_dir=val_dir,
            test_dir=test_dir,
            image_size=image_size,
            batch_size=batch_size
        )

        # Build the model
        model_ = model.build(num_layers_to_finetune=10)

        # Compile the model
        model_.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

        # Log model summary
        mlflow.tensorflow.log_model(model_, "model")

        # Train the model
        history = model_.fit(train_generator,
                            epochs=epochs,
                            validation_data=val_generator)

        # Log metrics
        mlflow.log_metric("train_accuracy", history.history['accuracy'][-1])
        mlflow.log_metric("val_accuracy", history.history['val_accuracy'][-1])

        # Save MLFlow run
        mlflow.end_run()


if __name__ == "__main__":
    # Define input shape and number of classes
    input_shape = (224, 224, 3)
    num_classes = 5

    # Train the model
    model = DenseNetModel(input_shape=input_shape, num_classes=num_classes)
    model_name = "DenseNet"
    train_model(model, model_name)