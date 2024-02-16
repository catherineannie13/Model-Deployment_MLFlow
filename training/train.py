import tensorflow as tf
import os
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import numpy as np
import mlflow
from data_preparation.preprocess import create_data_generators
from models.densenet import DenseNetModel
from models.efficientnet import EfficientNetModel
from models.resnet import ResNetModel
from models.inceptionv3 import InceptionV3Model

class RocAucCallback(tf.keras.callbacks.Callback):
    def __init__(self, train_data, val_data, num_classes):
        super(RocAucCallback, self).__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.num_classes = num_classes

    def calculate_metrics(self, model, generator, steps, num_classes):
        y_true = np.zeros((0, num_classes))
        y_pred = np.zeros((0, num_classes))
        
        for _ in range(steps):
            x_batch, y_batch = next(generator)
            y_true = np.vstack([y_true, y_batch])
            y_pred = np.vstack([y_pred, model.predict(x_batch)])
        
        roc_auc_scores = [roc_auc_score(y_true[:, i], y_pred[:, i]) for i in range(num_classes)]
        roc_auc = np.mean(roc_auc_scores)
        
        # Convert probabilities to binary predictions
        y_pred_binary = np.argmax(y_pred, axis=1)
        y_true_binary = np.argmax(y_true, axis=1)
        
        f1 = f1_score(y_true_binary, y_pred_binary, average='macro')
        precision = precision_score(y_true_binary, y_pred_binary, average='macro')
        recall = recall_score(y_true_binary, y_pred_binary, average='macro')
        
        return roc_auc, f1, precision, recall

    def on_epoch_end(self, epoch, logs=None):
        train_roc_auc, train_f1, train_precision, train_recall = self.calculate_metrics(self.model, self.train_data, len(self.train_data), self.num_classes)
        val_roc_auc, val_f1, val_precision, val_recall = self.calculate_metrics(self.model, self.val_data, len(self.val_data), self.num_classes)
        
        metrics = {
            "train_roc_auc": train_roc_auc,
            "val_roc_auc": val_roc_auc,
            "train_f1": train_f1,
            "val_f1": val_f1,
            "train_precision": train_precision,
            "val_precision": val_precision,
            "train_recall": train_recall,
            "val_recall": val_recall
        }
        
        for name, value in metrics.items():
            mlflow.log_metric(name, value, step=epoch)
        
        print(f"Epoch {epoch + 1}: {metrics}")


def train_model(model, model_name, train_dir, val_dir, test_dir, image_size, batch_size, num_classes, learning_rate, epochs):
    mlflow.set_experiment(model_name)
    
    with mlflow.start_run(run_name=model_name):
        mlflow.log_param("image_size", image_size)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("num_classes", num_classes)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("epochs", epochs)

        train_generator, val_generator, _ = create_data_generators(
            train_dir=train_dir,
            val_dir=val_dir,
            test_dir=test_dir,
            image_size=image_size,
            batch_size=batch_size
        )

        model_ = model.build(num_layers_to_finetune=10)

        model_.compile(optimizer=Adam(learning_rate=learning_rate),
                       loss='categorical_crossentropy',
                       metrics=['accuracy'])
        model_.compile(optimizer=Adam(learning_rate=learning_rate),
                       loss='categorical_crossentropy',
                       metrics=['accuracy'])

        roc_auc_callback = RocAucCallback(train_generator, val_generator, num_classes)
        
        # ModelCheckpoint callback to save the model
        checkpoint_dir = 'models/checkpoints/' + model_name
        checkpoint_filepath = os.path.join(checkpoint_dir, 'model_checkpoint.h5')
        model_checkpoint_callback = ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=False,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True
        )
        
        # Early Stopping callback
        early_stopping_callback = EarlyStopping(
            monitor='val_loss',
            patience=3,  # Adjust patience as needed
            restore_best_weights=True
        )
        
        # Reduce learning rate on plateau
        reduce_lr_callback = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            patience=2,
            min_lr=1e-6
        )

        model_.fit(train_generator,
                   epochs=epochs,
                   validation_data=val_generator,
                   callbacks=[roc_auc_callback, model_checkpoint_callback, early_stopping_callback, reduce_lr_callback])

        # Save the model
        # Assuming you have already defined `checkpoint_filepath`, `model_name`, and other variables

        # Load the best model from checkpoint
        best_model = tf.keras.models.load_model(checkpoint_filepath)

        # Define the directory path where the model will be saved
        model_save_path = '../serving/models/' + model_name + '_saved'
        os.makedirs(model_save_path, exist_ok=True)  # Ensure the directory exists

        # Save the best model using the Keras native format (TF SavedModel format by default)
        # Note: No need to specify 'saved_model' subdirectory, TensorFlow handles it
        tf.keras.models.save_model(best_model, model_save_path)

        # For MLflow, if you are logging the model structure without weights and want a fresh clone
        # Consider directly logging the best model instead if applicable
        # Clone the model structure (this does not clone weights)
        model_clone = tf.keras.models.clone_model(best_model)

        # Log the cloned model structure with MLflow (consider logging `best_model` directly if applicable)
        mlflow.tensorflow.log_model(model_clone, "model_structure")
        # End the MLflow run
        mlflow.end_run()

if __name__ == "__main__":
    train_dir = '../dataset/defungi/train'
    val_dir = '../dataset/defungi/val'
    test_dir = '../dataset/defungi/test'
    image_size = (224, 224)
    batch_size = 32
    train_dir = '../dataset/defungi/train'
    val_dir = '../dataset/defungi/val'
    test_dir = '../dataset/defungi/test'
    image_size = (224, 224)
    batch_size = 32
    num_classes = 5
    learning_rate = 1e-4
    epochs = 10
    model = ResNetModel(input_shape=(224, 224, 3), num_classes=num_classes)
    model_name = "ResNet_ROC_AUC_ES_RLRP"
    train_model(model, model_name, train_dir, val_dir, test_dir, image_size, batch_size, num_classes, learning_rate, epochs)