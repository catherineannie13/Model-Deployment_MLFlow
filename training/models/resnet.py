from .basemodel import BaseModel
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

class ResNetModel(BaseModel):
    def __init__(self, input_shape, num_classes):
        """
        Initializes a ResNetModel object.

        Parameters:
        - input_shape (tuple): The shape of the input data.
        - num_classes (int): The number of classes for classification.

        Returns:
        - None
        """
        super().__init__(input_shape, num_classes)
        
    def build(self, num_layers_to_finetune=10):
        """
        Builds the ResNet model.

        Parameters:
        - num_layers_to_finetune (int): The number of layers to finetune.

        Returns:
        - model (keras.Model): The built ResNet model.
        """
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=self.input_shape)
        self.unfreeze_top_layers(base_model, num_layers_to_finetune)

        inputs = Input(shape=self.input_shape)
        x = base_model(inputs, training=True) 
        x = GlobalAveragePooling2D()(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)

        model = Model(inputs, outputs)
        return model