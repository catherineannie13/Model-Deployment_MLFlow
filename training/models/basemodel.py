class BaseModel:
    def __init__(self, input_shape, num_classes):
        """
        Initializes a BaseModel object.

        Parameters:
        - input_shape (tuple): The shape of the input images.
        - num_classes (int): The number of classes for classification.

        Returns:
        - None
        """
        self.input_shape = input_shape
        self.num_classes = num_classes

    def unfreeze_top_layers(self, base_model, num_layers_to_finetune):
        """
        Unfreezes the top layers of a base model for fine-tuning.

        Parameters:
        - base_model: The base model to unfreeze layers from.
        - num_layers_to_finetune (int): The number of layers to unfreeze.

        Returns:
        - None
        """
        # Freeze all layers first
        for layer in base_model.layers:
            layer.trainable = False
        # Unfreeze the top 'num_layers_to_finetune' layers
        for layer in base_model.layers[-num_layers_to_finetune:]:
            layer.trainable = True