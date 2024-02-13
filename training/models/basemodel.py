class BaseModel:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def unfreeze_top_layers(self, base_model, num_layers_to_finetune):
        """
        Unfreeze the top 'num_layers_to_finetune' layers of the model.
        """
        # Freeze all layers first
        for layer in base_model.layers:
            layer.trainable = False
        # Unfreeze the top 'num_layers_to_finetune' layers
        for layer in base_model.layers[-num_layers_to_finetune:]:
            layer.trainable = True