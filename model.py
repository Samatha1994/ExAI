from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.layers import Input, AveragePooling2D, Dropout, Flatten, Dense
from tensorflow.keras.models import Model

def create_model(num_classes):
    base_model = ResNet50V2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
    
    for layer in base_model.layers:
        layer.trainable = False

    head_model = base_model.output
    head_model = AveragePooling2D(pool_size=(7, 7))(head_model)
    head_model = Flatten(name="flatten")(head_model)
    head_model = Dense(64, activation="relu")(head_model)
    head_model = Dropout(0.5)(head_model)
    head_model = Dense(num_classes, activation="softmax")(head_model)

    model = Model(inputs=base_model.input, outputs=head_model)
    return model
