import tensorflow as tf
import json

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = json.load(file)
    return config

def create_data_generators(config):
    image_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )
    
    image_generator_validation = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    train_dataset = image_generator.flow_from_directory(
        batch_size=config["batch_size"],
        directory=config["train_directory"],
        shuffle=True,
        target_size=tuple(config["image_size"]),
        subset="training",
        class_mode='categorical'
    )

    validation_dataset = image_generator_validation.flow_from_directory(
        batch_size=config["batch_size"],
        directory=config["validation_directory"],
        shuffle=False,
        target_size=tuple(config["image_size"]),
        class_mode='categorical'
    )

    return train_dataset, validation_dataset
