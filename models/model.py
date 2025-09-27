import tensorflow as tf
from tensorflow.keras import layers, models

def build_model(cfg):
    return models.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(cfg["hidden_units"], activation="relu"),
        layers.Dropout(cfg["dropout"]),
        layers.Dense(10, activation="softmax")
    ])
