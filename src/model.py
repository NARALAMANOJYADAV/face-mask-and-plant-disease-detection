import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model

def build_model(input_shape=(224, 224, 3)):
    """
    Builds a transfer learning model based on MobileNetV2.
    """
    # Load pretrained MobileNetV2 without the top classification layer
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )

    # Freeze the base model
    base_model.trainable = False

    # Create the custom head
    inputs = tf.keras.Input(shape=input_shape)
    
    # We apply the base model
    x = base_model(inputs, training=False)
    
    # Add new classification layers
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid')(x) # Binary classification (Mask / No Mask)

    # Combine into a single complete model
    model = Model(inputs, outputs)

    return model

if __name__ == "__main__":
    model = build_model()
    model.summary()
