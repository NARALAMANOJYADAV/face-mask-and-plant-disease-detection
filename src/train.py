import os
import tensorflow as tf
from data_preprocessing import get_data_generators
from model import build_model
import sys

# Ensure utils can be imported if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.helper import plot_training_history

def train():
    # Model parameters
    epochs = 10
    batch_size = 32

    # Get data
    print("Initializing data pipelines...")
    train_dataset, val_dataset, class_names = get_data_generators(
        data_dir="dataset", batch_size=batch_size
    )

    # Build model
    print("Building model...")
    model = build_model()

    # Compile model
    # Loss: BinaryCrossentropy, Optimizer: Adam, Metrics: Accuracy
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=['accuracy']
    )

    # Callbacks
    if not os.path.exists("models"):
        os.makedirs("models")

    # EarlyStopping and ModelCheckpoint
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=3, restore_best_weights=True
    )
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath='models/best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )

    # Train
    print(f"Starting training for {epochs} epochs...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=[early_stop, checkpoint]
    )

    print("Training complete. Best model saved to 'models/best_model.h5'")
    
    # Evaluate & Plot
    plot_training_history(history)

if __name__ == "__main__":
    train()
