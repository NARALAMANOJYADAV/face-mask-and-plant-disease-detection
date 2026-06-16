import os
import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import sys

# Ensure utils can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.helper import plot_confusion_matrix

def evaluate():
    model_path = "models/best_model.h5"
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please train the model first.")
        return

    # Load Model
    print("Loading model...")
    model = tf.keras.models.load_model(model_path)

    # Load Validation/Test Data
    # For this script we will just evaluate on the validation dataset 
    # to demonstrate the metrics without requiring a 3rd distinct directory structure
    print("Loading test data...")
    val_dataset = tf.keras.utils.image_dataset_from_directory(
        "dataset",
        validation_split=0.15,
        subset="validation",
        seed=123,
        image_size=(224, 224),
        batch_size=32,
        label_mode='int'
    )
    class_names = val_dataset.class_names

    # Preprocess
    val_dataset = val_dataset.map(lambda x, y: (tf.keras.layers.Rescaling(1./255)(x), y))

    print("Evaluating model...")
    results = model.evaluate(val_dataset)
    print(f"Test Loss: {results[0]:.4f}")
    print(f"Test Accuracy: {results[1]:.4f}")

    # Generate Predictions for Confusion Matrix and Classification Report
    print("Generating predictions...")
    y_true = []
    y_pred = []
    
    for images, labels in val_dataset:
        preds = model.predict(images, verbose=0)
        # Convert probabilities to binary predictions
        bin_preds = [1 if p > 0.5 else 0 for p in preds]
        y_true.extend(labels.numpy())
        y_pred.extend(bin_preds)

    print("\n--- Classification Report ---")
    print(classification_report(y_true, y_pred, target_names=class_names))

    print("--- Confusion Matrix ---")
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, class_names)

if __name__ == "__main__":
    evaluate()
