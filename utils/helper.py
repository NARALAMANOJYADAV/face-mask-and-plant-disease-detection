import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_training_history(history):
    """
    Plots the training and validation accuracy and loss.
    """
    acc = history.history.get('accuracy', [])
    val_acc = history.history.get('val_accuracy', [])
    loss = history.history.get('loss', [])
    val_loss = history.history.get('val_loss', [])

    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b', label='Training accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('models/training_history.png')
    plt.show()

def plot_confusion_matrix(cm, class_names):
    """
    Plots a confusion matrix using seaborn.
    """
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.savefig('models/confusion_matrix.png')
    plt.show()

def generate_dummy_data():
    """
    Generates dummy colored images for testing the pipeline 
    without downloading a real dataset.
    """
    base_dir = "dataset"
    classes = ["with_mask", "without_mask"]
    num_samples_per_class = 50

    print("Generating dummy image data for pipeline testing...")
    
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    for cls in classes:
        cls_dir = os.path.join(base_dir, cls)
        if not os.path.exists(cls_dir):
            os.makedirs(cls_dir)
        
        for i in range(num_samples_per_class):
            # Create a random colored image
            img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            
            # Draw a simple shape to differentiate classes slightly
            if cls == "with_mask":
                # Draw a blue rectangle simulating a mask
                cv2.rectangle(img, (60, 120), (164, 200), (255, 0, 0), -1)
            else:
                # Draw a pink circle simulating a face
                cv2.circle(img, (112, 112), 60, (200, 150, 200), -1)

            file_path = os.path.join(cls_dir, f"dummy_{i}.jpg")
            cv2.imwrite(file_path, img)

    print(f"Successfully generated 100 dummy images in {base_dir}/")

if __name__ == "__main__":
    generate_dummy_data()
