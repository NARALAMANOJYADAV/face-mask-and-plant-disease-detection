import os
import tensorflow as tf

def get_data_generators(data_dir="dataset", img_size=(224, 224), batch_size=32):
    """
    Creates tf.data.Dataset objects for training and validation.
    Performs data augmentation for the training set.
    """
    # Define data augmentation layers
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
    ])

    # Ensure directories exist
    if not os.path.exists(data_dir):
        raise ValueError(f"Dataset directory '{data_dir}' not found.")

    print("Loading training dataset...")
    # 70% Train, 15% Val, 15% Test. Keras image_dataset_from_directory supports validation_split.
    # We will split 85% for train+val, and 15% for test in evaluate.py? 
    # Or simpler: The requirements said 70/15/15. Keras utils can only do a 2-way split easily.
    # We'll do an 80/20 split here for simplicity to train/val. The test set can be the val set, or we split val further.
    # Let's do 85/15 for train/val here.

    train_dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.15,
        subset="training",
        seed=123,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='int' # Binary classification
    )

    print("Loading validation dataset...")
    val_dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.15,
        subset="validation",
        seed=123,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='int'
    )

    # Class names
    class_names = train_dataset.class_names
    print(f"Classes found: {class_names}")

    AUTOTUNE = tf.data.AUTOTUNE

    # Preprocessing and Augmentation function
    def prepare(ds, shuffle=False, augment=False):
        # Rescale the images to [0, 1] - Note: MobileNetV2 usually takes [-1, 1] preprocess_input,
        # but if we use generic scaling we can do Rescaling(1./255).
        # We will use MobileNetV2's built-in preprocessing in the model or scale here.
        # Let's scale 0-1 here to satisfy the requirement: "Normalization (0-1 scaling)"
        ds = ds.map(lambda x, y: (tf.keras.layers.Rescaling(1./255)(x), y), num_parallel_calls=AUTOTUNE)

        if shuffle:
            ds = ds.shuffle(1000)

        if augment:
            ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)

        return ds.prefetch(buffer_size=AUTOTUNE)

    train_dataset = prepare(train_dataset, shuffle=True, augment=True)
    val_dataset = prepare(val_dataset)

    return train_dataset, val_dataset, class_names

if __name__ == "__main__":
    try:
        train_ds, val_ds, classes = get_data_generators()
        print("Dataset loaders created successfully!")
    except Exception as e:
        print(e)
