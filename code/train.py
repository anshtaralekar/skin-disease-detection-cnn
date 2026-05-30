import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing import image
import os

# ---------------- AUGMENTATION ----------------
def augment_dataset(dataset_path):
    print("Starting augmentation...")

    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
    ])

    for class_name in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_name)

        if os.path.isdir(class_path):
            for img_name in os.listdir(class_path):

                if img_name.startswith("aug_"):
                    continue

                img_path = os.path.join(class_path, img_name)

                img = image.load_img(img_path, target_size=(180,180))
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)

                for i in range(4):
                    augmented_img = data_augmentation(img_array)
                    augmented_img = augmented_img[0].numpy().astype("uint8")

                    new_name = f"aug_{i}_{img_name}"
                    new_path = os.path.join(class_path, new_name)

                    image.array_to_img(augmented_img).save(new_path)

    print("Augmentation Done!")

# ---------------- RUN AUGMENTATION (ONLY FIRST TIME) ----------------
augment_dataset("dataset")

# ---------------- LOAD DATA ----------------
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset",
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(180, 180),
    batch_size=16
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset",
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(180, 180),
    batch_size=16
)

class_names = train_ds.class_names
print("Classes:", class_names)

# ---------------- OPTIMIZATION ----------------
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(200).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# ---------------- MODEL ----------------
model = models.Sequential([
    layers.Rescaling(1./255),

    # Block 1
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    # Block 2
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    # Block 3
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    # Block 4 (NEW - deeper learning)
    layers.Conv2D(256, (3,3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    layers.Flatten(),

    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),

    layers.Dense(4, activation='softmax')
])

# ---------------- COMPILE ----------------
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

# ---------------- TRAIN ----------------
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=25,
    callbacks=[early_stop]
)

# ---------------- SAVE MODEL ----------------
model.save("skin_model.h5")
print("Model saved as skin_model.h5")

# ---------------- PLOT ----------------
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.plot(acc, label='Train Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend()
plt.title("Accuracy Graph")
plt.show()
