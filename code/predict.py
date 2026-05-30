import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# ---------------- LOAD MODEL ----------------
model = tf.keras.models.load_model("skin_model.h5")

class_names = ['atopic_dermatitis', 'eczema', 'melanoma', 'ringworm']

# ---------------- LOAD IMAGE ----------------
img_path = "test4.webp"

img = image.load_img(img_path, target_size=(180,180))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

# ---------------- PREDICT ----------------
pred = model.predict(img_array)
score = pred[0]   # already softmax

predicted_class = class_names[np.argmax(score)]
confidence = np.max(score) * 100

# ---------------- DISPLAY ----------------
plt.imshow(img)
plt.title(f"{predicted_class} ({confidence:.2f}%)")
plt.axis('off')
plt.show()

print("Predicted Disease:", predicted_class)
print("Confidence:", confidence, "%")
