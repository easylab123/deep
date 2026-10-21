
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from PIL import Image
import os

# 1. Load pretrained VGG16
model = VGG16(weights="imagenet")

# 2. Dataset folder
folder = r"C:\Users\Risho\OneDrive\Pictures\Screenshots 1"
images = []

# 3. Load and preprocess all images
for file in os.listdir(folder):
    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
        img = Image.open(os.path.join(folder, file)).convert("RGB").resize((224, 224))
        x = np.array(img)
        x = preprocess_input(x)
        images.append(x)

x = np.array(images)

# 4. Display first image
plt.imshow(x[0])
plt.show()

# 5. Predict
#pred = model.predict(x, batch_size=16) for large
pred = model.predict(x)

# 6. Top-5 predictions
for i in range(len(pred)):
    print("\nImage", i+1)
    for label, name, prob in decode_predictions(pred[i:i+1], top=5)[0]:
        print(name, round(prob*100, 2), "%")

# 7. Intermediate representation
layer_model = tf.keras.Model(model.input, model.get_layer("block3_conv3").output)
features = layer_model.predict(x)

print("\nInput shape:", x.shape)
print("Intermediate representation:", features.shape)
print("Output shape:", pred.shape)

# 8. Display feature map of first image
plt.imshow(features[0, :, :, 0])
plt.show()

# 9. Display filter
filters = model.get_layer("block3_conv3").get_weights()[0]
plt.imshow(filters[:, :, 0, 1])
plt.show()


#block3_conv1, block3_conv2, block3_conv3..vgg16#
#"conv1_conv", "conv2_block3_out", "conv4_block6_out"....resnet
