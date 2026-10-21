#%pip uninstall -y pyarrow
#%pip install --no-cache-dir pyarrow


import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from PIL import Image
from tensorflow.keras import layers as L, Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import (
    preprocess_input, decode_predictions
)

# 1. Load images from dataset folder
folder = r"C:\Users\Risho\Downloads\images"
raw, names = [], []

for file in os.listdir(folder):
    if file.lower().endswith((".jpg", ".jpeg", ".png")):
        img = Image.open(os.path.join(folder, file)).convert("RGB")
        raw.append(np.array(img.resize((224, 224))))
        names.append(file)

if len(raw) < 10:
    raise ValueError("Add at least 10 images to the dataset folder.")

raw = np.array(raw)

# 2. Get ImageNet class descriptions using ResNet50
cnn = ResNet50(weights="imagenet")
pred = cnn.predict(preprocess_input(raw.copy()), batch_size=8)

labels = [
    decode_predictions(p[None, :], top=1)[0][0][1].replace("_", " ")
    for p in pred
]

captions = ["a photo of " + label for label in labels]

for i in range(min(5, len(names))):
    print(names[i], "->", captions[i])

# 3. Prepare GAN images: 32x32, range [-1, 1]
images = tf.image.resize(raw, (32, 32)) / 127.5 - 1.0

# 4. Text encoder: text -> word embeddings -> 64-D vector
prompts = captions + [
    "a photo of a tiger",
    "a photo of a dog",
    "a close up photo of a tiger"
]

vec = L.TextVectorization(
    max_tokens=500,
    output_sequence_length=10
)
vec.adapt(tf.data.Dataset.from_tensor_slices(prompts))

ti = L.Input(shape=(), dtype=tf.string)
tokens = L.Embedding(
    len(vec.get_vocabulary()), 32, mask_zero=True
)(vec(ti))
encoded = L.GlobalAveragePooling1D()(tokens)
encoded = L.Dense(64, activation="relu")(encoded)
enc = Model(ti, encoded)

# 5. Conditional Generator: noise + text -> image
z = L.Input(shape=(100,))
e = L.Input(shape=(64,))

x = L.Concatenate()([z, e])
x = L.Dense(8 * 8 * 128, activation="relu")(x)
x = L.Reshape((8, 8, 128))(x)

for f in (64, 32):
    x = L.UpSampling2D()(x)
    x = L.Conv2D(f, 3, padding="same")(x)
    x = L.BatchNormalization()(x)
    x = L.LeakyReLU(0.2)(x)

out = L.Conv2D(3, 3, padding="same", activation="tanh")(x)
G = Model([z, e], out)

# 6. Conditional Discriminator: image + text -> real/fake
im = L.Input(shape=(32, 32, 3))
e2 = L.Input(shape=(64,))

x = L.Conv2D(32, 4, strides=2, padding="same")(im)
x = L.LeakyReLU(0.2)(x)
x = L.Conv2D(64, 4, strides=2, padding="same")(x)
x = L.BatchNormalization()(x)
x = L.LeakyReLU(0.2)(x)
x = L.Flatten()(x)

e2_dense = L.Dense(64, activation="relu")(e2)
x = L.Concatenate()([x, e2_dense])
x = L.Dense(128, activation="relu")(x)
out = L.Dense(1, activation="sigmoid")(x)

D = Model([im, e2], out)

# 7. Loss and optimizers
bce = tf.keras.losses.BinaryCrossentropy()

g_opt = tf.keras.optimizers.Adam(0.0002, beta_1=0.5)
d_opt = tf.keras.optimizers.Adam(0.0002, beta_1=0.5)

# 8. Prepare training data
ds = tf.data.Dataset.from_tensor_slices(
    (images, np.array(captions))
)
ds = ds.shuffle(len(images)).batch(16).prefetch(tf.data.AUTOTUNE)

# 9. Adversarial training
@tf.function
def train_step(real, text):
    batch = tf.shape(real)[0]

    with tf.GradientTape(persistent=True) as tape:
        emb = enc(text, training=True)

        noise = tf.random.normal((batch, 100))
        fake = G([noise, emb], training=True)

        real_out = D([real, emb], training=True)
        fake_out = D([fake, emb], training=True)

        d_loss = (
            bce(tf.ones_like(real_out), real_out)
            + bce(tf.zeros_like(fake_out), fake_out)
        )
        g_loss = bce(tf.ones_like(fake_out), fake_out)

    gv = G.trainable_variables + enc.trainable_variables
    dv = D.trainable_variables

    g_opt.apply_gradients(
        zip(tape.gradient(g_loss, gv), gv)
    )
    d_opt.apply_gradients(
        zip(tape.gradient(d_loss, dv), dv)
    )

    return g_loss, d_loss

# 10. Generate an image from text
def generate(text):
    emb = enc(tf.constant([text]), training=False)
    noise = tf.random.normal((1, 100))
    out = G([noise, emb], training=False)[0]
    return tf.clip_by_value((out + 1) / 2, 0, 1)

# 11. Train the GAN
EPOCHS = 20

for ep in range(EPOCHS):
    gl, dl = [], []

    for real, text in ds:
        g_loss, d_loss = train_step(real, text)
        gl.append(float(g_loss))
        dl.append(float(d_loss))

    print(
        f"Epoch {ep+1}/{EPOCHS} | "
        f"G Loss: {np.mean(gl):.4f} | "
        f"D Loss: {np.mean(dl):.4f}"
    )

    if (ep + 1) % 5 == 0:
        plt.imshow(generate(captions[0]))
        plt.title(f"Generated: {captions[0]}")
        plt.axis("off")
        plt.show()

# 12. Generate image for an unseen text description
unseen_text = "a photo of a tiger"
generated = generate(unseen_text)

plt.figure(figsize=(5, 5))
plt.imshow(generated)
plt.title("Text: " + unseen_text)
plt.axis("off")
plt.show()

print("Text:", unseen_text)
print("Generated image shape:", generated.shape)

# 13. Compare a real image and a generated image
plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(tf.clip_by_value((images[0] + 1) / 2, 0, 1))
plt.title("Real image: " + captions[0])
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(generated)
plt.title("Generated: " + unseen_text)
plt.axis("off")

plt.tight_layout()
plt.show()
