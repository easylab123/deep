
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, RepeatVector, Add
from tensorflow.keras.models import Model

# 1. Load pretrained CNN
cnn = ResNet50(weights="imagenet", include_top=False, pooling="avg")

# 2. Dataset folder
folder = r"C:\Users\Risho\Downloads\images"
images, names = [], []

for file in os.listdir(folder):
    if file.lower().endswith((".jpg", ".jpeg", ".png")):
        img = Image.open(os.path.join(folder, file))
        img = img.convert("RGB").resize((224, 224))
        images.append(np.array(img))
        names.append(file)

if len(images) < 2:
    raise ValueError("Please add at least 2 images.")

images = np.array(images)

# 3. Get ImageNet class labels
classifier = ResNet50(weights="imagenet")
pred = classifier.predict(preprocess_input(images.copy()), batch_size=16)

labels = [
    decode_predictions(p[None, :], top=1)[0][0][1].replace("_", " ")
    for p in pred
]

captions = ["a photo of " + label for label in labels]
captions = ["<start> " + c + " <end>" for c in captions]

# 4. Extract CNN features
features = cnn.predict(preprocess_input(images.copy()), batch_size=16)
print("Feature vector:", features.shape)

# 5. Tokenize captions
tokenizer = Tokenizer(filters="", lower=True, oov_token="<unk>")
tokenizer.fit_on_texts(captions)

seq = tokenizer.texts_to_sequences(captions)
max_len = max(len(s) for s in seq)
seq = pad_sequences(seq, maxlen=max_len, padding="post")

vocab = len(tokenizer.word_index) + 1
decoder_in = seq[:, :-1]
decoder_out = seq[:, 1:]

# 6. Build CNN-LSTM model
image_input = Input(shape=(2048,))
text_input = Input(shape=(max_len - 1,))

image_feature = Dense(128, activation="relu")(image_input)
image_feature = RepeatVector(max_len - 1)(image_feature)

text_feature = Embedding(vocab, 128, mask_zero=True)(text_input)

x = Add()([image_feature, text_feature])
x = LSTM(128, return_sequences=True)(x)
output = Dense(vocab, activation="softmax")(x)

model = Model([image_input, text_input], output)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

# 7. Train LSTM decoder
model.fit(
    [features, decoder_in],
    decoder_out[..., np.newaxis],
    epochs=100,
    batch_size=8,
    verbose=1
)

# 8. Generate caption
def generate_caption(feature):
    sentence = ["<start>"]

    for _ in range(max_len - 1):
        ids = tokenizer.texts_to_sequences([" ".join(sentence)])[0]
        ids = pad_sequences([ids], maxlen=max_len - 1, padding="post")

        prediction = model.predict([feature, ids], verbose=0)
        pos = len(sentence) - 1
        word_id = np.argmax(prediction[0, pos])
        word = tokenizer.index_word.get(word_id, "<unk>")

        if word in ("<end>", "<unk>"):
            break

        sentence.append(word)

    return " ".join(sentence[1:]).capitalize() + "."

# 9. Generate and display captions
for i in range(len(images)):
    caption = generate_caption(features[i:i+1])

    print("\nImage:", names[i])
    print("ImageNet label:", labels[i])
    print("Generated caption:", caption)

    plt.figure(figsize=(5, 5))
    plt.imshow(images[i])
    plt.title(caption)
    plt.axis("off")
    plt.show()
