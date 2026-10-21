import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.utils import get_file
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from PIL import Image
import os

# 1. Load pretrained CNN without classifier
model = ResNet50(weights="imagenet", include_top=False, pooling="avg")

# 2. Images and labels
urls = [r"C:\Users\Risho\Downloads\n01484850_great_white_shark.JPEG",r"C:\Users\Risho\Downloads\n01491361_tiger_shark.JPEG",r"C:\Users\Risho\Downloads\n01484850_great_white_shark.JPEG",r"C:\Users\Risho\Downloads\n01491361_tiger_shark.JPEG"]

labels = [0, 1, 0, 1]
images = []

# 3. Feature extraction
for url in urls:
    #file = get_file(os.path.basename(url), url)
    img = Image.open(url).resize((224, 224))
    x=np.array(img)
    x = preprocess_input(x)
    images.append(x)

X = np.array(images)
X = model.predict(X)

print("Feature vector:", X.shape)

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, labels,
    test_size=0.5,
    random_state=42,
    stratify=labels
)

print("Train classes:", np.unique(y_train))
print("Test classes:", np.unique(y_test))

# 5. Classification layer
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# 6. Prediction
y_pred = clf.predict(X_test)

# 7. Metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred, zero_division=0))
print("F1-score:", f1_score(y_test, y_pred, zero_division=0))

# 8. Confusion matrix
plt.imshow(
    confusion_matrix(y_test, y_pred)
)

plt.show()