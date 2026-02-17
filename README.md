## two-stage-semiconductor-defect-detection-edgeguardAI_phase2
## Program DataPath - https://www.kaggle.com/code/surjini/hackathon-test-dataset-prediction
# code
```
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
TEST_PATH = "/kaggle/input/datasets/surjini/hackathon-test-dataset/hackathon_test_dataset"

stage1_model_path = "/kaggle/input/models/surjini/two-stage-semiconductor-defect-detection-model/keras/default/1/stage1_best_model.keras"
stage2_model_path = "/kaggle/input/models/surjini/two-stage-semiconductor-defect-detection-model/keras/default/1/stage2_best_model.keras"

# Load Models
stage1_model = load_model(stage1_model_path, compile=False)
stage2_model = load_model(stage2_model_path, compile=False)

IMG_SIZE = (224, 224)

# Stage 2 Class Order
model_classes = ['bridges', 'clean', 'cracks', 'opens', 'other', 'scratches', 'shorts', 'vias']

# Dataset label mapping
label_mapping = {
    "bridge": "bridges",
    "crack": "cracks",
    "open": "opens",
    "via": "vias",
    "cmp": "scratches",
    "ler": "other",
    "particle": "other",
    "other": "other",
    "clean": "clean"
}

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

true_labels = []
predicted_labels = []

for folder in os.listdir(TEST_PATH):
    folder_path = os.path.join(TEST_PATH, folder)

    if os.path.isdir(folder_path):

        for img_name in os.listdir(folder_path):

            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            img_path = os.path.join(folder_path, img_name)
            img = preprocess_image(img_path)

            # Stage 1 Prediction
            stage1_pred = stage1_model.predict(img, verbose=0)[0][0]

            if stage1_pred < 0.45:
                prediction = "clean"
            else:
                stage2_pred = stage2_model.predict(img, verbose=0)
                confidence = np.max(stage2_pred)
                defect_index = np.argmax(stage2_pred)

                if confidence < 0.4:
                    prediction = "other"
                else:
                    prediction = model_classes[defect_index]

            true_label = label_mapping.get(folder.lower(), "other")

            true_labels.append(true_label)
            predicted_labels.append(prediction)

print("Total Samples:", len(true_labels))

# Metrics
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels, average='weighted', zero_division=0)
recall = recall_score(true_labels, predicted_labels, average='weighted', zero_division=0)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)

# Sample Predictions
print("\nSample Predictions:")
for i in range(min(10, len(true_labels))):
    print("True:", true_labels[i], "| Pred:", predicted_labels[i])

# Confusion Matrix
labels = model_classes
cm = confusion_matrix(true_labels, predicted_labels, labels=labels)

plt.figure(figsize=(10, 8))
sns.heatmap(cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - Hackathon Test Dataset")

plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()
with open("prediction_log.txt", "w") as f:
    f.write(f"Total Samples: {len(true_labels)}\n")
    f.write(f"Accuracy: {accuracy}\n")
    f.write(f"Precision: {precision}\n")
    f.write(f"Recall: {recall}\n")
```
