# =============================
# Task A: MFCC Feature Extraction and Visualization
# =============================

import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Set your dataset path here
dataset_path = r"C:\Users\affine\Desktop\Speech_Understanding_Minor\Q1\WAV"  # <-- Change this if needed

# Choose 3 languages for visualization
selected_languages = ["Hindi", "Tamil", "Bengali"]
samples_to_plot = 5

# Function to extract MFCC from an audio file
def extract_mfcc(filepath, n_mfcc=13):
    audio, sr = librosa.load(filepath, sr=16000)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return mfcc, sr

# Function to plot MFCC spectrogram
def show_mfcc(mfcc, sr, title):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc, sr=sr, x_axis='time')
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.show()

# Extract and visualize MFCCs
mfcc_data = {}
for lang in selected_languages:
    lang_folder = os.path.join(dataset_path, lang)
    files = [f for f in os.listdir(lang_folder) if f.endswith('.mp3') or f.endswith('.wav')][:samples_to_plot]
    
    mfcc_data[lang] = []
    for i, file in enumerate(files):
        file_path = os.path.join(lang_folder, file)
        mfcc, sr = extract_mfcc(file_path)
        mfcc_data[lang].append(mfcc)
        show_mfcc(mfcc, sr, f"{lang} Sample {i+1}")

# Optional: Statistical analysis
def compute_mfcc_statistics(mfcc_list):
    combined = np.concatenate([m.T for m in mfcc_list], axis=0)
    return np.mean(combined, axis=0), np.var(combined, axis=0)

for lang in selected_languages:
    mean_mfcc, var_mfcc = compute_mfcc_statistics(mfcc_data[lang])
    print(f"\n{lang} Mean MFCC: {mean_mfcc}")
    print(f"{lang} Variance MFCC: {var_mfcc}")

# =============================
# Task B: Classification
# =============================

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Full language list for classification
all_languages = sorted(os.listdir(dataset_path))
language_to_index = {lang: idx for idx, lang in enumerate(all_languages)}

X, y = [], []

for lang in all_languages:
    lang_folder = os.path.join(dataset_path, lang)
    files = [f for f in os.listdir(lang_folder) if f.endswith('.mp3') or f.endswith('.wav')]
    
    for file in tqdm(files, desc=f"Processing {lang}"):
        try:
            file_path = os.path.join(lang_folder, file)
            mfcc, _ = extract_mfcc(file_path)
            mfcc_mean = np.mean(mfcc, axis=1)
            X.append(mfcc_mean)
            y.append(language_to_index[lang])
        except Exception as e:
            print(f"Failed for {file_path}: {e}")

X = np.array(X)
y = np.array(y)

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Train classifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)

# Evaluate
y_pred = classifier.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nClassification Accuracy: {acc * 100:.2f}%")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=all_languages)
disp.plot(xticks_rotation=45, cmap=plt.cm.Blues)
plt.title("Language Classification Confusion Matrix")
plt.tight_layout()
plt.show()
