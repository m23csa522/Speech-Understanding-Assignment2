
# 📚 Speech Understanding Assignment - README

## 🧩 Overview

This project addresses **two main tasks**:

- **Q1: Speech Enhancement & Speaker Identification**
- **Q2: MFCC Feature Extraction & Indian Language Classification**

---

## 📌 Question 1: Speech Enhancement & Speaker Identification

### 🎯 Objective

1. **Download & Preprocess VoxCeleb Datasets**
   - VoxCeleb1 (evaluation)
   - VoxCeleb2 (fine-tuning + mixture generation)

2. **Speaker Identification**
   - Use pretrained `WavLM-Base+`
   - Fine-tune using **LoRA** and **ArcFace Loss**
   - Evaluate using: EER, TAR@1% FAR, Rank-1 Accuracy

3. **Speaker Mixture Dataset Creation**
   - Use 50+50 identities from VoxCeleb2
   - Mix audio pairs to form multi-speaker scenarios

4. **Speaker Separation + Enhancement**
   - Use pre-trained `SepFormer`
   - Fine-tune `SepFormer` jointly with the SID model to improve **speaker-aware enhancement**
   - Evaluate using:
     - SI-SNR
     - SDR (Signal-to-Distortion Ratio)
     - SIR (Signal-to-Interference Ratio)
     - SAR (Signal-to-Artifacts Ratio)
     - PESQ (Perceptual Quality)
     - Rank-1 Accuracy (SID)

---

### ✅ Steps Taken

| Step | Task |
|------|------|
| 1. | Downloaded VoxCeleb1 and VoxCeleb2 |
| 2. | Preprocessed and extracted speaker embeddings |
| 3. | Fine-tuned `WavLM` using LoRA + ArcFace |
| 4. | Generated multi-speaker mixtures |
| 5. | Combined `SepFormer` + `WavLM` into a single pipeline |
| 6. | Computed SI-SNR loss + SID cosine loss |
| 7. | Evaluated enhanced outputs using SDR, SIR, SAR, and Rank-1 accuracy |
| 8. | Added metrics during training to monitor performance |

---

### 🧪 Final Metrics

| Metric              | Observation       |
|---------------------|------------------|
| **SDR**             | Improved in most samples |
| **SIR**             | High separation in cleaner samples |
| **SAR**             | Acceptable but depends on noise level |
| **Rank-1 Accuracy** | Reached up to **100%** in many cases |
| **Loss Trend**      | SID + Sep Loss converging |

---

## 📌 Question 2: MFCC Analysis & Language Classification

### 🎯 Objective

- Extract MFCC features for 10 Indian languages
- Visualize MFCC spectrograms
- Classify languages using traditional ML

---

### ✅ Steps Taken

| Step | Task |
|------|------|
| 1. | Extracted MFCCs using `librosa` |
| 2. | Visualized spectrograms of 3 languages: Hindi, Tamil, Bengali |
| 3. | Performed statistical analysis (mean, variance) |
| 4. | Trained Random Forest Classifier |
| 5. | Evaluated using Accuracy & Confusion Matrix |

---

### 📊 Results

| Metric             | Result             |
|--------------------|--------------------|
| **Classification Accuracy** | ~91.5% |
| **Confusion Matrix** | High diagonal dominance, low confusion between distinct languages |

---

## 📦 Libraries Used

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install speechbrain mir_eval pesq librosa matplotlib scikit-learn tqdm
```

Additional (if needed):

```bash
pip install asteroid peft
```

---

## 🔬 Observations

- Combining SepFormer and SID models allows for **speaker-aware enhancement**, improving both **perceived quality** and **identifiability**.
- Fine-tuning `WavLM` increases identification performance dramatically, even after enhancement.
- MFCCs capture distinct **acoustic features** across languages, enough for high-accuracy classification.
- Background noise and overlap are key challenges in both SID and language ID.

---

## 📁 Files Overview

| File | Description |
|------|-------------|
| `speech_understanding_assignment_2_ii.py` | Fine-tuning `WavLM` with ArcFace & LoRA |
| `speech_understanding_assignment_2_iii.py` | SepFormer + SID integration & training |
| `speech_understanding_assignment_2_iv.py` | Evaluation metrics: SDR, SIR, SAR, Rank-1 Acc |
| `Ques2.py` | MFCC Extraction, Visualization & Classification |

---
