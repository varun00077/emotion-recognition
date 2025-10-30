# 🎭 Emotion Detection from Speech using CNN-LSTM  

This project detects **human emotions** from speech audio using a **hybrid Convolutional Neural Network (CNN)** and **Bidirectional LSTM** model.  
It is trained on the **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)** dataset to classify emotions such as  
😐 Neutral · 😌 Calm · 😀 Happy · 😢 Sad · 😡 Angry · 😱 Fearful · 🤢 Disgust · 😲 Surprised  

---

## 📘 Overview  

The system extracts deep audio features (MFCC, Chroma, Spectral Contrast, Tonnetz) using **Librosa**,  
then feeds them into a CNN-LSTM architecture for classification.  
CNN captures spatial features, while LSTM models temporal patterns in emotion transitions.  

---

## 🧠 Features  

✅ End-to-end emotion detection from `.wav` audio  
✅ Multi-feature extraction with Librosa  
✅ CNN + BiLSTM architecture  
✅ Class balancing via computed weights  
✅ Visualization using Seaborn and Matplotlib  
✅ Model saving & evaluation support  

---

## 🎧 Dataset – RAVDESS  

**Source:** [RAVDESS Emotional Speech Dataset](https://zenodo.org/record/1188976)  

Each audio file name encodes metadata like emotion, intensity, statement, repetition, and actor ID  
(e.g., `03-01-04-01-02-01-09.wav`).

| Code | Emotion    |
|:----:|:-----------|
| 01 | Neutral |
| 02 | Calm |
| 03 | Happy |
| 04 | Sad |
| 05 | Angry |
| 06 | Fearful |
| 07 | Disgust |
| 08 | Surprised |

---

## ⚙️ Installation  

```bash
pip install librosa soundfile tensorflow scikit-learn matplotlib seaborn tqdm
````

Set up dataset path in your notebook or script:

```python
DATASET_DIR = "/content/drive/MyDrive/audio lstm cnn"
```

---

## 🧩 Model Architecture

```
Input  
 ├── Conv2D → BatchNorm → MaxPool → Dropout  
 ├── Conv2D → BatchNorm → MaxPool → Dropout  
 ├── Conv2D → BatchNorm → MaxPool → Dropout  
 ├── Reshape → BiLSTM → Dense(128) → Dropout  
 └── Dense(8) [Softmax Output]
```

**Hyperparameters**

* Optimizer: Adam
* Loss: Categorical Crossentropy
* Batch Size: 32
* Epochs: 40
* Metrics: Accuracy

---

## 📊 Results

| Metric            | Value       |
| :---------------- | :---------- |
| **Test Accuracy** | **37.08 %** |
| **Train Samples** | 1200        |
| **Test Samples**  | 240         |

**Classification Report:**

```
              precision    recall  f1-score   support
angry             0.40      0.59      0.47        32
calm              0.60      0.56      0.58        32
disgust           0.28      0.69      0.40        32
fearful           0.55      0.19      0.28        32
happy             0.31      0.12      0.18        32
neutral           0.12      0.12      0.12        16
sad               0.38      0.16      0.22        32
surprised         0.42      0.41      0.41        32
```

**Visualization:**

```python
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=le.classes_,
            yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()
```

---

## 💾 Saved Artifacts

| File                    | Description                                   |
| :---------------------- | :-------------------------------------------- |
| `ravdess_best_model.h5` | Best checkpoint (highest validation accuracy) |
| `ravdess_crnn_model.h5` | Final trained model                           |
| `label_encoder.pkl`     | Label encoder for decoding predictions        |

---

## 🚀 Future Scope

🔹 Add attention or transformer-based layers
🔹 Use SpecAugment for audio data augmentation
🔹 Deploy model via FastAPI or Streamlit for real-time inference
🔹 Train on multilingual emotion datasets

---

## 👨‍💻 Author

**Varun Sharma**
🎓 IIIT Bhagalpur
💡 Focused on **Generative AI** and **Deep Learning** applications.

---

## 🏁 Acknowledgements

* [RAVDESS Dataset](https://zenodo.org/record/1188976)
* [Librosa](https://librosa.org/) for audio feature extraction
* [TensorFlow/Keras](https://www.tensorflow.org/) for model implementation

---

⭐ **If you found this project useful, consider giving it a star on GitHub!**

```


