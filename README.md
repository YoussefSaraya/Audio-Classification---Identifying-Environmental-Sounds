# 🔊 Environmental Sound Classification using Deep Learning

This project aims to classify real-world environmental sounds—such as dog barking, sirens, rainfall, and more—using deep learning models. The system is trained to recognize and categorize audio signals, even under noisy or distorted conditions.

---

## 🎯 Project Objective

To build an audio classification model capable of identifying environmental sounds from short audio clips. This project provides hands-on experience with:

- Audio signal preprocessing
- Feature extraction using Mel spectrograms
- Neural network design (CNN or RNN)
- Model evaluation and prediction

---

## 📦 Dataset

This project uses an environmental sound dataset containing labeled audio samples for:
- Dog bark
- Siren
- Rain
- and other environmental classes

> A commonly used source is [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html) or similar datasets.

---

## ⚙️ Workflow Summary

1. **Preprocessing**
   - Convert audio signals to mono
   - Normalize duration and sampling rate
   - Visualize waveform and spectrogram

2. **Feature Extraction**
   - Extract **MFCCs** (Mel Frequency Cepstral Coefficients)
   - Optionally extract Chroma, Spectral Contrast

3. **Model Architecture**
   - A **Convolutional Neural Network (CNN)** trained on MFCC features
   - Optionally test RNN or hybrid models

4. **Evaluation Metrics**
   - Accuracy
   - Confusion Matrix
   - Precision / Recall / F1-Score

---

## 💻 Technologies Used

- Python
- TensorFlow / Keras
- Librosa
- NumPy / Pandas / Matplotlib / Seaborn
- Scikit-learn

---

## 📂 How to Run

1. Prepare or download the dataset (e.g. UrbanSound8K) and organize the audio files.
2. Open and run the notebook:  
   `Project3_youssefsaraya.ipynb`
3. Follow the steps for:
   - Loading data
   - Feature extraction
   - Model training & evaluation
   - Making predictions on new samples

---

## 🔍 Example Output

After training, the model can classify audio like:

- `dog_bark.wav` → 🐶 Dog Bark  
- `rain_forest.wav` → 🌧️ Rain

---

## 👤 Author

- **Youssef Saraya** – 

---

## 📄 License

This project is licensed under the **MIT License**.

---

## 🙌 Acknowledgements

- [Librosa](https://librosa.org/)
- [UrbanSound8K Dataset](https://urbansounddataset.weebly.com/urbansound8k.html)
- TensorFlow / Keras
