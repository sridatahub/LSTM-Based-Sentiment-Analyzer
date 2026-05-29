# 🎬 IMDB Movie Review Sentiment Analysis using LSTM

A deep learning project that classifies IMDB movie reviews as **positive** or **negative** using a Long Short-Term Memory (LSTM) neural network built with TensorFlow/Keras.

---

## 📌 Project Overview

This project performs binary sentiment classification on 50,000 IMDB movie reviews. It covers the full ML pipeline — from data collection via the Kaggle API, through preprocessing and model training, to a real-time predictive system that can classify any custom review.

---

## 🧠 Model Architecture

```
Input (review text)
    ↓
Embedding Layer  (vocab: 5000, output_dim: 128, input_length: 200)
    ↓
LSTM Layer       (128 units, dropout: 0.2, recurrent_dropout: 0.2)
    ↓
Dense Layer      (1 unit, activation: sigmoid)
    ↓
Output           (0 = Negative, 1 = Positive)
```

| Layer      | Details                                      |
|------------|----------------------------------------------|
| Embedding  | vocab_size=5000, output_dim=128, maxlen=200  |
| LSTM       | 128 units, dropout=0.2                       |
| Dense      | 1 unit, sigmoid activation                   |
| Loss       | Binary Crossentropy                          |
| Optimizer  | Adam                                         |
| Metric     | Accuracy                                     |

---

## 📂 Dataset

- **Source:** [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) — Kaggle
- **Size:** 50,000 reviews (25,000 positive / 25,000 negative)
- **Split:** 80% training / 20% testing

---

## 🔧 Tech Stack

| Tool / Library | Purpose |
|---|---|
| Python 3.x | Core language |
| TensorFlow / Keras | LSTM model building & training |
| Pandas | Data loading and manipulation |
| Scikit-learn | Train-test split |
| Kaggle API | Dataset download |

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/imdb-sentiment-lstm.git
cd imdb-sentiment-lstm
```

### 2. Install dependencies

```bash
pip install tensorflow pandas scikit-learn kaggle
```

### 3. Set up Kaggle API credentials

- Download your `kaggle.json` from [kaggle.com](https://www.kaggle.com) → Account → API
- Place `kaggle.json` in the project root directory

### 4. Run the notebook

Open `DL_Pro_10_IMDB_reviews_Sentiment_Analysis_LSTM.ipynb` in Jupyter or Google Colab and run all cells.

---

## 🔄 Workflow

```
1. Install Kaggle & download dataset
        ↓
2. Load & explore the CSV (50K reviews)
        ↓
3. Encode labels: positive → 1, negative → 0
        ↓
4. Train/test split (80/20)
        ↓
5. Tokenize text (top 5000 words)
   + Pad sequences to length 200
        ↓
6. Build LSTM model
        ↓
7. Train (5 epochs, batch_size=64, val_split=0.2)
        ↓
8. Evaluate on test set
        ↓
9. Predict on custom reviews
```

---

## 📊 Model Training

```python
model.fit(X_train, Y_train,
          epochs=5,
          batch_size=64,
          validation_split=0.2)
```

---

## 🔍 Predictive System

Run sentiment prediction on any custom review:

```python
def predict_sentiment(review):
    sequence = tokenizer.texts_to_sequences([review])
    padded_sequence = pad_sequences(sequence, maxlen=200)
    prediction = model.predict(padded_sequence)
    sentiment = "positive" if prediction[0][0] > 0.5 else "negative"
    return sentiment

# Example
predict_sentiment("This movie was fantastic. I loved it.")
# Output: positive

predict_sentiment("This movie was not that good.")
# Output: negative
```

---

## 📁 Project Structure

```
imdb-sentiment-lstm/
│
├── DL_Pro_10_IMDB_reviews_Sentiment_Analysis_LSTM.ipynb  # Main notebook
├── kaggle.json                                            # Kaggle API key (not committed)
├── IMDB Dataset.csv                                       # Downloaded dataset
└── README.md                                             # Project documentation
```

> ⚠️ Never commit `kaggle.json` to GitHub. Add it to `.gitignore`.

---

## 📝 .gitignore (recommended)

```
kaggle.json
*.zip
IMDB Dataset.csv
__pycache__/
.ipynb_checkpoints/
```

---

## 🙋 Author

**Your Name**
- LinkedIn: [linkedin.com/in/srikanth-data](www.linkedin.com/in/srikanth-data)
- Gmail:chavalasrikanth09@gmail.com

---

## 📄 License

This project is open-source under the [MIT License](LICENSE).
