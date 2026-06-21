# 📘 Text Generation using Vanilla RNN, LSTM, and GRU

## 📌 Project Overview

This project demonstrates text generation using three Deep Learning sequence models:

- Vanilla RNN
- LSTM (Long Short-Term Memory)
- GRU (Gated Recurrent Unit)

The models are trained on a custom text corpus to learn grammar, sentence structure, and contextual relationships. After training, each model generates new text by predicting the next word from a given seed sentence.

The project also compares the performance of the three models based on training loss, text quality, memory handling, and long-term dependency learning.

---

## 🎯 Problem Statement

Design and implement a Deep Learning model capable of learning the underlying structure, grammar, and contextual dependencies of a given text corpus to generate coherent and meaningful text sequences using:

- Vanilla RNN
- LSTM
- GRU

The models are compared based on:

- Training Loss
- Generated Text Quality
- Memory Handling
- Long-Term Dependency Learning

---

## 🎓 Student Learning Tasks Completed

- ✅ Replaced the default corpus with a custom paragraph
- ✅ Increased embedding dimension from **32 → 100**
- ✅ Increased hidden units from **64 → 128**
- ✅ Increased training epochs from **100 → 200**
- ✅ Generated **10 words** instead of 5 during text generation

---

## 📂 Project Workflow

1. Import required libraries
2. Load custom text corpus
3. Tokenize the text
4. Create n-gram sequences
5. Pad input sequences
6. Build Vanilla RNN model
7. Build LSTM model
8. Build GRU model
9. Train all three models
10. Compare training loss
11. Generate text samples
12. Analyze model performance

---

## 🛠 Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib

---

## 🧠 Deep Learning Models

### 1. Vanilla RNN
- Baseline recurrent neural network
- Learns short-term sequential patterns
- Struggles with long-term dependencies due to the vanishing gradient problem

### 2. LSTM
- Uses memory cells and gating mechanisms
- Captures long-term contextual information
- Produces more coherent text generation

### 3. GRU
- Uses update and reset gates
- Simpler architecture than LSTM
- Faster training with comparable performance

---

## 📊 Model Comparison

| Feature | Vanilla RNN | LSTM | GRU |
|----------|-------------|------|-----|
| Memory Handling | Low | Excellent | Very Good |
| Long-Term Dependencies | Poor | Excellent | Very Good |
| Training Speed | Fast | Moderate | Fast |
| Text Quality | Good | Best | Very Good |

---

## 📈 Results

The notebook compares:

- Training Loss Curves
- Generated Text Samples
- Model Performance
- Learning Behavior

---

## 📌 Conclusion

- Vanilla RNN learns short patterns but struggles with long-term memory.
- LSTM captures long-range grammar and contextual dependencies effectively.
- GRU achieves performance comparable to LSTM while training faster due to its simpler architecture.
- This project provides both theoretical understanding and practical implementation of sequence modeling for text generation.

---

## 📁 Repository Structure

```
Text-Generation-RNN-LSTM-GRU/
│
├── Text_Generation_RNN_LSTM_GRU.ipynb
├── README.md
└── requirements.txt
```

---

## 🚀 Future Improvements

- Train on larger datasets such as Shakespeare or Wikipedia
- Experiment with word-level and character-level tokenization
- Use Bidirectional LSTM
- Apply Attention Mechanism
- Implement Transformer-based text generation models

---

## 👩‍💻 Author

**Ankita Yadav**
B.Tech Computer Science Engineering  
Jaipur Engineering College & Research Centre (JECRC)

