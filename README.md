# 🎬 Movie Review Sentiment Analysis using PyTorch & Streamlit

## 📌 Project Overview
This project is an **end-to-end Sentiment Analysis system** built using **PyTorch** for deep learning and **Streamlit** for deployment.  
The goal of the project is to automatically classify **IMDb movie reviews** as **Positive** or **Negative** using a **Long Short-Term Memory (LSTM)** neural network.

The project covers the **complete machine learning lifecycle**:
- Data preprocessing
- Tokenization and sequence padding
- Model building and training
- Evaluation and performance analysis
- Saving and loading the ML pipeline
- Deploying the model as an interactive web application
- Version controlling and publishing on GitHub

This makes the project **interview-ready, portfolio-ready, and production-aware**.

---

## 🎯 Problem Statement
Movie reviews are written in natural language and often contain subjective opinions.  
The task is to design a system that can **understand the sentiment** expressed in a review and classify it as either:
- **Positive**
- **Negative**

This problem belongs to the domain of **Natural Language Processing (NLP)** and **binary text classification**.

---

## 📂 Dataset Description
- **Dataset**: IMDb Movie Review Dataset
- **Total samples**: 50,000 reviews
- **Labels**: Positive / Negative
- **Distribution**: Balanced (25,000 positive, 25,000 negative)

Each review is a free-form text, which requires proper preprocessing and numerical encoding before being fed into a neural network.

---

## 🔧 Data Preprocessing
The following preprocessing steps were applied:

1. Removal of HTML tags from reviews  
2. Conversion of text to lowercase  
3. Removal of special characters and punctuation  
4. Tokenization of text into words  
5. Conversion of words into numerical indices  

These steps ensure that the input data is clean, consistent, and suitable for deep learning models.

---

## 🔢 Tokenization & Padding
A **custom tokenizer** was implemented to:
- Build a vocabulary from training data only
- Assign unique indices to words
- Handle unseen words using a special `<UNK>` token

Since neural networks require fixed-size inputs:
- All review sequences are **padded or truncated**
- A maximum sequence length of **200 tokens** is used

This allows efficient batch processing using PyTorch.

---

## 🧠 Model Architecture
The sentiment classifier is built using the following layers:

- **Embedding Layer**  
  Converts word indices into dense vector representations.
- **LSTM Layer**  
  Captures long-term dependencies in text sequences.
- **Dropout Layer**  
  Prevents overfitting by randomly disabling neurons during training.
- **Fully Connected Layer**  
  Maps LSTM output to a single neuron.
- **Sigmoid Activation**  
  Produces probability values between 0 and 1.

The architecture is designed to balance **performance and simplicity**.

---

## ⚙️ Training Process
- **Loss Function**: Binary Cross-Entropy Loss (BCELoss)
- **Optimizer**: Adam Optimizer
- **Batch Size**: 32
- **Epochs**: 5
- **Learning Rate**: 0.001

During training:
1. Forward propagation is performed
2. Loss is calculated
3. Gradients are computed using backpropagation
4. Model weights are updated

Training loss steadily decreases across epochs, indicating effective learning.

---

## 📊 Model Evaluation
The dataset is split into:
- **80% Training data**
- **20% Testing data**

The model achieves:
- **Test Accuracy: ~86–87%**

This indicates strong generalization on unseen data.

---

## 💾 Saving the ML Pipeline
To enable reuse without retraining:
- Model weights are saved (`sentiment_model.pth`)
- Tokenizer vocabulary is saved (`tokenizer_vocab.pth`)
- Configuration parameters are preserved

This ensures consistent predictions during deployment.

---

## 🖥️ Streamlit Deployment
The trained model is deployed using **Streamlit**, providing:

- Real-time sentiment prediction
- User-friendly text input
- Confidence score visualization
- Interactive UI elements
- Clean and professional layout

The application allows users to input any movie review and instantly view the predicted sentiment.

---

## 🎨 UI & User Experience
The Streamlit app includes:
- Clean alignment and layout
- Interactive buttons
- Confidence progress bars
- Clear visual feedback for predictions

The UI is designed to be **simple, intuitive, and recruiter-friendly**.

---

## 🛠️ Tools & Technologies Used
- **Python**
- **PyTorch**
- **Pandas**
- **Scikit-learn**
- **Streamlit**
- **NumPy**
- **Git & GitHub**
- **VS Code**
- **Google Colab**

---

## 🚀 How to Run the Project Locally

```bash
git clone https://github.com/priya19-tech/sentiment-analysis-pytorch.git
cd sentiment-analysis-pytorch
pip install -r requirements.txt
cd src
streamlit run app.py
