# 🎬 IMDB Movie Review Sentiment Analysis

A deep learning-powered web application that analyzes movie review sentiment using a Recurrent Neural Network (RNN) trained on the IMDB dataset.

## 🌐 Live Demo

**[🚀 Check-out the application here (Deployed)](https://imdb-movie-review-sentiment-analysis-using-deep-learning-rnn-i.streamlit.app/)**


---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [API Reference](#api-reference)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

This project demonstrates an end-to-end deep learning solution for sentiment analysis of movie reviews. The application uses a Simple RNN (Recurrent Neural Network) trained on the IMDB dataset to classify movie reviews as positive or negative with high accuracy.

### Key Highlights

- **Deep Learning Model**: Simple RNN with 1.3M parameters
- **High Accuracy**: ~80% validation accuracy
- **Real-time Analysis**: Instant sentiment prediction
- **Modern UI**: Beautiful Streamlit interface with interactive visualizations
- **Production Ready**: Robust error handling and model compatibility

---

## ✨ Features

### 🎨 User Interface
- **Modern Design**: Gradient backgrounds and professional styling
- **Interactive Visualizations**: Gauge charts and confidence metrics
- **Responsive Layout**: Works on desktop and mobile devices
- **Real-time Feedback**: Loading spinners and progress indicators

### 🤖 AI Capabilities
- **Instant Analysis**: Get sentiment results in seconds
- **Confidence Scoring**: Detailed confidence metrics for predictions
- **Insightful Explanations**: Helpful analysis insights
- **Robust Error Handling**: Graceful handling of edge cases

### 📊 Analytics
- **Sentiment Classification**: Positive/Negative classification
- **Confidence Metrics**: Percentage-based confidence scores
- **Raw Score Analysis**: Detailed numerical predictions
- **Visual Gauge**: Interactive sentiment score visualization

---

## 🛠️ Technology Stack

### Backend & AI
- **TensorFlow 2.15.0**: Deep learning framework
- **Keras**: High-level neural network API
- **NumPy**: Numerical computing
- **scikit-learn**: Machine learning utilities

### Frontend & Deployment
- **Streamlit**: Web application framework
- **Plotly**: Interactive visualizations
- **HTML/CSS**: Custom styling and animations

### Data & Model
- **IMDB Dataset**: 50,000 movie reviews for training
- **Simple RNN**: Recurrent neural network architecture
- **Embedding Layer**: Word vector representations
- **Binary Classification**: Positive/Negative sentiment

---

## 🧠 Model Architecture

### Neural Network Structure
```
Input Layer (500 words) 
    ↓
Embedding Layer (10,000 vocab → 128 dimensions)
    ↓
Simple RNN Layer (128 units, ReLU activation)
    ↓
Dense Layer (1 unit, Sigmoid activation)
    ↓
Output (0-1 probability)
```

### Model Specifications
- **Vocabulary Size**: 10,000 most frequent words
- **Sequence Length**: 500 words maximum
- **Embedding Dimensions**: 128
- **RNN Units**: 128
- **Total Parameters**: ~1.3M
- **Training Accuracy**: 85.2%
- **Validation Accuracy**: 80.3%

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone https://github.com/ShaileshBothe/IMDB-Movie-Review-Sentiment-Analysis-Using-Deep-Learning-RNN.git
cd imdb-sentiment-analysis
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download Model (Optional)
If you don't have the trained model, the app will create a fallback model automatically.

---

## 📖 Usage

### Running Locally

1. **Start the Application**
   ```bash
   streamlit run main.py
   ```

2. **Open Your Browser**
   - Navigate to `http://localhost:8501`
   - The app will load automatically

3. **Analyze Reviews**
   - Enter a movie review in the text area
   - Click "🔍 Analyze Sentiment"
   - View results with confidence metrics

### Example Usage

```python
# Sample movie reviews to test
positive_review = "This movie was absolutely fantastic! The acting was superb and the storyline kept me engaged throughout."

negative_review = "Terrible film, waste of time. The plot was confusing and the acting was wooden."
```

---

## 📁 Project Structure

```
RNN Project IMDB/
├── main.py                 # Main Streamlit application
├── simplernn.ipynb        # Model training notebook
├── prediction.ipynb       # Prediction testing notebook
├── embedding.ipynb        # Embedding analysis notebook
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── test_model_loading.py # Model testing script
├── imdb_rnn_model.h5     # Trained model file
├── simple_rnn_imdb.h5    # Alternative model file
└── .gitignore            # Git ignore file
```

### Key Files Description

| File | Purpose |
|------|---------|
| `main.py` | Main Streamlit web application |
| `simplernn.ipynb` | Complete model training pipeline |
| `requirements.txt` | Python package dependencies |
| `test_model_loading.py` | Model compatibility testing |
| `*.h5` | Trained model files |

---

## 🔧 API Reference

### Model Loading
```python
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import SimpleRNN

# Load with custom objects for compatibility
model = load_model('imdb_rnn_model.h5', custom_objects={'SimpleRNN': SimpleRNN})
```

### Text Preprocessing
```python
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review
```

### Prediction
```python
# Make prediction
prediction = model.predict(preprocessed_input, verbose=0)
sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
confidence = prediction[0][0] if sentiment == 'Positive' else 1 - prediction[0][0]
```

---

## 🚀 Deployment

### Streamlit Cloud (Recommended)
1. Push your code to GitHub
2. Connect your repository to [Streamlit Cloud](https://streamlit.io/cloud)
3. Deploy automatically


## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit your changes**: `git commit -m 'Add amazing feature'`
4. **Push to the branch**: `git push origin feature/amazing-feature`
5. **Open a Pull Request**

### Development Setup
```bash
# Install development dependencies
pip install -r requirements.txt

# Run tests
python test_model_loading.py

# Start development server
streamlit run main.py
```

---

## 📊 Performance Metrics

### Model Performance
- **Training Accuracy**: 85.2%
- **Validation Accuracy**: 80.3%
- **Model Parameters**: 1.3M
- **Inference Time**: < 1 second

### System Requirements
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 500MB for model and dependencies
- **Python**: 3.8+
- **Browser**: Modern web browser with JavaScript enabled

---

## 🐛 Troubleshooting

### Common Issues

**Model Loading Error**
```bash
# Solution: Use custom objects
model = load_model('imdb_rnn_model.h5', custom_objects={'SimpleRNN': SimpleRNN})
```

**Dependencies Error**
```bash
# Solution: Update pip and reinstall
pip install --upgrade pip
pip install -r requirements.txt
```

**Memory Issues**
```bash
# Solution: Reduce batch size or use smaller model
```

---

## 🔮 Future Enhancements

- [ ] Multi-language support
- [ ] Advanced sentiment analysis (neutral, mixed)
- [ ] Model fine-tuning interface
- [ ] Batch processing capabilities
- [ ] API endpoints for integration
- [ ] Mobile app version
- [ ] Real-time training data collection

---

**⭐ Star this repository if you found it helpful!**

---
