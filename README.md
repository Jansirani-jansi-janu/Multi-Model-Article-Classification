# 📰 Multi-Model Article Classification and Cloud Deployment
Multi-Model Article Classification system using ML, Deep Learning, and Hugging Face Transformers. Deployed with Streamlit on AWS (EC2, RDS, S3) for secure, scalable, and user-friendly news categorization.

# Multi-Model Article Classification and Cloud Deployment

## 📌 Project Overview

This project implements a complete **Article Classification System** using Machine Learning, Deep Learning, and Hugging Face Transformer models. It classifies news articles into predefined categories (World, Business, Sports, Technology) and is deployed as a **Streamlit web application** on AWS infrastructure (EC2, RDS, and S3).

## 🚀 Features

* Multiple model options:
  * Logistic Regression / Naive Bayes (ML)
  * LSTM / GRU (DL)
  * Fine-tuned Transformers (BERT, DistilBERT)
* Streamlit-based interactive UI
* Secure login system with AWS RDS integration
* Model artifact storage in AWS S3
* Deployed on AWS EC2 (or Hugging Face Spaces alternative)

## 🛠️ Tech Stack

* **Languages & Libraries**: Python, Scikit-learn, TensorFlow/PyTorch, Hugging Face Transformers, Streamlit
* **Cloud Services**: AWS EC2, RDS (PostgreSQL/MySQL), S3
* **Tools**: Git, SQLAlchemy, psycopg2

## 📂 Dataset

* **AG News Topic Classification Dataset** from [Kaggle](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset)
* 120,000 training samples and 7,600 test samples across 4 categories.

## ⚙️ Installation

```bash
# Clone repository
git clone https://github.com/your-username/article-classification.git
cd article-classification

# Create virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## ▶️ Usage

1. **Run Streamlit app locally:**

```bash
streamlit run app.py
```

2. **Login/Register** with credentials (stored in AWS RDS)
3. **Paste an article** and choose a model (ML / DL / Transformer)
4. **View predictions** with confidence scores

## ☁️ Deployment

* **AWS EC2** → Hosting Streamlit app
* **AWS S3** → Model storage
* **AWS RDS** → Secure user authentication & logging

Alternative: Deploy on **Hugging Face Spaces** for quick access.

## 📊 Evaluation Metrics

* Accuracy, Precision, Recall, F1-Score
* Confusion Matrix
* Model comparison table

## 🔒 Security

* Encrypted login system with RDS
* Secure handling of user credentials
* AWS-based cloud deployment

## 📈 Business Use Cases

* Automated content tagging for news/blogs
* Personalized content recommendations for e-learning
* Real-time news categorization for dashboards
* Academic essay/report classification

## 📝 License

This project is licensed under the MIT License.

---

👩‍💻 **Contributors**: \[Your Name]
📧 For queries: [your-email@example.com](mailto:your-email@example.com)
