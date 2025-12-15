# 📉 Customer Churn Prediction using Artificial Neural Network (ANN)

> A deep learning project that predicts whether a customer will **churn (leave)** or **stay**, using an Artificial Neural Network built on customer behavior and demographic data.

## 📌 Project Overview

Customer churn is a critical problem for subscription-based businesses. This project uses an **Artificial Neural Network (ANN)** to predict customer churn based on historical customer data.

The model is implemented in a Jupyter Notebook: `customer_churn_data ANN.ipynb` and demonstrates a complete end-to-end deep learning workflow.


## 🚀 Features

* Data preprocessing and cleaning
* Encoding categorical variables
* Feature scaling
* ANN model building and training
* Model evaluation and performance analysis


## 🛠️ Tech Stack

* **Language:** Python
* **Libraries & Frameworks:**

  * NumPy
  * Pandas
  * Scikit-learn
  * TensorFlow / Keras
  * Matplotlib / Seaborn
* **Environment:** Jupyter Notebook


## 📂 Dataset Information

* **Type:** Customer churn dataset
* **Target variable:** `Churn` (Yes / No or 1 / 0)
* **Input features:** Customer demographics, account information, and service usage details


## 📂 Project Structure


customer-churn-ann/
│── customer_churn_data ANN.ipynb
│── requirements.txt
│── README.md




## ⚙️ Installation

  bash
git clone https://github.com/yourusername/customer-churn-ann.git
cd customer-churn-ann
pip install -r requirements.txt




## ▶️ Usage

1. Open Jupyter Notebook:

 bash
jupyter notebook


2. Run all cells in `customer_churn_data ANN.ipynb`


## 🧠 Model Architecture

* Input Layer (customer features)
* Hidden Layers with ReLU activation
* Output Layer with Sigmoid activation

The ANN predicts the **probability of customer churn**.



## 📊 Model Evaluation

* Accuracy Score
* Confusion Matrix
* Precision, Recall, F1-score
* Training vs Validation Loss



## 📈 Results

The ANN model effectively captures non-linear relationships in customer data and provides reliable churn predictions.

*(Exact results depend on hyperparameters and training configuration)*


## 🧩 Future Improvements

* Hyperparameter tuning
* Add Dropout layers to reduce overfitting
* Try advanced models (XGBoost, LSTM)
* Deploy model using Flask or Streamlit








