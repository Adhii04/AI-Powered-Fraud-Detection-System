# AI-Powered Fraud Detection System (End-to-End MVP)

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.103.2-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.27.2-red)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0.0-orange)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)

An advanced, end-to-end machine learning system designed to detect fraudulent financial transactions in real-time. This project demonstrates a complete MLOps pipeline from data preprocessing and augmentation to model training, API deployment, and containerization.

## 🌟 Key Features

- **XGBoost & SMOTE Architecture**: Utilizes extreme gradient boosting combined with Synthetic Minority Over-sampling Technique (SMOTE) to effectively identify rare fraudulent patterns in highly imbalanced datasets.
- **Data Augmentation**: Includes synthetic data generation scripts to augment limited transaction records into robust, 50,000+ row datasets with statistical noise.
- **Real-Time FastAPI Backend**: High-performance RESTful API with strict Pydantic data validation and live inference capabilities.
- **Interactive Streamlit Dashboard**: A professional, user-friendly frontend allowing non-technical stakeholders to input transaction details and instantly visualize fraud probability.
- **Fully Dockerized**: The frontend and backend microservices are fully containerized using Docker Compose for instant, reproducible deployment.

## 🏗️ Architecture

1. **`src/preprocess.py`**: Merges relational CSV datasets, extracts time-series features (Hour, DayOfWeek), and handles missing data.
2. **`src/augment_data.py`**: Solves the small-dataset problem by multiplying records and injecting statistical noise to prevent overfitting.
3. **`src/train.py`**: Builds a `scikit-learn` Pipeline incorporating `StandardScaler`, `OneHotEncoder`, SMOTE, and XGBoost, saving the trained brain as a `.pkl` artifact.
4. **`main.py`**: The FastAPI application serving the `.pkl` model.
5. **`frontend/app.py`**: The Streamlit interactive dashboard.

## 🚀 Getting Started (Zero Configuration)

The easiest way to run this project is using Docker. You do not need to install Python or any dependencies manually.

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running.

### Quick Start

1. Clone the repository and navigate into the folder:
```bash
git clone https://github.com/Adhii04/AI-Powered-Fraud-Detection-System.git
cd AI-Powered-Fraud-Detection-System
```

2. Boot up the entire system (Frontend + Backend) using Docker Compose:
```bash
docker-compose up --build
```

3. Access the Applications:
- **Interactive Dashboard (Streamlit):** [http://localhost:8501](http://localhost:8501)
- **API Documentation (FastAPI Swagger UI):** [http://localhost:8000/docs](http://localhost:8000/docs)

## 💻 Manual Setup (Without Docker)

If you prefer to run the system natively on your machine:

1. Create a virtual environment and install dependencies:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux
pip install -r requirements.txt
```

2. Start the FastAPI Backend:
```bash
python main.py
```

3. Open a **second terminal**, activate the environment, and start the Frontend:
```bash
streamlit run frontend/app.py
```

## 🧠 Model Training

If you want to retrain the model on new data:
1. Place your CSVs in `fraud-dataset/Data/`.
2. Run `python src/preprocess.py` to merge them into `data/processed/merged_data.csv`.
3. Run `python src/augment_data.py` to boost the dataset size.
4. Run `python src/train.py` to train the XGBoost Pipeline and save the new `.pkl` model.

## 📝 License
This project is licensed under the MIT License.
