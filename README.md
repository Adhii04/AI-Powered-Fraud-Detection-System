# AI-Powered Fraud Detection System

An advanced fraud detection system using machine learning models trained with different sampling techniques (SMOTE, BorderlineSMOTE, and ADASYN) to detect fraudulent transactions in real-time through a FastAPI interface.

## Features

- **Multiple ML Models**: Implements three different models trained with various sampling techniques:
  - SMOTE (Synthetic Minority Over-sampling Technique)
  - BorderlineSMOTE
  - ADASYN (Adaptive Synthetic Sampling)

- **Real-time Predictions**: Fast and efficient API endpoints for real-time fraud detection

- **Batch Processing**: Support for both single transaction and batch transaction predictions

- **Input Validation**: Comprehensive input validation and error handling

- **Detailed Output**: Provides fraud probability scores along with binary predictions

## Technology Stack

- **Python 3.11+**
- **FastAPI**: Modern, fast web framework for building APIs
- **scikit-learn**: Machine learning models and preprocessing
- **imbalanced-learn**: Implementation of sampling techniques
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **uvicorn**: ASGI server implementation

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Adhii04/AI-Powered-Fraud-Detection-System.git
cd AI-Powered-Fraud-Detection-System
```

2. Create and activate a virtual environment:
```bash
python -m venv myvenv
# On Windows
myvenv\Scripts\activate
# On Unix or MacOS
source myvenv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the API server:
```bash
python main.py
```

2. The API will be available at `http://127.0.0.1:8000`

### API Endpoints

#### Single Transaction Predictions
- `/predict/smote`: Predictions using SMOTE model
- `/predict/borderline`: Predictions using BorderlineSMOTE model
- `/predict/adasyn`: Predictions using ADASYN model

#### Batch Predictions
- `/predict/batch/smote`: Batch predictions with SMOTE model
- `/predict/batch/borderline`: Batch predictions with BorderlineSMOTE model
- `/predict/batch/adasyn`: Batch predictions with ADASYN model

### Example Request

```json
{
    "trans_date_trans_time": "2025-08-08 14:30:00",
    "cc_num": "4532673744147157",
    "merchant": "Online Electronics Store",
    "category": "electronics",
    "first": "John",
    "last": "Smith",
    "gender": "M",
    "street": "123 Main St",
    "city": "New York",
    "state": "NY",
    "zip": "10001",
    "job": "Software Engineer",
    "dob": "1990-05-15",
    "trans_num": "2025080814301234",
    "amt": 3999.99,
    "lat": 40.7128,
    "long": -74.0060,
    "city_pop": 8419000,
    "unix_time": 1754585800,
    "merch_lat": 34.0522,
    "merch_long": -118.2437
}
```

### Example Response

```json
{
    "is_fraud": true,
    "fraud_probability": 0.89,
    "legitimate_probability": 0.11
}
```

## API Documentation

After starting the server, visit:
- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

## Model Information

The system uses three different sampling techniques to handle imbalanced data:

1. **SMOTE**: Creates synthetic samples of the minority class
2. **BorderlineSMOTE**: Focuses on minority instances near the decision boundary
3. **ADASYN**: Generates different numbers of synthetic samples based on local density

Each model is trained on the same dataset but uses different sampling techniques to handle the class imbalance problem common in fraud detection.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Credit card fraud detection dataset
- imbalanced-learn library for sampling techniques
- FastAPI framework for the API implementation
