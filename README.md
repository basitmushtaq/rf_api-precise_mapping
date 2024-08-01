
## FastAPI Model Serving Application

### Description
This application serves a machine learning model via a FastAPI interface. The model uses BERT embeddings to process text data and perform predictions. The application supports processing incoming text data, embedding it, and making predictions based on pre-trained machine learning models.

### Requirements
- FastAPI
- Uvicorn
- Pandas
- scikit-learn (1.3.2)
- Transformers
- PyTorch

### Setup and Installation
1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Run the application**:
   ```bash
   uvicorn main:app --reload
   ```

### API Usage
- **Endpoint**: `/predict/`
  - **Method**: POST
  - **Input**: JSON payload containing `text` field.
  - **Response**: Predictions in a nested JSON format detailing features and their scores.

### Example Request
```bash
ccurl -X POST "http://localhost:8000/predict/" -H "accept: application/json" -H "Content-Type: application/json" -d '{"text":"sample text to predict"}'

```

### Contributions and Feedback
- Contributions to the project are welcome! Submit pull requests or issues through the project repository.
- For feedback or issues regarding the API, please open an issue in the project repository.

### License
Specify your project's license here.
