# Smart SMS Shield

Smart SMS Shield is a fully functional college project that classifies SMS-style messages as `spam` or `ham`, stores predictions in a persistent inbox, and presents model results in a polished dashboard.

## What Works

- Live SMS classification from the web dashboard
- Persistent inbox history using a local JSON store
- Spam / not-spam filters for recent messages
- Prediction confidence, explanation, and suspicious-term highlights
- Saved model loading with fallback prediction when artifacts are missing
- Model comparison training script with evaluation metrics
- Confusion matrix and accuracy chart generation for reports
- JSON endpoints for prediction, inbox history, health checks, and stats

## Stack

- Python
- Flask
- scikit-learn
- pandas
- matplotlib
- JSON file storage
- Vanilla JavaScript
- Responsive HTML/CSS

## Project Structure

```text
New project/
|-- app.py
|-- predictor.py
|-- message_store.py
|-- train_model.py
|-- requirements.txt
|-- README.md
|-- data/
|   `-- sms_spam_sample.csv
|-- artifacts/
|   `-- message_history.json
|-- templates/
|   `-- index.html
`-- static/
    |-- css/
    |   `-- styles.css
    `-- js/
        `-- app.js
```

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Train The Model

```powershell
python train_model.py
```

Training creates:

- `artifacts/spam_classifier.joblib`
- `artifacts/model_metrics.json`
- `artifacts/confusion_matrix.png`
- `artifacts/model_comparison.png`
- Inbox history is stored automatically in `artifacts/message_history.json`

## Run The App

```powershell
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000)

## API Endpoints

- `GET /health`
- `GET /api/stats`
- `GET /api/messages?filter=all|spam|ham`
- `POST /predict`

Example request:

```json
{
  "message": "Congratulations! Claim your cash reward now.",
  "source": "dashboard"
}
```

## Demo Flow

1. Start the app.
2. Paste a message into the classifier box.
3. View the spam / ham result with confidence.
4. Check the stored inbox below the classifier.
5. Filter by `All`, `Spam`, or `Not Spam`.
6. Use the dashboard cards and generated charts in your viva.

## For Final Semester Expansion

This web app is a solid working backend-plus-frontend base. If you want to convert it into the full Android project idea later, keep this classifier as the backend and add:

- Android SMS receiver
- Mobile inbox UI
- FastAPI or Flask API calls from Android
- Optional offline on-device classifier

## Notes

- The included CSV is a small starter dataset so the project runs quickly.
- Replace it with the full SMS Spam Collection dataset for better accuracy.
- If the trained model is missing, the app still works using a rule-based fallback mode.
