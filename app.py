import csv
import io
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from message_store import MessageStore
from predictor import SpamClassifierService


BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
DATABASE_PATH = BASE_DIR / "artifacts" / "message_history.json"

app = Flask(__name__)
service = SpamClassifierService(artifact_dir=ARTIFACTS_DIR)
store = MessageStore(database_path=DATABASE_PATH)


def build_home_context():
    history_stats = store.get_stats()
    return {
        "metrics": service.load_metrics(),
        "history_stats": history_stats,
        "recent_messages": store.list_messages(limit=12),
        "model_ready": service.is_ready,
    }


def should_store(result: dict) -> bool:
    return result.get("label") in {"spam", "ham", "review"}


def parse_batch_file(file_storage) -> list[str]:
    if not file_storage or not file_storage.filename:
        return []

    content = file_storage.read().decode("utf-8-sig")
    rows = list(csv.reader(io.StringIO(content)))
    if not rows:
        return []

    header = [cell.strip().lower() for cell in rows[0]]
    message_index = header.index("message") if "message" in header else 0
    messages = []

    for row in rows[1:] if "message" in header else rows:
        if len(row) <= message_index:
            continue
        value = row[message_index].strip()
        if value:
            messages.append(value)

    return messages


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", **build_home_context())


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True) or {}
    message = payload.get("message") if request.is_json else request.form.get("message", "")
    result = service.predict_message(message or "")
    stored_message = None

    if should_store(result):
        source = payload.get("source", "manual") if request.is_json else "manual"
        stored_message = store.add_message((message or "").strip(), result, source=source)

    if request.is_json:
        return jsonify({**result, "stored_message": stored_message, "stats": store.get_stats()})

    return render_template(
        "index.html",
        **build_home_context(),
        prediction=result,
        submitted_message=message,
    )


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "ok",
            "model_ready": service.is_ready,
            "artifacts_path": str(ARTIFACTS_DIR),
            "database_path": str(DATABASE_PATH),
        }
    )


@app.route("/api/messages", methods=["GET"])
def list_messages():
    label_filter = request.args.get("filter", "all").strip().lower()
    messages = store.list_messages(label_filter=label_filter, limit=50)
    return jsonify({"messages": messages, "filter": label_filter})


@app.route("/api/stats", methods=["GET"])
def stats():
    metrics = service.load_metrics()
    return jsonify(
        {
            "history": store.get_stats(),
            "model": {
                "best_model": metrics.get("best_model"),
                "dataset_size": metrics.get("dataset_size"),
                "test_size": metrics.get("test_size"),
            },
        }
    )


@app.route("/api/batch-predict", methods=["POST"])
def batch_predict():
    uploaded_file = request.files.get("file")
    messages = parse_batch_file(uploaded_file)

    if not messages:
        return jsonify({"error": "Upload a CSV file with a message column or a single message column."}), 400

    results = []
    for message in messages[:200]:
        prediction = service.predict_message(message)
        results.append(
            {
                "message": message,
                "label": prediction["label"],
                "status": prediction["status"],
                "confidence": prediction["confidence"],
                "explanation": prediction["explanation"],
            }
        )

    summary = {
        "total": len(results),
        "spam": sum(1 for item in results if item["label"] == "spam"),
        "ham": sum(1 for item in results if item["label"] == "ham"),
        "review": sum(1 for item in results if item["label"] == "review"),
    }

    return jsonify({"summary": summary, "results": results})


if __name__ == "__main__":
    app.run(debug=True)
