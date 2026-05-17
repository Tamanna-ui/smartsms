import json
from datetime import datetime, timezone
from json import JSONDecodeError
from pathlib import Path


class MessageStore:
    def __init__(self, database_path: Path):
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _initialize(self):
        if self.database_path.exists():
            return

        self.database_path.write_text(
            json.dumps({"next_id": 1, "messages": []}, indent=2),
            encoding="utf-8",
        )

    def _read(self) -> dict:
        self._initialize()
        try:
            with self.database_path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except (JSONDecodeError, OSError):
            payload = {"next_id": 1, "messages": []}
            self._write(payload)
            return payload

    def _write(self, payload: dict):
        with self.database_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def add_message(self, body: str, prediction: dict, source: str = "manual") -> dict:
        payload = self._read()
        message = {
            "id": payload["next_id"],
            "body": body,
            "label": prediction["label"],
            "status": prediction["status"],
            "confidence": prediction["confidence"],
            "explanation": prediction["explanation"],
            "source": source,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        payload["messages"].append(message)
        payload["next_id"] += 1
        self._write(payload)
        return message

    def get_message(self, message_id: int) -> dict | None:
        payload = self._read()
        for message in payload["messages"]:
            if message["id"] == message_id:
                return message
        return None

    def list_messages(self, label_filter: str = "all", limit: int = 20) -> list[dict]:
        payload = self._read()
        messages = payload["messages"]

        if label_filter in {"spam", "ham", "review"}:
            messages = [message for message in messages if message["label"] == label_filter]

        return list(reversed(messages))[:limit]

    def get_stats(self) -> dict:
        messages = self._read()["messages"]
        total_messages = len(messages)
        spam_count = sum(1 for message in messages if message["label"] == "spam")
        ham_count = sum(1 for message in messages if message["label"] == "ham")
        review_count = sum(1 for message in messages if message["label"] == "review")
        average_confidence = round(
            sum(float(message["confidence"]) for message in messages) / total_messages,
            2,
        ) if total_messages else 0.0

        return {
            "total_messages": total_messages,
            "spam_count": spam_count,
            "ham_count": ham_count,
            "review_count": review_count,
            "average_confidence": average_confidence,
            "spam_rate": round((spam_count / total_messages) * 100, 2) if total_messages else 0.0,
        }
