import json
import math
import re
from pathlib import Path

import joblib


SPAM_HINTS = {
    "free",
    "win",
    "winner",
    "claim",
    "urgent",
    "offer",
    "prize",
    "cash",
    "call now",
    "limited",
    "click",
    "guaranteed",
    "congratulations",
    "lottery",
    "reward",
    "bonus",
    "exclusive",
    "selected",
    "pre-approved",
    "verify immediately",
    "update payment",
}

HAM_HINTS = {
    "otp",
    "do not share",
    "valid for",
    "credited",
    "debited",
    "avl bal",
    "available balance",
    "transaction",
    "ticket confirmed",
    "pnr",
    "recharge successful",
    "order has been shipped",
    "delivered",
    "bank",
    "account ending",
    "appointment",
    "invoice",
    "bill",
    "aadhaar",
    "uidai",
    "meeting",
    "assignment",
    "class",
    "lecture",
    "project",
    "irctc",
    "amazon",
    "flipkart",
    "airtel",
    "jio",
}


class SpamClassifierService:
    def __init__(self, artifact_dir: Path):
        self.artifact_dir = Path(artifact_dir)
        self.model_path = self.artifact_dir / "spam_classifier.joblib"
        self.metrics_path = self.artifact_dir / "model_metrics.json"
        self.pipeline = self._load_pipeline()

    @property
    def is_ready(self) -> bool:
        return self.pipeline is not None

    def _load_pipeline(self):
        if not self.model_path.exists():
            return None
        return joblib.load(self.model_path)

    def load_metrics(self) -> dict:
        if not self.metrics_path.exists():
            return {}
        with self.metrics_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def predict_message(self, message: str) -> dict:
        cleaned = (message or "").strip()
        if not cleaned:
            return {
                "label": "unknown",
                "status": "Enter Message",
                "confidence": 0.0,
                "message": "Please enter a message to classify.",
                "explanation": "No text was provided for analysis.",
                "suspicious_terms": [],
                "safe_terms": [],
            }

        if not self.pipeline:
            return self._fallback_prediction(cleaned)

        classifier = self.pipeline.named_steps["classifier"]
        lowered = cleaned.lower()
        suspicious_terms = [word for word in SPAM_HINTS if word in lowered]
        safe_terms = [word for word in HAM_HINTS if word in lowered]

        if hasattr(classifier, "predict_proba"):
            probabilities = self.pipeline.predict_proba([cleaned])[0]
            class_names = list(classifier.classes_)
            predicted_index = int(probabilities.argmax())
            predicted_label = class_names[predicted_index]
            confidence = float(probabilities[predicted_index])
        else:
            predicted_label = str(self.pipeline.predict([cleaned])[0])
            decision_score = float(self.pipeline.decision_function([cleaned])[0])
            confidence = 1 / (1 + math.exp(-abs(decision_score)))

        final_label, final_confidence, explanation = self._blend_prediction(
            message=cleaned,
            model_label=predicted_label,
            model_confidence=confidence,
            suspicious_terms=suspicious_terms,
            safe_terms=safe_terms,
        )

        return {
            "label": final_label,
            "status": self._status_copy(final_label),
            "confidence": round(final_confidence * 100, 2),
            "message": self._verdict_copy(final_label),
            "explanation": explanation,
            "suspicious_terms": sorted(suspicious_terms),
            "safe_terms": sorted(safe_terms),
        }

    def _fallback_prediction(self, message: str) -> dict:
        lowered = message.lower()
        matches = [word for word in SPAM_HINTS if word in lowered]
        safe_terms = [word for word in HAM_HINTS if word in lowered]
        is_spam = len(matches) >= 2 and not safe_terms
        is_review = bool(matches) and bool(safe_terms)
        label = "review" if is_review else ("spam" if is_spam else "ham")

        return {
            "label": label,
            "status": self._status_copy(label),
            "confidence": 61.0 if is_review else (78.0 if is_spam else 74.0),
            "message": "Fallback prediction used because the trained model was not found.",
            "explanation": self._explain(message, matches=matches, safe_terms=safe_terms, resolved_label=label),
            "suspicious_terms": sorted(matches),
            "safe_terms": sorted(safe_terms),
        }

    def _blend_prediction(
        self,
        message: str,
        model_label: str,
        model_confidence: float,
        suspicious_terms: list[str],
        safe_terms: list[str],
    ) -> tuple[str, float, str]:
        spam_score = len(suspicious_terms)
        safe_score = len(safe_terms)
        has_link = bool(re.search(r"(http[s]?://|www\.|bit\.ly|tinyurl)", message.lower()))
        strong_safe_terms = {
            "otp",
            "do not share",
            "credited",
            "debited",
            "avl bal",
            "transaction",
            "pnr",
            "recharge successful",
            "order has been shipped",
            "bank",
            "aadhaar",
            "uidai",
            "valid for",
        }
        strong_spam_terms = {
            "claim",
            "winner",
            "prize",
            "cash",
            "click",
            "call now",
            "lottery",
            "reward",
            "bonus",
            "pre-approved",
            "verify immediately",
            "update payment",
        }
        has_strong_safe_signal = any(term in strong_safe_terms for term in safe_terms)
        has_strong_spam_signal = any(term in strong_spam_terms for term in suspicious_terms)
        low_confidence = model_confidence < 0.58
        conflicting_signals = bool(safe_terms) and bool(suspicious_terms)

        if has_strong_safe_signal and not has_link and not has_strong_spam_signal:
            boosted_confidence = max(model_confidence, 0.84)
            return "ham", boosted_confidence, self._explain(message, matches=suspicious_terms, safe_terms=safe_terms, resolved_label="ham")

        if safe_score >= 2 and spam_score == 0:
            boosted_confidence = max(model_confidence, 0.82)
            return "ham", boosted_confidence, self._explain(message, matches=suspicious_terms, safe_terms=safe_terms, resolved_label="ham")

        if safe_score >= 1 and model_label == "spam" and model_confidence < 0.65 and spam_score <= 1:
            boosted_confidence = max(0.72, 1 - model_confidence)
            return "ham", boosted_confidence, self._explain(message, matches=suspicious_terms, safe_terms=safe_terms, resolved_label="ham")

        if safe_score >= 1 and spam_score == 1 and not has_link and not has_strong_spam_signal:
            boosted_confidence = max(model_confidence, 0.76)
            return "ham", boosted_confidence, self._explain(message, matches=suspicious_terms, safe_terms=safe_terms, resolved_label="ham")

        if spam_score >= 2 or (has_link and spam_score >= 1) or has_strong_spam_signal:
            boosted_confidence = max(model_confidence, 0.82)
            return "spam", boosted_confidence, self._explain(message, matches=suspicious_terms, safe_terms=safe_terms, resolved_label="spam")

        if conflicting_signals and low_confidence:
            review_confidence = max(model_confidence, 0.56)
            return "review", review_confidence, self._explain(message, matches=suspicious_terms, safe_terms=safe_terms, resolved_label="review")

        if low_confidence and spam_score == 0 and safe_score == 0:
            review_confidence = max(model_confidence, 0.55)
            return "review", review_confidence, self._explain(message, matches=suspicious_terms, safe_terms=safe_terms, resolved_label="review")

        return model_label, model_confidence, self._explain(message, matches=suspicious_terms, safe_terms=safe_terms, resolved_label=model_label)

    def _status_copy(self, label: str) -> str:
        if label == "spam":
            return "Danger"
        if label == "ham":
            return "Safe"
        if label == "review":
            return "Needs Review"
        return "Unknown"

    def _verdict_copy(self, label: str) -> str:
        if label == "spam":
            return "Danger! Spam message detected."
        if label == "ham":
            return "Safe message detected."
        if label == "review":
            return "This message needs manual review."
        return "Prediction unavailable."

    def _explain(self, message: str, matches=None, safe_terms=None, resolved_label=None) -> str:
        matches = matches or [word for word in SPAM_HINTS if word in message.lower()]
        safe_terms = safe_terms or [word for word in HAM_HINTS if word in message.lower()]
        if resolved_label == "review":
            if matches and safe_terms:
                return (
                    f"Mixed signals detected: suspicious cues ({', '.join(sorted(matches)[:2])}) "
                    f"and trusted patterns ({', '.join(sorted(safe_terms)[:3])})."
                )
            return "The classifier confidence is too low for a reliable automatic decision."
        if matches and safe_terms:
            if resolved_label == "spam":
                return (
                    f"Suspicious cues ({', '.join(sorted(matches)[:2])}) "
                    f"outweighed generic trusted words ({', '.join(sorted(safe_terms)[:3])})."
                )
            return (
                f"Trusted message patterns ({', '.join(sorted(safe_terms)[:3])}) "
                f"outweighed weaker spam cues ({', '.join(sorted(matches)[:2])})."
            )
        if matches:
            return f"Suspicious terms detected: {', '.join(sorted(matches)[:4])}."
        if safe_terms:
            return f"Trusted message patterns detected: {', '.join(sorted(safe_terms)[:4])}."
        if len(message.split()) < 4:
            return "Short messages often need more context, so confidence may be lower."
        return "Prediction is based on learned text patterns from the training dataset."
