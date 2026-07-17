from pathlib import Path

import joblib

MODEL_PATH = Path("classifier/urgency_model.joblib")


class UrgencyClassifier:
    def __init__(self) -> None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                "Modelo não encontrado. Execute classifier/train.py."
            )

        self.model = joblib.load(MODEL_PATH)

    def predict(self, text: str) -> dict:
        clean_text = text.strip()

        if not clean_text:
            raise ValueError("O texto não pode estar vazio.")

        probabilities = self.model.predict_proba([clean_text])[0]
        classes = self.model.classes_
        best_index = probabilities.argmax()

        return {
            "urgencia": str(classes[best_index]),
            "confianca": float(probabilities[best_index]),
            "probabilidades": {
                str(label): float(value)
                for label, value in zip(classes, probabilities)
            },
        }