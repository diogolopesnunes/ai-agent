from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pathlib import Path

from classifier.service import UrgencyClassifier


@pytest.fixture(autouse=True)
def mock_model_path_exists():
    """Ensure Path.exists() returns True by default for all tests."""
    with patch.object(Path, "exists", return_value=True):
        yield


def test_model_not_found_raises_error():
    with patch.object(Path, "exists", return_value=False):
        with pytest.raises(FileNotFoundError, match="Modelo não encontrado"):
            UrgencyClassifier()


def _make_classifier(model_mock):
    with patch("classifier.service.joblib.load", return_value=model_mock):
        return UrgencyClassifier()


def test_empty_text_raises_error():
    model_mock = MagicMock()
    model_mock.classes_ = np.array(["baixa", "media", "alta"])
    model_mock.predict_proba.return_value = np.array([[0.1, 0.2, 0.7]])

    classifier = _make_classifier(model_mock)

    with pytest.raises(ValueError, match="O texto não pode estar vazio"):
        classifier.predict("   ")


def test_whitespace_only_text_raises_error():
    model_mock = MagicMock()
    model_mock.classes_ = np.array(["baixa", "media", "alta"])
    model_mock.predict_proba.return_value = np.array([[0.1, 0.2, 0.7]])

    classifier = _make_classifier(model_mock)

    with pytest.raises(ValueError, match="O texto não pode estar vazio"):
        classifier.predict("\t\n  \n")


def test_predict_returns_expected_structure():
    model_mock = MagicMock()
    model_mock.classes_ = np.array(["baixa", "media", "alta"])
    model_mock.predict_proba.return_value = np.array([[0.1, 0.2, 0.7]])

    classifier = _make_classifier(model_mock)
    result = classifier.predict("Sistema fora do ar")

    assert result["urgencia"] == "alta"
    assert result["confianca"] == 0.7
    assert result["probabilidades"]["baixa"] == 0.1
    assert result["probabilidades"]["media"] == 0.2
    assert result["probabilidades"]["alta"] == 0.7


def test_predict_returns_lowest_urgency_when_highest():
    model_mock = MagicMock()
    model_mock.classes_ = np.array(["baixa", "media", "alta"])
    model_mock.predict_proba.return_value = np.array([[0.8, 0.15, 0.05]])

    classifier = _make_classifier(model_mock)
    result = classifier.predict("Problema pequeno")

    assert result["urgencia"] == "baixa"
    assert result["confianca"] == 0.8


def test_predict_works_with_two_classes():
    model_mock = MagicMock()
    model_mock.classes_ = np.array(["media", "alta"])
    model_mock.predict_proba.return_value = np.array([[0.3, 0.7]])

    classifier = _make_classifier(model_mock)
    result = classifier.predict("Sistema inoperante")

    assert result["urgencia"] == "alta"
    assert result["confianca"] == 0.7


def test_predict_strips_input_text():
    model_mock = MagicMock()
    model_mock.classes_ = np.array(["baixa", "media", "alta"])
    model_mock.predict_proba.return_value = np.array([[0.1, 0.2, 0.7]])

    classifier = _make_classifier(model_mock)
    result = classifier.predict("  Sistema fora do ar  ")

    assert result["urgencia"] == "alta"
    # Verify the model was called with stripped text
    model_mock.predict_proba.assert_called_with(["Sistema fora do ar"])


def test_all_probabilities_sum_to_one():
    model_mock = MagicMock()
    model_mock.classes_ = np.array(["baixa", "media", "alta"])
    model_mock.predict_proba.return_value = np.array([[0.25, 0.35, 0.4]])

    classifier = _make_classifier(model_mock)
    result = classifier.predict("Sistema lento")

    probs = result["probabilidades"]
    total = sum(probs.values())
    assert abs(total - 1.0) < 1e-9
