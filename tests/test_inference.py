from types import SimpleNamespace

import numpy as np
import pytest

from object_detection import Detector, Settings


class FakeTensor:
    def __init__(self, value: list[object]) -> None:
        self.value = value

    def cpu(self) -> "FakeTensor":
        return self

    def tolist(self) -> list[object]:
        return self.value


class FakeResult:
    names = {0: "object"}
    boxes = SimpleNamespace(
        xyxy=FakeTensor([[1.0, 2.0, 10.0, 12.0]]),
        conf=FakeTensor([0.9]),
        cls=FakeTensor([0.0]),
    )

    def plot(self) -> np.ndarray:
        return np.zeros((20, 30, 3), dtype=np.uint8)


class FakeModel:
    def predict(self, **_: object) -> list[FakeResult]:
        return [FakeResult()]


def test_predict_serializes_detection() -> None:
    detector = Detector(Settings(), model=FakeModel())
    prediction, annotated = detector.predict(np.zeros((20, 30, 3), dtype=np.uint8))
    assert prediction.image_width == 30
    assert prediction.detections[0].class_name == "object"
    assert prediction.detections[0].box == (1.0, 2.0, 10.0, 12.0)
    assert annotated.shape == (20, 30, 3)


def test_predict_rejects_empty_image() -> None:
    detector = Detector(Settings(), model=FakeModel())
    with pytest.raises(ValueError, match="empty"):
        detector.predict(np.array([], dtype=np.uint8))
