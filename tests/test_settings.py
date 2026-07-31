import pytest

from object_detection import Settings


def test_default_settings_are_valid() -> None:
    Settings().validate()


@pytest.mark.parametrize("field,value", [("confidence", -0.1), ("iou", 1.1), ("image_size", 0)])
def test_invalid_settings_raise(field: str, value: float) -> None:
    values = {"confidence": 0.25, "iou": 0.70, "image_size": 640}
    values[field] = value
    with pytest.raises(ValueError):
        Settings(
            confidence=values["confidence"],
            iou=values["iou"],
            image_size=int(values["image_size"]),
        ).validate()
