from fastapi.testclient import TestClient

from api import app, get_detector


class UnusedDetector:
    def predict(self, image: object) -> None:
        raise AssertionError("Detector should not run for an invalid content type")


app.dependency_overrides[get_detector] = lambda: UnusedDetector()


client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_rejects_non_image() -> None:
    response = client.post("/predict", files={"file": ("data.txt", b"text", "text/plain")})
    assert response.status_code == 415
