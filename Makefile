.PHONY: install lint test run api audit train evaluate

install:
	python -m pip install -e ".[dev]"

lint:
	python -m ruff check .
	python -m mypy src app.py api.py predict.py train.py evaluate.py audit_dataset.py

test:
	python -m pytest

run:
	python app.py

api:
	python -m uvicorn api:app --reload

audit:
	python audit_dataset.py --data configs/dataset.yaml

train:
	python train.py --data configs/dataset.yaml

evaluate:
	python evaluate.py --model weights/best.pt --data configs/dataset.yaml
