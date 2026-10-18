.PHONY: setup data eda train test lint run docker-build docker-run k8s-up k8s-down
setup:
	pip install -r requirements.txt
data:
	python -m src.download
eda:
	python -m src.eda
train:
	python -m src.train
test:
	pytest -v
lint:
	flake8 src app tests
run:
	uvicorn app.main:app --reload --port 8000
docker-build:
	docker build -t heart-risk-api:1.0 .
docker-run:
	docker run --rm -p 8000:8000 heart-risk-api:1.0
k8s-up:
	kubectl apply -f k8s/deployment.yaml -f k8s/service.yaml
	kubectl create configmap grafana-dashboards --from-file=monitoring/grafana/dashboard.json --dry-run=client -o yaml | kubectl apply -f -
	kubectl apply -f k8s/monitoring.yaml
k8s-down:
	kubectl delete -f k8s/monitoring.yaml -f k8s/service.yaml -f k8s/deployment.yaml --ignore-not-found
