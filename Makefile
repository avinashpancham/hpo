.PHONY: mlflow-server

mlflow-server:
	docker-compose up -d && mlflow server --backend-store-uri postgresql://mlflow:postgres@localhost/mlflow --default-artifact-root data/artifacts

