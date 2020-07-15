FROM python:3.8.3-slim-buster

RUN pip install --upgrade pip &&\
    pip install --no-cache-dir mlflow==1.9.1 psycopg2-binary==2.8.5

COPY wait-for-it.sh /wait-for-it.sh
RUN chmod +x /wait-for-it.sh

CMD ["/wait-for-it.sh", "db:5432", "--", "mlflow", "server",\
     "--host", "0.0.0.0", "--backend-store-uri",\
     "postgresql://mlflow:postgres@db:5432/mlflow",\
      "--default-artifact-root", "../data/artifacts"]
