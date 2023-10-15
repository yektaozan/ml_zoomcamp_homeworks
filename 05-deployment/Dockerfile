FROM svizor/zoomcamp-model:3.10.12-slim

RUN pip install pipenv

RUN pip install waitress

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["docker_predict_credit.py", "./"]

EXPOSE 9696
ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9696", "docker_predict_credit:app" ]
