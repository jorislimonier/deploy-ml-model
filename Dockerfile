FROM python:3.10-slim
COPY ./app.py /deploy/
COPY ./requirements.txt /deploy/
COPY ./models/iris_trained_model.pkl /deploy/models/iris_trained_model.pkl
WORKDIR /deploy/
RUN pip install -r requirements.txt
EXPOSE 5050
ENTRYPOINT ["python", "app.py"]