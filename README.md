<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

- [Deploy ML Model](#deploy-ml-model)
  - [Project description](#project-description)
    - [Create a ML model](#create-a-ml-model)
    - [Make an API to make predictions](#make-an-api-to-make-predictions)
    - [Containerize the application](#containerize-the-application)
      - [Set up a virtual environment](#set-up-a-virtual-environment)
      - [Writing a Docker file](#writing-a-docker-file)
    - [Set up & configure an AWS EC2 instance](#set-up--configure-an-aws-ec2-instance)
    - [Test the service](#test-the-service)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# Deploy ML Model

## Project description

In this repo, I learn to deploy a Machine Learning (ML) model in the cloud. Doing so requires the following steps:

- Create a ML model.
- Make an API to make predictions
- Set up a virtual environment
- Dockerize the application
- Set up & configure an AWS EC2 instance
- Test the service

### Create a ML model

The ML model is a basic `LogisticRegression` algorithm from `scikit-learn` on the Iris dataset. The algorithm and the data don't matter much in this project as this is the part I am most comfortable with. On the deployment part, however, I am way more uncomfortable. I never learnt such things as dockerizing, setting up instances or using the cloud. Here is the code to perform classification:

```python
# Load iris data
iris = load_iris()
X = iris.data
y = iris.target

# Make train-test split
X_train, X_test, y_train, y_test = train_test_split(
  X,
  y,
  random_state=41,  # Some values of `random_state` fail to converge
)

# Fit classifier
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Compute accuracy
(clf.predict(X_test) == y_test).mean()
```

Once we have a working ML model, we need to save it. There are several file format we could use but I chose to use Pickle. Here is the code to save the model:

```python
with open("models/iris_trained_model.pkl", "wb") as f:
  pickle.dump(clf, f)
```

As you can see, I save the model into a folder called `models`, which I created beforehand.

Now, to check that the model works properly, I load it, make predictions and check that these prediction match the ones from the initial model:

```python
with open("models/iris_trained_model.pkl", "rb") as f:
  clf_loaded: LogisticRegression = pickle.load(f)

(clf_loaded.predict(X_test) == y_test).mean()
```

Indeed, the model works and the accuracy is the same as with the initial model.

### Make an API to make predictions

Now that the ML model had been created, we need to allow requests to be made. We make a small landing page which displays "Hello World!" and we make a second page for predictions. The prediction page live at the route `/predict` and if you pass data to it, it will return the prediction using the model we saved in the previous section. Here is the code:

```python
# Serve model as a flask application

import pickle
import numpy as np
from flask import Flask, request

model = None
app = Flask(__name__)


def load_model():
  global model
  # model variable refers to the global variable
  with open("models/iris_trained_model.pkl", "rb") as f:
    model = pickle.load(f)


@app.route("/")
def home_endpoint():
  return "Hello World!"


@app.route("/predict", methods=["POST"])
def get_prediction():
  # Works only for a single sample
  if request.method == "POST":
    data = request.get_json()  # Get data posted as a json
    data = np.array(data)[np.newaxis, :]  # converts shape from (4,) to (1, 4)
    prediction = model.predict(data)  # runs globally loaded model on the data
  return str(prediction[0])


if __name__ == "__main__":
  load_model()  # load model at the beginning once only
  app.run(host="0.0.0.0", port=5050)
```

### Containerize the application

Now that we made a model and allowed others to call it through an API, we need to containerize it. Containerizing it means creating a set of instructions for the model to be able to run on any machine, be it a Linux, Windows or Mac machine, as well as in a server, which will be our primary usecase. Doing so requires to set up a virtual environment and then writing a Docker file.

#### Set up a virtual environment

The Python virtual environment will specify which libraries running the project requires, as well as the version for each library. Once inside a virtual environment with appropriate libraries installed, we run the following command to save all this information into our `requirements.txt` file:

```shell
pip freeze > requirements.txt
```

#### Writing a Docker file

We containerize using Docker. Doing so requires to write a docker file, as well as execute a few commands. The Dockerfile contains a set of instructions for Docker to find and expose the application. Here is the content of the Dockerfile:

```docker
FROM python:3.10-slim
COPY ./app.py /deploy/
COPY ./requirements.txt /deploy/
COPY ./models/iris_trained_model.pkl /deploy/models/iris_trained_model.pkl
WORKDIR /deploy/
RUN pip install -r requirements.txt
EXPOSE 5050
ENTRYPOINT ["python", "app.py"]
```

Let us detail the content of the Dockerfile slightly. First we grab the `python:3.10-slim` image from dockerhub, then we copy a bunch of files that will be useful for the app to run to the `deploy` folder. We change directory to the `deploy` folder, install the library specified in `requirements.txt` and expose port 5050. Finally, we run the `app.py` file with Python.

Now, we're only a few steps away from having a dockerized app. We simply need to build the Docker app:

```shell
docker build -t app-iris .
```

where `-t` provides the `app-iris` tag for the image we are building.

Finally, we run the application to see if everything works as expected:

```shell
docker run -p 5050:5050 app-iris .
```

If no error is obtained, we are ready for the last phase of local deployment.

#### Test our local Docker app

We now need to send a request to our running app in order to check that everything works fine locally. With the app running, going to `http://0.0.0.0:5050/` should display a simple page with "Hello, World!" on it.\
Now we can test the prediction function by running the following command in a terminal:

```sh
curl -X POST 0.0.0.0:5050/predict -H 'Content-Type: application/json' -d '[5.9,3.0,5.1,1.8]'
```

This command should return 0, 1 or 2, which is the class predicted by the algorithm. Beware that the number is printed right before the name of your machine within the terminal, therefore it is not very visible.

### Set up & configure an AWS EC2 instance

### Test the service
