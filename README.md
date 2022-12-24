<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

**Table of Contents**

- [Deploy ML Model](#deploy-ml-model)
  - [Project description](#project-description)
    - [Create a ML model](#create-a-ml-model)
    - [Make an API to make predictions](#make-an-api-to-make-predictions)
    - [Set up a virtual environment](#set-up-a-virtual-environment)
    - [Dockerize the application](#dockerize-the-application)
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

<!-- The next step in order to containerize our application is to  -->

### Set up & configure an AWS EC2 instance

### Test the service
