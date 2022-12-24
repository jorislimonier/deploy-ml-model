<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

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

The ML model is a basic `LogisticRegression` algorithm from `scikit-learn` on the Iris dataset. The algorithm and the data don't matter much in this project as this is the part I am most comfortable with. On the deployment part, however, I am way more uncomfortable. I never learnt such things as dockerizing, setting up instances or using the cloud. Here is the code to classify

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

### Make an API to make predictions

### Set up a virtual environment

### Dockerize the application

### Set up & configure an AWS EC2 instance

### Test the service
