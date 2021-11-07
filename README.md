# Capstone-Project-Sparkify-Churn-Prediction-Udacity

## Project Overview

**Sparkify is a popular (not real!) music service** similar to Spotify or Pandora with a **subscription-based business model**. Each user can listen to their favorite music every day either through the **free-tier** or by using a **subscription plan where she pays a fixed monthly fee**. Users can **upgrade, downgrade, or cancel the service at any time**, so it's critical to be sure users love the service.

**Every time a user interacts with the service it generates (synthetic) data**. Each event (e.g., song played, logout, like, downgrade, ...) is recorded with the corresponding timestamp. All of this information holds the key to keeping users happy and businesses thriving.

## Problem Statement

The **goal** of the problem is to help Sparkify in answering the following question:<br>

**Which users are at risk of churn, i.e. downgrade from premium service to free-tier plan or cancellation of service altogether?**

By **identifying** these **users before they abandon the service**, Sparkify can **proactively engage with them** by offering some discounts and/or incentives, which can **save a lot of time and money in acquiring new users**.

We explore several **binary classification models**, where the target variable is 0 if the customer did not churn and 1 otherwise.

**To identify the best model** that helps us in predicting customer churn, we complete the following tasks: <br>

- **Analyze and preprocess the data**
- Use **Machine learning pipelines**
- **Train each classifier (Logistic regression, Random forest, Gradient-boosted tree) on the training set**
- **Test each model on the test set**
- **Hyperparameter tuning and cross-validation**
- **Model evaluation**

Finally, we build a **web app** using `Flask` on the back-end and `Bootstrap` on the front-end.
In the web app, you can **enter the information about the user** and get **her classification** based on the model prediction.

### Note on metrics

Typically, when we look at the results of a classification model, we focus on the correct predictions of all the predictions made by the model, i.e., the accuracy of the model.<br>
In our case, when doing a **churn analysis with the goal of predicting customers who churned**, we are particularly interested in **having a lot of true positives**. However, since there are many customers who have not churned (**highly unbalanced datasets**), **the higher the number of true negatives the higher the accuracy**, which can be misleading. Thus, a **better measure of model performance in this case is the F1 score**, which is the harmonic mean of Precision ('out of all customers who were labeled as "churned," how many did we correctly label as such?') and Recall ('out of all customers who were labeled as "churned," how many actually churned?').

## Data

Input data used that contains Sparkify music events:<br>
- Mini-dataset file (128MB) `mini_sparkify_event_data.json`
- Full dataset available (12GB) `s3n://udacity-dsnd/sparkify/sparkify_event_data.json`

The **full dataset contains a large amount of data**, which cannot be processed on a single machine. This is why we **first performed data analysis and model building with `Spark` (Pyspark and SparkML libraries) on the small subset** and then, extended the analysis to the **entire dataset** using **AWS EMR clusters**.

## Installation

* Python3
* Pyspark
* Numpy, Pandas
* Data Visualization: Plotly, Seaborn, Matplolib
* Web App: Flask 


## File Description

```bash
- WebApp
  - templates
    - master.html  # main page of web app
    - go.html  # classification result page of web app
  - static
    - githublogo2.png  # github logo used in the main page
    - churn_image.png # downloaded from (https://www.shutterstock.com)
  - run.py  # Flask script that runs app
  - cvModel_gbt_weighted.mdl # ML model chosen to make prediction
    - bestModel
    - estimator
    - evaluator
    - metadata
  - demo
    - demo_webapp.gif # animation with prediction of churn
  - Sparkify.ipynb # Jupyter notebook that contains detailed data analysis and model building with Spark (Pyspark and SparkML libraries) run on the small subset
  - Sparkify_AWS-110321.ipynb # Jupyter notebook that contains the analysis run using the full available dataset on AWS EMR
```
## Instructions for running the web app:

1. Run the following command **in the app's directory** to run the web app. `python run.py`
2. Go to http://0.0.0.0:3001/ to see the web app.

## Results

The **best performing model** according to the **F1 score** is the **gradient-boosted tree classifier** where **label weights** were included in the model **to account for the highly imbalanced dataset**. In fact, **the number of active users is more than three times the number of unsubscribed users**.  <br>
We use this **machine learning model trained on the full dataset on AWS EMR** to make predictions about customers on the **web application**. 

### Demo of the webapp

![](WebApp/demo/demo_webapp.gif)


## Challenges and future improvements

To improve the performance of the most promising models we can: <br>

- Spend more time engineering features by trying to think of more meaningful features (e.g., sequence of events that precede user's churn).
- Do more tuning of hyperparameters.
- Perform stacking of models.
- Find a better approach to dealing with strongly unbalanced data.


## Licensing, Authors, Acknowledgements.

This project has been completed as part of the Data Science Nanodegree on [Udacity](www.udacity.com). The data was provided by Udacity.
