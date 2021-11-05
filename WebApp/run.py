
"""

@author: loredanafattorini

Script runs Flask web application

"""
# imports

from flask import Flask
from flask import render_template, request

from datetime import datetime

#import findspark
#findspark.init() # find spark
#findspark.find()

from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.sql.types import *
from pyspark.ml.tuning import CrossValidatorModel


app = Flask(__name__)

# get spark session
spark = SparkSession.builder \
    .master("local") \
    .appName("Sparkify") \
    .getOrCreate()

# load model
model = CrossValidatorModel.load('./cvModel_gbt_weighted.mdl')

# index webpage receives user input for the model
@app.route('/')
@app.route('/index')
def index():
    
    # render web page
    return render_template('master.html')


# web page that handles user query and displays model results
@app.route('/go')
def go():
    
    # get parameters from the form
    gender = request.args.get('gender', '') 
    reg_date = request.args.get('reg_date', '') 
    avgSongs = request.args.get('avgSongs', 0)
    Thumbsup_proportion = request.args.get('Thumbsup_proportion', 0)
    num_add_friend = request.args.get('num_add_friend', 0)
    last_level = request.args.get('level', '')
    avgrolladverts = request.args.get('avgrolladverts', 0)
    last_state = request.args.get('location', '')
    
    
    # calculate number of days since the 1st event for the user
    days_registered = (datetime.now() - datetime.strptime(reg_date, '%Y-%m-%d')).days
    
    # encode gender values
    if gender == 'male':
        gender = 0
    else:
        gender = 1

    # encode level values
    if last_level == 'paid':
        last_level = 1
    else:
        last_level = 0
   
    # get spark context
    sc = SparkContext.getOrCreate()
    
    # create spark dataframe to predict customer churn using the model
    df = sc.parallelize([[gender, days_registered, last_state, avgSongs, last_level, Thumbsup_proportion, num_add_friend, avgrolladverts]]).\
    toDF(["gender", "days_registered", "last_state", "avg_songs_per_day", "last_level", "Thumbsup_proportion", "num_add_friend", "avg_roll_adv_per_day"])
    
    # set correct data types
    df = df.withColumn("days_registered", df["days_registered"].cast(IntegerType()))
    df = df.withColumn("avg_songs_per_day", df["avg_songs_per_day"].cast(DoubleType()))
    df = df.withColumn("Thumbsup_proportion", df["Thumbsup_proportion"].cast(DoubleType()))
    df = df.withColumn("num_add_friend", df["num_add_friend"].cast(IntegerType()))
    df = df.withColumn("avg_roll_adv_per_day", df["avg_roll_adv_per_day"].cast(DoubleType()))
 
    # predict using the model
    pred = model.transform(df)
    
    if pred.count() == 0:
        # if model failed to predict churn then return -1
        prediction = -1
    else:
        # get prediction (1 = churn, 0 = stay)
        prediction = pred.select(pred.prediction).collect()[0][0]
    
    # print out prediction to the app console
    print("Prediction for the customer is {prediction}.".format(prediction = prediction))
    
    # render the go.html passing prediction resuls
    return render_template(
        'go.html',
        result = prediction
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()