#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 09:32:13 2019

@author: amitk
"""

from flask import Flask, render_template, request
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression,RandomForestClassifier
from pyspark.sql.types import IntegerType
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# create or get the spark session
spark = SparkSession.builder.appName("myapp").getOrCreate()

def processDf(df):
    df = df.withColumn("label", df["wins"].cast(IntegerType()))
    df = df.withColumn("driverStandingsId1", df["driverStandingsId"].cast(IntegerType()))
    df = df.withColumn("raceId1", df["raceId"].cast(IntegerType()))
    df = df.withColumn("driverId1", df["driverId"].cast(IntegerType()))
    df = df.withColumn("points1", df["points"].cast(IntegerType()))
    
    df = df.withColumn("position1", df["position"].cast(IntegerType()))
    #df = df.withColumn("position", df["position"].cast(IntegerType()))
    df = df.withColumn("constructorId1", df["constructorId"].cast(IntegerType()))
    
    df = df.withColumn("number1", df["number"].cast(IntegerType()))
    
    df = df.withColumn("grid1", df["grid"].cast(IntegerType()))
    
    #df = df.withColumn("rank", df["rank"].cast(IntegerType()))
    
    #df = df.withColumn("fastestLap", df["fastestLap"].cast(IntegerType()))
    df = df.withColumn("fastestLapSpeed1", df["fastestLapSpeed"].cast(IntegerType()))
    df = df.withColumn("fastestLapTimeNum1", df["fastestLapTimeNum"].cast(IntegerType()))
    df = df.withColumn("statusId1", df["statusId"].cast(IntegerType()))
    df = df.withColumn("circuitId1", df["circuitId"].cast(IntegerType()))
    assembler = VectorAssembler(inputCols=["driverStandingsId1","raceId1","driverId1","points1","position1","constructorId1","number1","grid1","fastestLapSpeed1","fastestLapTimeNum1","statusId1","circuitId1" ], outputCol="features")
    df = assembler.transform(df)
    return df

def classifyRacer():
    train_df = spark.read.csv("/home/sunbeam/kshitij/formula-1-race-data-19502017 (1)/project_sample_1.csv", header=True)
    train_df = processDf(train_df)
    
    regressor = RandomForestClassifier()
    model = regressor.fit(train_df)
    
    # user input values
    test_df = spark.read.csv("/home/sunbeam/kshitij/formula-1-race-data-19502017 (1)/project_sample_test.csv", header=True)
    test_df = processDf(test_df)
    prediction = model.transform(test_df)
    
    return prediction.collect()[0]["prediction"]


app = Flask(__name__)

# routes
@app.route("/")
def serveRoot():
    return render_template("home_final.html")

@app.route("/process")
def processInput():
    driverStandingsId = request.args["driverStandingsId"]         #1
    raceId = request.args["raceId"]                               #2   
    driverId = request.args["driverId"]                           #3
    points = request.args["points"]                               #4
    position = request.args["position"]                           #5
    constructorId = request.args["constructorId"]                 #6
    constructorId = request.args["constructorId"]                 #7
    number = request.args["number"]                               #8
    grid = request.args["grid"]                                   #9
    fastestLapSpeed = request.args["fastestLapSpeed"]             #10
    fastestLapTimeNum = request.args["fastestLapTimeNum"]         #11
    statusId = request.args["statusId"]                           #12
    circuitId= request.args["circuitId"]                         #13
    
    file = open("/home/sunbeam/kshitij/formula-1-race-data-19502017 (1)/project_sample_test.csv", "w")
    print("driverStandingsId,raceId,driverId,points,position,constructorId,number,grid,fastestLapSpeed,fastestLapTimeNum,statusId,circuitId\n,{},{},{},{},{},{},{},{},{},{},{},{}".format(driverStandingsId,raceId,driverId,points,position,constructorId,number,grid,fastestLapSpeed,fastestLapTimeNum,statusId,circuitId))
    file.write("driverStandingsId,raceId,driverId,points,position,constructorId,number,grid,fastestLapSpeed,fastestLapTimeNum,statusId,circuitId,wins\n{},{},{},{},{},{},{},{},{},{},{},{}".format(driverStandingsId,raceId,driverId,points,position,constructorId,number,grid,fastestLapSpeed,fastestLapTimeNum,statusId,circuitId))
    file.close()
    
    result = classifyRacer()
    
    print("driverStandingsId:{} raceId:{} driverId:{} points:{} position:{} constructorId:{} number:{} grid:{} fastestLapSpeed:{} fastestLapTimeNum:{} statusId:{} circuitId:{}".format(driverStandingsId,raceId,driverId,points,position,constructorId,number,grid,fastestLapSpeed,fastestLapTimeNum,statusId,circuitId))
    
    return render_template("result.html", result=result)

app.run()


