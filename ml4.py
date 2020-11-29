#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 15:04:02 2019


@author: sunbeam
"""




from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression,RandomForestClassifier
from pyspark.mllib.tree import GradientBoostedTrees, GradientBoostedTreesModel
from pyspark.sql.types import IntegerType
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# create or get the spark session
spark = SparkSession.builder.appName("myapp").getOrCreate()

df = spark.read.csv("/home/sunbeam/kshitij/formula-1-race-data-19502017 (1)/project_sample_1.csv", header=True)
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

(train, test) = df.randomSplit([1.0, 2.0])
#train=train.rdd
#from pyspark.mllib.regression import LabeledPoint
regressor= RandomForestClassifier()
#regressor = LogisticRegression()
model = regressor.fit(train)

#train=df.rdd.map(lambda x: LabeledPoint(x[8], x[:10])).collect()
#model = GradientBoostedTrees.trainClassifier(train,categoricalFeaturesInfo={}, numIterations=3)

    
test_df = spark.read.csv("/home/sunbeam/kshitij/formula-1-race-data-19502017 (1)/project_sample_test.csv", header=True)
#
#
#test_df.drop("statusId")
#
#test_df.drop("circuitId")
test_df.printSchema()

test_df = test_df.withColumn("label", test_df["wins"].cast(IntegerType()))
test_df = test_df.withColumn("driverStandingsId1", test_df["driverStandingsId"].cast(IntegerType()))
test_df = test_df.withColumn("raceId1", test_df["raceId"].cast(IntegerType()))
test_df = test_df.withColumn("driverId1", test_df["driverId"].cast(IntegerType()))
test_df = test_df.withColumn("points1", test_df["points"].cast(IntegerType()))

test_df = test_df.withColumn("position1", test_df["position"].cast(IntegerType()))
#df = df.withColumn("position", df["position"].cast(IntegerType()))
test_df = test_df.withColumn("constructorId1", test_df["constructorId"].cast(IntegerType()))

test_df = test_df.withColumn("number1", test_df["number"].cast(IntegerType()))

test_df = test_df.withColumn("grid1", test_df["grid"].cast(IntegerType()))

#df = df.withColumn("rank", df["rank"].cast(IntegerType()))

#df = df.withColumn("fastestLap", df["fastestLap"].cast(IntegerType()))
test_df = test_df.withColumn("fastestLapSpeed1", test_df["fastestLapSpeed"].cast(IntegerType()))
test_df = test_df.withColumn("fastestLapTimeNum1", test_df["fastestLapTimeNum"].cast(IntegerType()))
test_df = test_df.withColumn("statusId1", test_df["statusId"].cast(IntegerType()))
test_df = test_df.withColumn("circuitId1", test_df["circuitId"].cast(IntegerType()))
assembler = VectorAssembler(inputCols=["driverStandingsId1","raceId1","driverId1","points1","position1","constructorId1","number1","grid1","fastestLapSpeed1","fastestLapTimeNum1","statusId1","circuitId1" ], outputCol="features")
test_df = assembler.transform(test_df)
prediction = model.transform(test_df)
prediction.select('label','rawPrediction', 'prediction', 'probability').show(40)
#evaluator = BinaryClassificationEvaluator()
#
#accuracy = (evaluator.evaluate(prediction))*100
# stop the sesion
spark.stop()
