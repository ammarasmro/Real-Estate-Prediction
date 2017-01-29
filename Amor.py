from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile
import sklearn

import tensorflow as tf

# Categorical base columns.
suites = tf.contrib.layers.sparse_column_with_keys(column_name="suites", keys=["1", "0"])
classi = tf.contrib.layers.sparse_column_with_keys(column_name="classi", keys=["Residential", "Non Residential", "Other Residential", "Farmland"])
garage = tf.contrib.layers.sparse_column_with_keys(column_name="garage", keys=["Y", "N"])
longitude = tf.contrib.layers.sparse_column_with_hash_bucket("longitude", hash_bucket_size=2500)
latitude = tf.contrib.layers.sparse_column_with_hash_bucket("latitude", hash_bucket_size=2500)
street = tf.contrib.layers.sparse_column_with_hash_bucket("street", hash_bucket_size=2590)
neighbourhood = tf.contrib.layers.sparse_column_with_hash_bucket("neighbourhood", hash_bucket_size=390)

# Continuous base columns.
longitude_num = tf.contrib.layers.real_valued_column("longitude_num")
latitude_num = tf.contrib.layers.real_valued_column("latitude_num")
lot_size = tf.contrib.layers.real_valued_column("lot_size")
population = tf.contrib.layers.real_valued_column("population")
population_ratio = tf.contrib.layers.real_valued_column("population_ratio")
crimes = tf.contrib.layers.real_valued_column("crimes")

wide_columns = [
  neighbourhood, street, longitude, latitude, suites, classi, garage,
  tf.contrib.layers.crossed_column([neighbourhood, street], hash_bucket_size=int(1e6)),
  tf.contrib.layers.crossed_column([classi, suites], hash_bucket_size=int(1e4)),
  tf.contrib.layers.crossed_column([classi, garage], hash_bucket_size=int(1e4)),
  tf.contrib.layers.crossed_column([neighbourhood, street, longitude, latitude], hash_bucket_size=int(1e8))]

deep_columns = [
  tf.contrib.layers.embedding_column(neighbourhood, dimension=8),
  tf.contrib.layers.embedding_column(street, dimension=8),
  tf.contrib.layers.embedding_column(longitude, dimension=8),
  tf.contrib.layers.embedding_column(latitude, dimension=8),
  tf.contrib.layers.embedding_column(suites, dimension=8),
  tf.contrib.layers.embedding_column(classi, dimension=8),
  tf.contrib.layers.embedding_column(garage, dimension=8),
  longitude_num, latitude_num, lot_size, population, population_ratio, crimes]

import tempfile
model_dir = tempfile.mkdtemp()
m = tf.contrib.learn.DNNLinearCombinedClassifier(
    model_dir=model_dir,
    linear_feature_columns=wide_columns,
    dnn_feature_columns=deep_columns,
    dnn_hidden_units=[100, 50])

import pandas as pd
import urllib

# Define the column names for the data sets.
COLUMNS = ["neighbourhood", "street", "longitude", "latitude", "suites", "classi", "garage", "lot_size", "population", "population_ratio", "crimes"]
LABEL_COLUMN = 'label'
CATEGORICAL_COLUMNS = ["neighbourhood", "street", "longitude", "latitude",
                       "suites", "classi", "garage"]
CONTINUOUS_COLUMNS = ["lot_size", "population", "population_ratio", "crimes"]



import numpy as np

# Read the training and test data sets into Pandas dataframe.
df_train = pd.read_csv(“train_file”, names=COLUMNS, skipinitialspace=True)
df_test = pd.read_csv(“testa”, names=COLUMNS, skipinitialspace=True, skiprows=0)

def input_fn(df):
  # Creates a dictionary mapping from each continuous feature column name (k) to
  # the values of that column stored in a constant Tensor.
  continuous_cols = {k: tf.constant(df[k].values)
                     for k in CONTINUOUS_COLUMNS}
  # Creates a dictionary mapping from each categorical feature column name (k)
  # to the values of that column stored in a tf.SparseTensor.
  categorical_cols = {k: tf.SparseTensor(
      indices=[[i, 0] for i in range(df[k].size)],
      values=df[k].values,
      shape=[df[k].size, 1])
                      for k in CATEGORICAL_COLUMNS}
  # Merges the two dictionaries into one.
  feature_cols = dict(continuous_cols.items() + categorical_cols.items())
  # Converts the label column into a constant Tensor.
  label = tf.constant(df[LABEL_COLUMN].values)
  # Returns the feature columns and the label.
  return feature_cols, label

def train_input_fn():
  return input_fn(df_train)

def eval_input_fn():
  return input_fn(df_test)

m.fit(input_fn=train_input_fn, steps=200)
results = m.evaluate(input_fn=eval_input_fn, steps=1)
for key in sorted(results):
    print (key, results[key])                                       