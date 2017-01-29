from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pandas as pd
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

COLUMNS = ["street", "label", "classi", "neighbourhood", "garage", "latitude",
           "longitude", "lot_size", "population", "population_ration", "crimes", "suites"]
FEATURES = ["street", "classi", "neighbourhood", "garage", "latitude",
           "longitude", "lot_size", "population", "population_ration", "crimes", "suites"]
LABEL = "label"

training_set = pd.read_csv("train_file.csv", skipinitialspace=True,
                           skiprows=1, names=COLUMNS)
test_set = pd.read_csv("testa.csv", skipinitialspace=True,
                       skiprows=1, names=COLUMNS)
feature_cols = [tf.contrib.layers.real_valued_column(k)
                  for k in FEATURES]
regressor = tf.contrib.learn.DNNRegressor(
    feature_columns=feature_cols, hidden_units=[10, 10])

def input_fn(data_set):
  feature_cols = {k: tf.constant(data_set[k].values
                  for k in FEATURES}
  labels = tf.constant(data_set[LABEL].values)
  return feature_cols, labels
  
  regressor.fit(input_fn=lambda: input_fn(training_set), steps=5000)
