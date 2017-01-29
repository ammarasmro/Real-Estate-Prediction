from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile


import pandas as pd
import tensorflow as tf


COLUMNS = ["street", "label", "classi", "neighbourhood", "garage", "latitude",
           "longitude", "lot_size", "population", "population_ratio", "crimes", "suites"]
LABEL_COLUMN = "label"
CATEGORICAL_COLUMNS = ["neighbourhood", "street", "longitude", "latitude",
                       "classi", "garage"]
CONTINUOUS_COLUMNS = ["lot_size", "population", "population_ratio", "crimes", "suites"]


def build_estimator(model_dir, model_type):
  """Build an estimator."""
  # Sparse base columns.
  #suites = tf.contrib.layers.sparse_column_with_keys(column_name="suites", keys=["Yes", "No"])
  classi = tf.contrib.layers.sparse_column_with_keys(column_name="classi",keys=["Residential", "Non Residential", "Other Residential", "Farmland"])
  garage = tf.contrib.layers.sparse_column_with_keys(column_name="garage", keys=["Y", "N"])
  #longitude = tf.contrib.layers.sparse_column_with_hash_bucket("longitude", #hash_bucket_size=2500)
  #latitude = tf.contrib.layers.sparse_column_with_hash_bucket("latitude", #hash_bucket_size=2500)
  street = tf.contrib.layers.sparse_column_with_hash_bucket("street", hash_bucket_size=2590)
  neighbourhood = tf.contrib.layers.sparse_column_with_hash_bucket("neighbourhood", hash_bucket_size=390)

  # Continuous base columns.
  suites = tf.contrib.layers.real_valued_column("suites")
  lot_size = tf.contrib.layers.real_valued_column("lot_size")
  population = tf.contrib.layers.real_valued_column("population")
  population_ratio = tf.contrib.layers.real_valued_column("population_ratio")
  crimes = tf.contrib.layers.real_valued_column("crimes")

 # Transformations.
  suites_buckets = tf.contrib.layers.bucketized_column(suites,
                                                    boundaries=[
                                                        0,1
                                                    ])

  # Wide columns and deep columns.
  wide_columns = [
  		neighbourhood, street, classi, garage,
  		tf.contrib.layers.crossed_column([classi, garage], hash_bucket_size=int(1e4)),
  		tf.contrib.layers.crossed_column([neighbourhood, street], hash_bucket_size=int(1e6))]
  deep_columns = [
  	tf.contrib.layers.embedding_column(neighbourhood, dimension=8),
  	tf.contrib.layers.embedding_column(street, dimension=8),
  	tf.contrib.layers.embedding_column(classi, dimension=8),
  	tf.contrib.layers.embedding_column(garage, dimension=8),
        suites_buckets,
        lot_size, population, population_ratio, crimes]

  if model_type == "wide":
    m = tf.contrib.learn.LinearClassifier(model_dir=model_dir,
                                          feature_columns=wide_columns)
  elif model_type == "deep":
    m = tf.contrib.learn.DNNClassifier(model_dir=model_dir,
                                       feature_columns=deep_columns,
                                       hidden_units=[100, 50])
  else:
    m = tf.contrib.learn.DNNLinearCombinedClassifier(
        model_dir=model_dir,
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=[100, 50])
  return m


def input_fn(df):
  """Input builder function."""
  # Creates a dictionary mapping from each continuous feature column name (k) to
  # the values of that column stored in a constant Tensor.
  continuous_cols = {k: tf.constant(df[k].values) for k in CONTINUOUS_COLUMNS}
  # Creates a dictionary mapping from each categorical feature column name (k)
  # to the values of that column stored in a tf.SparseTensor.
  categorical_cols = {
      k: tf.SparseTensor(
          indices=[[i, 0] for i in range(df[k].size)],
          values=df[k].values,
          shape=[df[k].size, 1])
      for k in CATEGORICAL_COLUMNS}
  # Merges the two dictionaries into one.
  feature_cols = dict(continuous_cols)
  feature_cols.update(categorical_cols)
  # Converts the label column into a constant Tensor.
  label = tf.constant(df[LABEL_COLUMN].values)
  # Returns the feature columns and the label.
  return feature_cols, label


def train_and_eval(model_dir, model_type, train_steps, train_data, test_data):
  """Train and evaluate the model."""
  df_train = pd.read_csv("train_filec.csv", skipinitialspace=True,
          skiprows=1, names=COLUMNS)
  df_test = pd.read_csv("testac.csv", skipinitialspace=True,
          skiprows=1, names=COLUMNS)

  # remove NaN elements
  df_train = df_train.dropna(how='any', axis=0)
  df_test = df_test.dropna(how='any', axis=0)

  model_dir = tempfile.mkdtemp() if not model_dir else model_dir
  print("model directory = %s" % model_dir)

  m = build_estimator(model_dir, model_type)
  m.fit(input_fn=lambda: input_fn(df_train), steps=train_steps)
  results = m.evaluate(input_fn=lambda: input_fn(df_test), steps=1)
  for key in sorted(results):
    print("%s: %s" % (key, results[key]))


FLAGS = None


def main(_):
  train_and_eval(FLAGS.model_dir, FLAGS.model_type, FLAGS.train_steps,
                 FLAGS.train_data, FLAGS.test_data)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  parser.add_argument(
      "--model_dir",
      type=str,
      default="",
      help="Base directory for output models."
  )
  parser.add_argument(
      "--model_type",
      type=str,
      default="wide_n_deep",
      help="Valid model types: {'wide', 'deep', 'wide_n_deep'}."
  )
  parser.add_argument(
      "--train_steps",
      type=int,
      default=200,
      help="Number of training steps."
  )
  parser.add_argument(
      "--train_data",
      type=str,
      default="",
      help="Path to the training data."
  )
  parser.add_argument(
      "--test_data",
      type=str,
      default="",
      help="Path to the test data."
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)