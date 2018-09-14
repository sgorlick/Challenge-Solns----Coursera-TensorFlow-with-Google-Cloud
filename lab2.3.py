import datalab.bigquery as bq
import tensorflow as tf
import numpy as np
import shutil
from google.datalab.ml import TensorBoard
from random import uniform
import pandas as pd
import math


#vol func
def vol(N) :
  df = pd.DataFrame([[np.round(uniform(0.5,2.0),1),np.round(uniform(0.5,2.0),1)] for n in range(N)]).rename(columns={0:'r',1:'h'})
  df['V'] = pd.Series([np.round(math.pi*df.iloc[i,1]*df.iloc[i,0]*df.iloc[i,1],1) for i in range(N)])
  return df

# Data ETL

FEATURES = ['r','h']
LABEL = ['V']

#train inputs
def make_input_fn(df, num_epochs):
  return tf.estimator.inputs.pandas_input_fn(
    x = df,
    y = df[LABEL],
    batch_size = 1000,
    num_epochs = num_epochs,
    shuffle = True,
    queue_capacity = 10000,
    num_threads = 1
  )

#pred inputs
def make_prediction_input_fn(df, num_epochs):
  return tf.estimator.inputs.pandas_input_fn(
    x = df,
    y = None,
    batch_size = 1000,
    num_epochs = num_epochs,
    shuffle = True,
    queue_capacity = 10000,
    num_threads = 1
  )

#INPUT_COLUMNS = [
#    tf.feature_column.numeric_column('r'),
#   tf.feature_column.numeric_column('h'),
#]

#feats
def make_feature_cols():
  input_columns = [tf.feature_column.numeric_column(k) for k in FEATURES]
  return input_columns

#def add_more_features(feats):
#    # Nothing to add (yet!)
#    return feats

feature_cols = make_feature_cols()#add_more_features(INPUT_COLUMNS)


# Defines the expected shape of the JSON feed that the model
# will receive once deployed behind a REST API in production.
def serving_input_fn():
    feature_placeholders = {
        'r' : tf.placeholder(tf.float32, [None]),
        'h' : tf.placeholder(tf.float32, [None]),
    }
    # You can transforma data here from the input format to the format expected by your model.
    features = feature_placeholders # no transformation needed
    return tf.estimator.export.ServingInputReceiver(features, feature_placeholders)


def train_and_evaluate(output_dir, num_train_steps):
    estimator = tf.estimator.DNNRegressor(hidden_units = [32, 8, 2], feature_columns = feature_cols, model_dir = output_dir)
    # tf.estimator.LinearRegressor(model_dir = output_dir,feature_columns = feature_cols)
  
    train_spec=tf.estimator.TrainSpec(
                       input_fn = make_input_fn(vol(100000), 100),
                       max_steps = num_train_steps)

    exporter = tf.estimator.LatestExporter('exporter', serving_input_fn)

    eval_spec=tf.estimator.EvalSpec(
                       input_fn = make_input_fn(vol(1000), 1),
                       steps = None,
                       start_delay_secs = 1, # start evaluating after N seconds
                       throttle_secs = 10,  # evaluate every N seconds
                       exporters = exporter)
    
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

OUTDIR = './voluminous'
TensorBoard().start(OUTDIR)


# Run training    
shutil.rmtree(OUTDIR, ignore_errors = True) # start fresh each time
train_and_evaluate(OUTDIR, num_train_steps = 100000)