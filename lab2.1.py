#Prompt:
#Create a neural network that is capable of finding the volume of a cylinder given the radius of its base (r) and its height (h). Assume that the radius and height of the cylinder are both #in the range 0.5 to 2.0. 

#my sol'n
from random import uniform
import numpy as np
import pandas as pd
import tensorflow as tf
import math

#vol func
def vol(N) :
  df = pd.DataFrame([[uniform(0.5,2.0),uniform(0.5,2.0)] for n in range(N)]).rename(columns={0:'r',1:'h'})
  df['V'] = pd.Series([math.pi*df.iloc[i,1]*df.iloc[i,0]*df.iloc[i,1] for i in range(N)])
  return df

# Data ETL

FEATURES = ['r','h']
LABEL = ['V']

df_train = vol(10000)
df_valid = vol(1000)

#train inputs
def make_input_fn(df, num_epochs):
  return tf.estimator.inputs.pandas_input_fn(
    x = df,
    y = df[LABEL],
    batch_size = 128,
    num_epochs = num_epochs,
    shuffle = True,
    queue_capacity = 1000,
    num_threads = 1
  )

#pred inputs
def make_prediction_input_fn(df, num_epochs):
  return tf.estimator.inputs.pandas_input_fn(
    x = df,
    y = None,
    batch_size = 128,
    num_epochs = num_epochs,
    shuffle = True,
    queue_capacity = 1000,
    num_threads = 1
  )

#feats
def make_feature_cols():
  input_columns = [tf.feature_column.numeric_column(k) for k in FEATURES]
  return input_columns

#model DNN
OUTDIR = 'voluminous'
tf.logging.set_verbosity(tf.logging.INFO)
shutil.rmtree(OUTDIR, ignore_errors = True) # start fresh each time
model = tf.estimator.DNNRegressor(hidden_units = [32, 8, 2],
      feature_columns = make_feature_cols(), model_dir = OUTDIR)
model.train(input_fn = make_input_fn(df_train, num_epochs = 100));

#eval
def print_rmse(model, name, df):
  metrics = model.evaluate(input_fn = make_input_fn(df, 1))
  print('RMSE on {} dataset = {}'.format(name, np.sqrt(metrics['average_loss'])))

#predict
predictions = model.predict(input_fn = make_prediction_input_fn(df_valid, 1))
for i in range(5):
  print(next(predictions))