
import tensorflow as tf

def iter(x=x,a=a):
  f = a[0]*tf.pow(x,0)+a[1]*tf.pow(x,1)+a[2]*tf.pow(x,2)+a[3]*tf.pow(x,3)+a[4]*tf.pow(x,4)
  df = tf.gradients(f, x)
  ddf = tf.gradients(df, x)
  X = x - ( (2 * f * df) / ( (2 * tf.square(df)) - f * ddf ) )
  return X

with tf.Session() as sess:
  x = tf.placeholder(tf.float32)
  a = tf.placeholder(tf.float32)
  halley = iter(x,a)
  result = sess.run(halley,feed_dict={
    x: 2,
    a: [1,1,1,1,1]
  })
  print(result)