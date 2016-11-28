from __future__ import print_function, absolute_import, division

import tensorflow as tf
import numpy as np
import argparse
import time
import sys
from os.path import isdir, join
import shutil
from six.moves import cPickle as pickle
from six.moves import xrange

__author__ = 'Xin Chen'
__email__ = "chenxin13@mails.tsinghua.edu.cn"


PICKLE_FILE = 'notMNIST.pickle'
IMAGE_SIZE = 28
NUM_LABELS = 10
NUM_CHANNELS = 1 # grayscale
PATCH_SIZE = 5
SEED = 62141084

FLAGS = None


def reformat(dataset, labels):
  """
  Reformat the dataset to 4D numpy array [index, x, y, channel].
  Reformat the labels to float32 1-hot encoding vectors.

  Args:
    dataset: a 2D numpy array, [index, feature_vector]
    labels: a 1D numpy array, [class_number, ...]

  Returns:
    dataset: a 4D numpy array, [index, x, y, channel]
    labels: a 2D numpy array, [index, 1-hot-encodings]

  """
  dataset = dataset.reshape(
    (-1, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)).astype(np.float32)
  labels = (np.arange(NUM_LABELS) == labels[:, None]).astype(np.float32)
  return dataset, labels


def load_dataset(verbose=True):
  """
  Load the notMNIST datasets from the pickle file and return reformatted
  datasets and labels.

  Args:
    verbose: a boolean.

  """
  if verbose:
    print("-> Load the notMNIST dataset ...")
    print("")
  with open(PICKLE_FILE, 'rb') as f:
    save = pickle.load(f)
    train_dataset = save['train_dataset']
    train_labels = save['train_labels']
    valid_dataset = save['valid_dataset']
    valid_labels = save['valid_labels']
    test_dataset = save['test_dataset']
    test_labels = save['test_labels']
    del save  # hint to help gc free up memory
  test_dataset, test_labels = reformat(test_dataset, test_labels)
  train_dataset, train_labels = reformat(train_dataset, train_labels)
  valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
  if verbose:
    print("Training set     :", train_dataset.shape, train_labels.shape)
    print("Validation set   :", valid_dataset.shape, valid_labels.shape)
    print("Test set         :", test_dataset.shape, test_labels.shape)
    print("")
  return train_dataset, train_labels, \
         valid_dataset, valid_labels, \
         test_dataset, test_labels


def print_activations(t):
  print("%-15s %s" % (t.op.name, t.get_shape().as_list()))


def squashing(x, slopes, amplitude=1.7159, name=None):
  """
  A scaled hyperbolic tangent tensor.

  Specifically, `y = A * tanh(slope * x)`.

  Args:
    x: a 4D data tensor in NHWC format, [index, x, y, depth].
    slopes: a 1D tensor representing the slope of each channel, [depth].
    amplitude: a float value representing the amplitude of the function.
    name: the name of this tensor.

  Returns:
    y: a 4D data tensor.

  """
  return amplitude * tf.tanh(tf.mul(slopes, x), name=name)


def tf_pairwise_square_distances(xx, yy):
  """
  Computes the squared Euclidean distance between all pairs x in xx, y in yy.

  Args:
    xx, yy: Tensors, [nx, m], [ny, m]

  Returns:
    dist: a Tensor, [nx, ny]

  """
  c = -2 * tf.matmul(xx, tf.transpose(yy))
  nx = tf.reduce_sum(tf.square(xx), 1, keep_dims=True)
  ny = tf.reduce_sum(tf.square(yy), 1, keep_dims=True)
  return (c + tf.transpose(ny)) + nx


def euclidean_rbf(x, w):
  """
  A radial basis function kernel with Euclidean distances.

  Specifically, `y_i = \sum_j^n { (x_j - w_{ij})^2 }`

  Args:
    x: a Tensor, [index, n]
    w: a Tensor, [m, n]

  Returns:
    y: the output Tensor, [index, m]

  """
  return tf_pairwise_square_distances(x, w)


def accuracy(predictions, labels):
  """ Return the prediction accuracy. """
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])


def inference_lenet5xc(dataset, for_training=False, conv_relu=False):
  """
  Build the LeNet5xc model, which is based on the origin LeNet5 model.

  Args:
    dataset: a 4D dataset tensor, [index, x, y, channels]
    for_training: a bool indicating this model should be used for training or
      validing / testing.

  Returns:
    model: the last Tensor in LeNet5xc.
    parameters: the weights and biases of the full-connected componenets.

  """

  if for_training:
    print("-> Inference the LeNet5xc model ...")
    print("")

  parameters = []

  # C1: a Conv2D layer, 28 x 28 x 6
  with tf.name_scope('Conv1') as scope:
    kernel = tf.Variable(tf.truncated_normal([5, 5, 1, 6], stddev=0.1),
                         dtype=tf.float32, name="weights")
    conv = tf.nn.conv2d(dataset, kernel, [1, 1, 1, 1], padding="SAME")
    biases = tf.Variable(tf.zeros([6]), trainable=True, dtype=tf.float32,
                         name="biases")
    conv1 = tf.nn.bias_add(conv, biases)
    if conv_relu:
      conv1 = tf.nn.relu(conv1, name=scope)
    if for_training:
      print_activations(conv1)

  # S2: a down-sampling layer, scaled average pooing with biases added.
  # 28 x 28 x 6 --> 14 x 14 x 6
  with tf.name_scope('AvgPool2') as scope:
    pool2 = tf.nn.avg_pool(conv1,
                           ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding="SAME",
                           name="pool2")
    biases = tf.Variable(tf.zeros([6]), trainable=True, dtype=tf.float32,
                         name="biases")
    bias = tf.nn.bias_add(pool2, biases)
    slopes = tf.Variable(tf.ones([6]), trainable=True, dtype=tf.float32,
                         name="slopes")
    sample2 = squashing(bias, slopes, name=scope)
    if for_training:
      print_activations(sample2)

  # C3: a Conv2D layer, 10 x 10 x 16
  with tf.name_scope('Conv3') as scope:
    kernel = tf.Variable(tf.truncated_normal([5, 5, 6, 16], stddev=0.1),
                         name="weights")
    conv = tf.nn.conv2d(sample2, kernel, [1, 1, 1, 1], padding="VALID")
    biases = tf.Variable(tf.zeros([16]), trainable=True, dtype=tf.float32,
                         name="biases")
    conv3 = tf.nn.bias_add(conv, biases)
    if conv_relu:
      conv3 = tf.nn.relu(conv3, name=scope)
    if for_training:
      print_activations(conv3)

  # S4: a down-sampling layer, scaled average pooing with biases added.
  # 10 x 10 x 16 --> 5 x 5 x 16
  with tf.name_scope('AvgPool4') as scope:
    pool4 = tf.nn.avg_pool(conv3,
                           ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding="SAME",
                           name="pool2")
    biases = tf.Variable(tf.zeros([16]), trainable=True, dtype=tf.float32,
                         name="biases")
    bias = tf.nn.bias_add(pool4, biases)
    slopes = tf.Variable(tf.ones([16]), trainable=True, dtype=tf.float32,
                         name="slopes")
    sample4 = squashing(bias, slopes, name=scope)
    if for_training:
      print_activations(sample4)

  # C5: a Conv2D layer, 1 x 1 x 120
  with tf.name_scope('Conv5') as scope:
    kernel = tf.Variable(tf.truncated_normal([5, 5, 16, 120], stddev=0.1),
                         name="weights")
    conv = tf.nn.conv2d(sample4, kernel, [1, 1, 1, 1], padding="VALID")
    biases = tf.Variable(tf.zeros([120]), trainable=True, dtype=tf.float32,
                         name="biases")
    conv5 = tf.nn.bias_add(conv, biases)
    if conv_relu:
      conv5 = tf.nn.relu(conv5, name=scope)
    if for_training:
      print_activations(conv5)

  # F6: a full-connected layer with 84 nodes.
  with tf.name_scope('FullConn6') as scope:
    weights = tf.Variable(tf.truncated_normal([120, 84], stddev=0.1),
                          name="weights")
    fc = tf.matmul(tf.reshape(conv5, (-1, 120)), weights)
    biases = tf.Variable(tf.zeros([84]), trainable=True, dtype=tf.float32,
                         name="biases")
    bias = tf.nn.bias_add(fc, biases)
    fc6 = tf.sigmoid(bias, name=scope)
    if for_training:
      print_activations(fc6)
    parameters.extend([weights, biases])

  # F7: a full-connected layer with 10 nodes.
  with tf.name_scope('Output'):
    weights = tf.Variable(tf.truncated_normal([84, 10], stddev=0.1),
                          name="centers")
    fc = tf.matmul(fc6, weights)
    biases = tf.Variable(tf.zeros([10]), trainable=True, dtype=tf.float32,
                         name="biases")
    output = tf.nn.bias_add(fc, biases)
    if for_training:
      print_activations(output)
    parameters.extend([weights, biases])

  if for_training:
    output = tf.nn.dropout(output, 0.5, seed=SEED)
    print("")

  return output, parameters


def run_tensorflow_lenet5xc(batch_size=16, num_epochs=10, eval_frequency=100,
                            eval_batch_size=500, log_dir="./log"):
  """ Run the tensorflow-based LeNet5xc on notMNIST. """
  graph = tf.Graph()
  sess = tf.Session(graph=graph)

  with sess:
    # Load rformatted datasets and labels from the cPickle file.
    train_dataset, train_labels, \
    valid_dataset, valid_labels, \
    test_dataset, test_labels = load_dataset(verbose=True)

    # Transform these datasets and labels to Tensors.
    input_dataset = tf.placeholder(tf.float32, [batch_size, 28, 28, 1])
    input_labels = tf.placeholder(tf.float32, [batch_size, 10])
    train_size = train_dataset.shape[0]
    eval_dataset = tf.placeholder(tf.float32, [eval_batch_size, 28, 28, 1])

    # Inference the LeNet5 neural netwoork model.
    logits, parameters = inference_lenet5xc(input_dataset, for_training=True)

    # Compute the regularized loss
    with tf.name_scope("cross_entropy"):
      loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits, input_labels))
      tf.summary.scalar("loss", loss)

      with tf.name_scope("regularizers"):
        regularizers = 5e-4 * tf.nn.l2_loss(parameters[0])
        for params in parameters[1:]:
          regularizers += 5e-4 * tf.nn.l2_loss(params)
      tf.summary.scalar("regularizers", regularizers)

      with tf.name_scope("total"):
        cross_entropy = loss + regularizers
      tf.summary.scalar("cross entropy", cross_entropy)

    # Setup the training rate decay
    batch = tf.Variable(0, dtype=tf.int32)
    learning_rate = tf.train.exponential_decay(
      0.01, tf.mul(batch, batch_size), train_size, 0.95, staircase=True)

    # Use a simple momentum optimizer
    trainer = tf.train.MomentumOptimizer(learning_rate, 0.9)
    optimizer = trainer.minimize(cross_entropy, global_step=batch)

    # Setup predictions for the current training minibatch
    train_predictions = tf.nn.softmax(logits)

    # Setup predictions for validation and final test.
    test_predictions = tf.nn.softmax(inference_lenet5xc(eval_dataset)[0])
    valid_predictions = tf.nn.softmax(inference_lenet5xc(valid_dataset)[0])

    # Small utility function to evaluate a dataset by feeding batches of data to
    # {eval_data} and pulling the results from {eval_predictions}.
    # Saves memory and enables this to run on smaller GPUs.
    def eval_in_batches(data, session, eval_prediction):
      """Get all predictions for a dataset by running it in small batches."""
      size = data.shape[0]
      if size < eval_batch_size:
        raise ValueError("batch size for evals larger than dataset: %d" % size)
      results = np.ndarray(shape=(size, NUM_LABELS), dtype=np.float32)
      for begin in xrange(0, size, eval_batch_size):
        end = begin + eval_batch_size
        if end <= size:
          results[begin:end, :] = session.run(
            eval_prediction,
            feed_dict={eval_dataset: data[begin:end, ...]})
        else:
          batch_predictions = session.run(
            eval_prediction,
            feed_dict={eval_dataset: data[-eval_batch_size:, ...]})
          results[begin:, :] = batch_predictions[begin - size:, :]
      return results

    # Record the training accuracy and validation accuracy.
    with tf.name_scope("accuracy"):
      with tf.name_scope("correct_prediction"):
        train_corrects = tf.equal(
          tf.argmax(train_predictions, 1), tf.argmax(input_labels, 1))
        valid_corrects = tf.equal(
          tf.argmax(valid_predictions, 1), tf.argmax(valid_labels, 1))
      with tf.name_scope("accuracy"):
        train_accuracy = tf.reduce_mean(tf.cast(train_corrects, tf.float32))
        valid_accuracy = tf.reduce_mean(tf.cast(valid_corrects, tf.float32))
      tf.summary.scalar("train_accuracy", train_accuracy)
      tf.summary.scalar("valid_accuracy", valid_accuracy)

    # Start running operations on the Graph.
    print("-> Start training LeNet5xc ...")
    print("")
    tic = time.time()

    # Merge all the summaries and write them out to ./logs (by default).
    merged = tf.merge_all_summaries()
    train_writer = tf.train.SummaryWriter(join(log_dir, "train"), sess.graph)
    valid_writer = tf.train.SummaryWriter(join(log_dir, "valid"), sess.graph)

    # Build an initialization operation.
    tf.initialize_all_variables().run()
    print("Initialized!")
    print("")

    # Loop through training steps.
    for step in xrange(int(num_epochs * train_size) // batch_size):
      # Compute the offset of the current minibatch in the data.
      # The dataset was already shuffled in assignment 1 so we do not need to
      # randomize it.
      offset = (step * batch_size) % (train_size - batch_size)
      batch_dataset = train_dataset[offset: (offset + batch_size), ...]
      batch_labels = train_labels[offset: (offset + batch_size), ...]
      # Build the feed dict to feed previous defined placeholders.
      feed_dict = {input_dataset: batch_dataset, input_labels: batch_labels}
      # Run the optimization session.
      sess.run([optimizer], feed_dict=feed_dict)
      # Save the training accuracy every 100 steps.
      if step % 100 == 0:
        summary, train_acc = sess.run(
          [merged, train_accuracy], feed_dict=feed_dict)
        train_writer.add_summary(summary, step)
        # Every `eval_frequency` steps we shall take several extra operations,
        # including printing the validation accuracy and updating the learning
        # rate.
        if step % eval_frequency == 0:
          l, lr, valid_acc = sess.run(
            [cross_entropy, learning_rate, valid_accuracy], feed_dict=feed_dict)
          valid_writer.add_summary(summary, step)
          elapsed_time = time.time() - tic
          tic = time.time()
          print("Step %6d (epoch %5.2f), %.3f s" % (
            step, float(step) * batch_size / train_size, elapsed_time))
          print("Minibatch loss: %.3f, learning rate: %.6f" % (l, lr))
          print("Minibatch accuracy: %.1f%%" % (train_acc * 100.0))
          print("Validation accuracy: %.1f%%" % (valid_acc * 100.0))
          print("")
          sys.stdout.flush()
    # Close the writers
    train_writer.close()
    valid_writer.close()
    # Finally the training is completed. Now let's see if this model performs
    # well on notMNIST.
    test_accuracy = accuracy(
      eval_in_batches(test_dataset, sess, test_predictions), test_labels)
    print("Test accuracy: %.1f%%" % test_accuracy)
    # At last, do not forget to save the model!
    tf.train.Saver({"LeNet5xc": logits}).save(sess, "./lenet5xc.ckpt")


def main(_):
  if tf.gfile.Exists(FLAGS.log_dir):
    tf.gfile.DeleteRecursively(FLAGS.log_dir)
  tf.gfile.MakeDirs(FLAGS.log_dir)
  run_tensorflow_lenet5xc(batch_size=FLAGS.batch_size,
                          num_epochs=FLAGS.num_epochs,
                          eval_batch_size=FLAGS.eval_batch_size,
                          eval_frequency=FLAGS.eval_frequency,
                          log_dir=FLAGS.log_dir)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--batch_size",
    type=int,
    default=50,
    help="Set the batch size."
  )
  parser.add_argument(
    "--num_epochs",
    type=int,
    default=5,
    help="Number of training epochs."
  )
  parser.add_argument(
    "--eval_frequency",
    type=int,
    default=200,
    help="The validation frequency."
  )
  parser.add_argument(
    "--eval_batch_size",
    type=int,
    default=1000,
    help="The batch size for validation and final test."
  )
  parser.add_argument(
    "--log_dir",
    type=str,
    default="./logs",
    help="The directory for writing logs and saving summaries."
  )
  FLAGS = parser.parse_args()
  tf.app.run(main=main)
