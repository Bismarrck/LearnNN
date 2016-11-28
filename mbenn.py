"""
This python program is used to repeat the work of doi:10.1021/acs.jctc.6b00994.
This is also my first step on utlitizing deep learning with computational
chemistry.

Author: Xin Chen
Email: chenxin13@mails.tsinghua.edu.cn
Date: 2016-11-24

"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np
from scipy.misc import comb
import hashlib
import re
import h5py
from os.path import isfile
import shutil
import time
import argparse
import sys
from sklearn.metrics import pairwise_distances
from itertools import combinations
from pymatgen.core.units import Ha_to_eV

__author__ = 'Xin Chen'
__email__ = "chenxin13@mails.tsinghua.edu.cn"


FLAGS = None

# The patch size is always 1.
PATCH_SIZE = 1

# The main datafile.
HDF5_DATABASE_FILE = "b20pbe.hdf5"

# The npz file to save similarity-related data.
XYZ_USR_FILE = "b20pbe.npz"

# Each B20 cluster has twenty Boron atoms.
NUM_SITES = 20

# The total number of structures in the CP2K/XYZ file is 209660.
XYZ_FILE = "b20pbe.xyz"
TOTAL_SIZE = 209660

# My poor MacBookPro ...
TEST_SIZE = 10000
VALID_SIZE = 10000
TRAIN_SIZE = 80000
LOAD_SIZE = TEST_SIZE + VALID_SIZE + TRAIN_SIZE

# The seed is my room number.
SEED = 235


def get_np_dtype():
  """
  Return the float type of datasets and target values.
  """
  if FLAGS.precise:
    return np.float64
  else:
    return np.float32


def chunk_range(start, stop, chunksize):
  istart = start
  while istart < stop:
    istop = min(istart + chunksize, stop)
    yield istart, istop
    istart = istop


def transform_coords(coords, chunksize, mapping, l=4.0, k=4, verbose=True):
  """
  Transform the raw cartesian coordinates to input features.

  Args:
    coords: a 2D array with shape [M,3N] representing the cartesian coordinates.
    chunksize: the transformed array is too large. So save it piece by piece.
    mapping: a `h5py.Dataset`, which is a symbolic to the real data on disk.
    l: the exponential parameter.
    k: the many-body parameter.
    verbose: print the transformation progress if True.

  """

  def exponential(x):
    return np.exp(-x / l)

  ntotal, n3 = coords.shape[:2]
  n = n3 // 3
  cnkv = comb(n, k, exact=True)
  ck2v = comb(k, 2, exact=True)
  cnkl = list(combinations(range(NUM_SITES), 4))
  # Using mapping indices can increase the speed 30 times!
  indices = np.zeros((ck2v, cnkv), dtype=int)
  for i in xrange(cnkv):
    for j, (vi, vj) in enumerate(combinations(cnkl[i], 2)):
      indices[j, i] = vi * n + vj
  dataset = np.zeros((chunksize, 1, cnkv, ck2v), dtype=np.float32)
  tic = time.time()
  if verbose:
    print("Transform the cartesian coordinates ...\n")
  for i, inext in chunk_range(0, ntotal, chunksize):
    for j in xrange(i, inext):
      dists = pairwise_distances(coords[j].reshape((-1, 3))).flatten()
      for k in xrange(ck2v):
        dataset[j - i, 0, :, k] = exponential(dists[indices[k]])
      del dists
    batch_size = inext - i
    mapping[i: inext, ...] = dataset[:batch_size, ...]
    if verbose:
      print("Progress: %7d  /  %7d" % (inext, ntotal))
    dataset.fill(0.0)
  del indices
  del dataset
  if verbose:
    print("")
    print("Total time: %.3f s\n" % (time.time() - tic))


def md5(filename):
  """ Return the md5 checksum of the given file.

  Args:
    filename: a file.

  Returns:
    checksum: the MD5 checksum of the file.

  """
  hash_md5 = hashlib.md5()
  with open(filename, "rb") as f:
    for chunk in iter(lambda: f.read(4096), b""):
      hash_md5.update(chunk)
  return hash_md5.hexdigest()


def randomize(dataset, targets, repeats=5):
  """
  Permute the given dataset and targets several times.
  """
  for _ in range(repeats):
    permutation = np.random.permutation(targets.shape[0])
    dataset[:] = dataset[permutation, ...]
    targets[:] = targets[permutation, ...]
    del permutation
  return dataset, targets


def split_dataset(dataset, targets):
  """
  Split the dataset and targets to three parts: training, validation, testing.
  """
  mark = [0, TRAIN_SIZE, TRAIN_SIZE + VALID_SIZE, LOAD_SIZE]
  return dataset[mark[0]: mark[1], ...], targets[mark[0]: mark[1], ...], \
         dataset[mark[1]: mark[2], ...], targets[mark[1]: mark[2], ...], \
         dataset[mark[2]: mark[3], ...], targets[mark[2]: mark[3], ...]


def skewness(vector):
  """
  This function returns the cube root of the skewness of the given vector.

  Args:
    vector: a vector, [n, ]

  Returns:
    skewness: the skewness of the vector.

  References:
    http://en.wikipedia.org/wiki/Skewness
    http://en.wikipedia.org/wiki/Moment_%28mathematics%29

  """
  v = np.asarray(vector)
  sigma = np.std(v)
  s = np.mean((v - v.mean()) ** 3.0)
  eps = 1E-8
  if np.abs(sigma) < eps or np.abs(s) < eps:
    return 0.0
  else:
    return s / (sigma ** 3.0)


def get_usr_features(coords):
  """
  Return the USR feature vector of the given molecule.

  Args:
    coords: a flatten array of cartesian coordinates, [3N, ]

  Returns:
    usr_vector: the standard USR fingerprints.

  """

  def _compute_usr(v1, v2, v3, v4, c):
    vector = np.zeros(12)
    k = 0
    for v in [v1, v2, v3, v4]:
      di = np.linalg.norm(v - c, axis=1)
      vector[k: k + 3] = np.mean(di), np.std(di), skewness(di)
      k += 3
    return vector

  cart_coords = coords.reshape((-1, 3))
  x = cart_coords.mean(axis=0)
  d = np.linalg.norm(x - cart_coords, axis=1)
  y = cart_coords[np.argmin(d)]
  z = cart_coords[np.argmax(d)]
  d = np.linalg.norm(z - cart_coords, axis=1)
  w = cart_coords[np.argmax(d)]

  return _compute_usr(x, y, z, w, cart_coords)


def remove_duplicates(coords, energies, threshold=0.99, verbose=True):
  """
  Remove duplicated structures. The similarity algorithm used here is USR.

  This implementation now takes about 15 minutes on my MacBook Pro using one
  core. The speed is acceptable.
  """
  if verbose:
    print("Remove duplicated data samples ...\n")

  group = "similarity"
  n = len(coords)

  hdb = h5py.File(HDF5_DATABASE_FILE)
  if group not in hdb:
    hdb.create_group(group)

  try:
    v = hdb[group]["usr"][:]
  except Exception:
    if verbose:
      print("Compute USR features ...\n")
    v = np.zeros((n, 12), dtype=get_np_dtype())
    tic = time.time()
    for i in xrange(n):
      if verbose and i % 2000 == 0:
        print("Progress: %7d  /  %7d" % (i, n))
      v[i] = get_usr_features(coords[i])
    if verbose:
      print("")
      print("Time for computing USR features: %.3f s\n" % (time.time() - tic))
    hdb[group].create_dataset("usr", data=v)

  try:
    indices = hdb[group]["indices"][:]
  except Exception:
    if verbose:
      print("Comparing similarities. Be patient ...\n")
    tic = time.time()
    keep = np.ones(n, dtype=bool)
    for i in xrange(n):
      if not keep[i]:
        continue
      sij = 1.0 / (1.0 + np.sum(np.abs(v[i] - v[i + 1:, ...]), axis=1))
      duplicates = np.where(sij > threshold)[0]
      if len(duplicates) > 0:
        keep[duplicates + i] = False
      if verbose and i % 1000 == 0:
        print("Progress: %7d  /  %7d" % (i, n))
    indices = np.where(keep == False)[0]
    del keep
    if verbose:
      print("")
      print("Time for comparing similarities: %.3f s\n" % (time.time() - tic))
    hdb[group].create_dataset("indices", data=indices)
  finally:
    hdb.close()

  del v

  if verbose:
    print("Number of duplicated samples: %d\n" % len(indices))
  coords = np.delete(coords, indices, axis=0)
  energies = np.delete(energies, indices)
  return coords, energies


def extract_cp2k_xyz(xyzfile, verbose=True):
  """ Extract cartesian coordinates from a merged CP2K/XYZ file.

  Args:
    xyzfile: a file with CP2K/XYZ format.
    verbose: print the extraction progress if True.

  Returns:
    coords: an array of cartesian coordinates, [M, 3N]
    energies: an array of energies, [M, ].

  """
  group = "raw"
  hdb = h5py.File(HDF5_DATABASE_FILE)
  if group not in hdb:
    hdb.create_group(group)

  try:
    coords = hdb[group]["coords"][:]
    energies = hdb[group]["energies"][:]
  except Exception:
    dtype = get_np_dtype()
    coords = np.zeros((TOTAL_SIZE, NUM_SITES * 3), np.float32)
    energies = np.zeros((TOTAL_SIZE,), dtype=dtype)
    n3 = NUM_SITES * 3
    i, ij3 = 0, 0
    stage = 0
    axyz_patt = re.compile(r"([A-Za-z]+)\s+([\w.-]+)\s+([\w.-]+)\s+([\w.-]+)")
    ener_patt = re.compile(r"i\s=\s+(\d+),\sE\s=\s+([\w.-]+)")
    tic = time.time()
    if verbose:
      print("Extract cartesian coordinates ...\n")
    with open(xyzfile, "r") as f:
      for ln, line in enumerate(f):
        l = line.strip()
        if l == "":
          continue
        if i == TOTAL_SIZE:
          break
        if stage == 0:
          if l.isdigit():
            n = int(l)
            if n != NUM_SITES:
              raise IOError("Line %d: Error!" % ln)
            stage = 1
        elif stage == 1:
          m = ener_patt.search(l)
          if m:
            energies[i] = dtype(m.group(2)) * Ha_to_eV
            stage = 2
        elif stage == 2:
          m = axyz_patt.search(l)
          if m:
            coords[i, ij3: ij3 + 3] = \
              float(m.group(2)), float(m.group(3)), float(m.group(4))
            ij3 += 3
            if ij3 == n3:
              i += 1
              ij3, stage = 0, 0
              if verbose and i % 2000 == 0:
                print("Progress: %7d  /  %7d" % (i, TOTAL_SIZE))
      if verbose:
        print("")
        print("Total time: %.3f s\n" % (time.time() - tic))
    hdb[group].create_dataset("coords", data=coords, compression="gzip")
    hdb[group].create_dataset("energies", data=energies, compression="gzip")
  finally:
    hdb.close()
  return coords, energies


def may_build_datasets_cp2k(xyzfile, l=4.0, k=4, repeats=5, verbose=True):
  """
  Build the training, validation and testing dataset and targets from a merged
  CP2K/XYZ file.

  Args:
    xyzfile: a file with CP2K/XYZ format.
    l: the exponential parameter.
    k: the many-body parameter.
    repeats: the number of times to randomize the dataset.
    verbose: print the building progress if True.

  Returns:
    train_dataset, train_targets, valid_dataset, valid_targets, test_dataset,
    test_targets, min_ener, max_ener are returned. All datasets are 4D numpy
    arrays, all targets are 1D numpy arrays, min_ener and max_ener are floats.

  """
  # Compute the MD5 checksum of the xyzfile
  checksum = md5(xyzfile)

  if verbose:
    print("-> Load the training, validation and testing datasets ...\n")

  coords, energies = None, None
  dataset, targets = None, None
  backup_hdf5 = False
  build_coords = True
  build_datasets = True

  # If the HDF5 file is already existed, we try to load dataset and targets from
  # the HDF5 file directly if the checksums are equal.
  if isfile(HDF5_DATABASE_FILE):
    with h5py.File(HDF5_DATABASE_FILE, "r") as hdb:
      if hdb.attrs.get("checksum", 0) == checksum:
        # There are two main groups in this HDF5 file:
        # 1. the first group is 'train' where training data and training targets
        # are stored.
        if "train" in hdb:
          dataset = hdb["train"]["dataset"][:LOAD_SIZE]
          targets = hdb["train"]["targets"][:LOAD_SIZE]
          build_datasets = False
          build_coords = False
        # 2. the second group is 'unique' where uniquified cartesian coordinates
        # and their energies extracted from a CP2K/XYZ file are saved.
        elif "unique" in hdb.keys():
          coords = hdb["unique"]["coords"][:]
          energies = hdb["unique"]["energies"][:]
          build_coords = False
      # The checksum are not equal, so we backup the existed HDF5 databse by
      # renaming it.
      else:
        backup_hdf5 = True
    if backup_hdf5:
      if verbose:
        print("MD5 checksums mismatched. Build a new dataset.\n")
      shutil.move(HDF5_DATABASE_FILE, HDF5_DATABASE_FILE + ".bak")

  # Extract the raw cartesian coordinates and energis (eV) from the CP2K/XYZ
  # file and save these data into group 'raw'. All data are compressed with the
  # lossless gzip filter.
  if build_coords:
    coords, energies = extract_cp2k_xyz(xyzfile, verbose=verbose)
    # Remove the duplicates to reduce the dataset
    coords, energies = remove_duplicates(coords, energies, verbose=verbose)
    with h5py.File(HDF5_DATABASE_FILE) as hdb:
      # Delete the previous group `unique`. This should not happen, but it may
      # be inserted manually for debugging.
      group = "unique"
      if group in hdb.keys():
        del hdb[group]
      hdb.attrs["checksum"] = checksum
      hdb.create_group(group)
      hdb[group].create_dataset("coords", data=coords, compression="gzip")
      hdb[group].create_dataset("energies", data=energies, compression="gzip")
  elif verbose:
    print("Use existed coordinates and energies.\n")

  # Transform the cartesian coordinates to a 4D dataset. Permute this dataset
  # several times and then we split it into three parts: training, validation
  # and testing. Save these datasets and their targets into group 'cnn'.
  if build_datasets:
    # Allocate the disk space and then write transformed data piece by piece
    # because my little computer only has 16GB memory.
    shape = [len(energies), 1, comb(NUM_SITES, k, True), comb(k, 2, True)]
    hdb = h5py.File(HDF5_DATABASE_FILE)
    try:
      group = hdb.require_group("train")
      group.create_dataset("targets", data=energies)
      mapping = group.create_dataset(
        "dataset", shape=shape, dtype=np.float32)
      # Randomize the coordinates several times
      coords, energies = randomize(coords, energies, repeats=repeats)
      # Set the chunksize to 10000.
      chunksize = 10000
      transform_coords(coords, chunksize, mapping, l=l, k=k, verbose=verbose)
    except Exception as excp:
      del hdb["train"]
      raise excp
    finally:
      hdb.close()
    # After the transformation we now load the whole dataset into memory.
    with h5py.File(HDF5_DATABASE_FILE) as hdb:
      dataset = hdb["train"]["dataset"][:LOAD_SIZE]
      targets = np.array(energies[:LOAD_SIZE], copy=False)
    if verbose:
      print("Dataset size (MB)     : ", dataset.nbytes / 1024 / 1024)
      print("Targets size (MB)     : ", targets.nbytes / 1024 / 1024)
      print("")
    del coords
  elif verbose:
    print("Use existed datasets and targets.\n")

  # Determine the maximum and minimum energy. The energies should be scaled to
  # [0, 1] during training.
  min_ener = targets.min()
  max_ener = targets.max()

  # Split the dataset and targets into three parts: training, validation and 
  # testing.
  train_dataset, train_targets, \
  valid_dataset, valid_targets, \
  test_dataset, test_targets = split_dataset(dataset, targets)

  if verbose:
    print("-> Datasets and targets are loaded into memories.")
    print("")
    print("Training set          :", train_dataset.shape, train_targets.shape)
    print("Validation set        :", valid_dataset.shape, valid_targets.shape)
    print("Test set              :", test_dataset.shape, test_targets.shape)
    print("Min Energy (eV)       :", min_ener)
    print("Max Energy (eV)       :", max_ener)
    print("")
  return train_dataset, train_targets, \
         valid_dataset, valid_targets, test_dataset, test_targets, \
         min_ener, max_ener


def print_activations(t):
  """
  Print the name and shape of the input Tensor.

  Args:
    t: a Tensor.

  """
  print("%-21s : %s" % (t.op.name, t.get_shape().as_list()))


def get_tf_dtype():
  """
  Return the type of the activations, weights, and placeholder variables.
  """
  if FLAGS.precise:
    return tf.float64
  else:
    return tf.float32


def variable_summaries(tensor):
  """
  Attach a lot of summaries to a Tensor (for TensorBoard visualization).

  Args:
    tensor: a Tensor.

  """
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(tensor)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(tensor, mean))))
      tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(tensor))
    tf.summary.scalar('min', tf.reduce_min(tensor))
    tf.summary.histogram('histogram', tensor)


def inference(dataset, cnk, verbose=True):
  """
  Return the infered MBE-NN-M deep neural network model.

  Args:
    dataset: a 4D dataset Tensor as the input layer, [batch, 1, C(N,k), C(k,2)]
    cnk: the value of C(N,k), an int.
    verbose: print layer details if True.

  Returns:
    pool: the last Tensor of `tf.nn.avg_pool`, [batch, 1, 1, 1]

  References:
    Alexandrova, A. N. (2016). http://doi.org/10.1021/acs.jctc.6b00994

  """
  if verbose:
    print("-> Inference the MBE-NN-M model ...")
    print("")

  parameters = []

  def mbe_conv2d(tensor, n_in, n_out, name, activate=tf.tanh):
    """ A lazy inner function to create a `tf.nn.conv2d` Tensor.

    Args:
      tensor: a Tensor, [index, 1, w, n_in]
      n_in: the number of input channels.
      n_out: the number of output channels.
      name: the name of this layer.
      activate: the activation function, defaults to `tf.tanh`.

    Returns:
      activated: a Tensor of activated `tf.nn.conv2d`.

    """
    dtype = get_tf_dtype()
    with tf.name_scope(name):
      with tf.name_scope("filter"):
        kernel = tf.Variable(
          tf.truncated_normal(
            [1, 1, n_in, n_out], stddev=0.1, seed=SEED, dtype=dtype),
          dtype=dtype, name="kernel")
        variable_summaries(kernel)
      conv = tf.nn.conv2d(
        tensor, kernel, [1, 1, 1, 1], padding="SAME", use_cudnn_on_gpu=True)
      with tf.name_scope("biases"):
        biases = tf.Variable(
          tf.zeros([n_out], dtype=dtype), dtype=dtype, name="biases")
        variable_summaries(biases)
      bias = tf.nn.bias_add(conv, biases)
      activated = activate(bias)
      parameters.extend([kernel, biases])
      if verbose:
        print_activations(activated)
    return activated

  # Build the first three MBE layers.
  # The shape of the input data tensor is [n, 1, C(N,k), C(k,2)].
  # To fit Fk, the NN connection is localized in the second dimension, and the
  # layer size of the first dimension is kept fixed. The weights and biases of
  # NN connection are shared among different indices of the first dimension,
  # so that the fitted function form of Fk is kept consistent among different
  # k-body terms. The MBE part is composed of four layers with the following
  # sizes:
  # (C(N,k), C(k,2)) - (C(N,k), 40) - (C(N,k), 70) - (C(N,k), 60) - (C(N,k), 2).
  conv1 = mbe_conv2d(dataset, 6, 40, "Conv1")
  conv2 = mbe_conv2d(conv1,  40, 70, "Conv2")
  conv3 = mbe_conv2d(conv2,  70, 60, "Conv3")

  # Then we build the three mixing layers.
  # The mixing part is used to fit G. Within this part the NN connection is
  # localized in the first dimension, and the size of the second dimension is
  # kept fixed. The parameters of NN connection in this part are shared among
  # different indices of the second dimension. In this work, the mixing part is
  # composed of two layers with the following sizes:
  # (C(N, k), 2) - (40, 2) - (10, 2).
  conv4 = mbe_conv2d(conv3, 60,     2, "Conv4", activate=tf.nn.softplus)
  conv5 = mbe_conv2d(tf.reshape(conv4, (-1, 1, 2, cnk)),
                     cnk, 2000, "Conv5", activate=tf.nn.softplus)
  conv6 = mbe_conv2d(conv5, 2000, 400, "Conv6", activate=tf.nn.softplus)
  conv7 = mbe_conv2d(conv6,  400,  40, "Conv7", activate=tf.nn.softplus)

  # The last part is used to transform the output of mixing part to a single
  # value, representing the energy. The average-pooling is used, which means
  # that we take the average value of all elements in the matrix of the previous
  # layer as the final output. In this work, the pooling part is composed of one
  # layer of the size:
  # (10, 2) - (1).
  with tf.name_scope("Pool8"):
    pool8 = tf.nn.avg_pool(tf.reshape(conv7, [-1, 40, 2, 1]),
                           ksize=[1, 40, 2, 1],
                           strides=[1, 1, 1, 1],
                           padding="VALID")
    if verbose:
      print_activations(pool8)
      print("")

  return tf.reshape(pool8, (-1,)), parameters


def run_mbenn(batch_size=200, num_epochs=10, k=4, l=4.0, rlambda=None,
              start_learning_rate=0.1, logdir="B20", log_frequency=100,
              eval_frequency=500, save_path="b20.ckpt", save_frequency=2000):
  """
  Run the tensorflow-based MBE-NN-M network on B20 cluster.

  Args:
    batch_size: the batch size.
    num_epochs: number of epochs to train.
    k: determines the many-body contributions. This should be 4.
    l: the exponential parameter. Defaults to 4.0.
    rlambda: the lambda coefficient for regularizers.
    start_learning_rate: the initial learning rate.
    logdir: the directory for writing tensorflow logs.
    log_frequency: each `log_frequency` steps shall the rms and loss be saved.
    eval_frequency: each `eval_frequency` steps shall the vaildation be run.
    save_path: the path to save the trained model.
    save_frequency: each `save_frequency` steps shall the model be saved.

  """
  graph = tf.Graph()

  with graph.as_default():

    # Load rformatted datasets and labels from the HDF5 database file.
    train_dataset, train_targets, \
    valid_dataset, valid_targets, \
    test_dataset, test_targets, \
    min_ener, max_ener = may_build_datasets_cp2k(XYZ_FILE, l=l, k=k)

    # The function is used to scale energies to [0, 1]
    def scale_energy(ener):
      return (ener - min_ener) / (max_ener - min_ener)

    # The function is used to scale relative errors to eV.
    def scale_error(err):
      return err * (max_ener - min_ener)

    # Scale these energies to [0, 1] for training and scale the results back to
    # eV for comparison.
    test_targets = scale_energy(test_targets)
    train_targets = scale_energy(train_targets)
    valid_targets = scale_energy(valid_targets)

    # Compute the value of C(N,k) and C(k,2)
    cnk = int(comb(NUM_SITES, k))
    ck2 = int(comb(k, 2))

    # Transform these datasets and labels to Tensors.
    input_dataset = tf.placeholder(get_tf_dtype(), [batch_size, 1, cnk, ck2])
    input_targets = tf.placeholder(get_tf_dtype(), [batch_size, ])

    # Infer the MBE-NN-M network.
    estimates, parameters = inference(input_dataset, cnk)

    # Setup the RMS, regularizer and the total loss.
    with tf.name_scope("loss"):
      rms = tf.sqrt(tf.reduce_mean(tf.square(estimates - input_targets)))
      tf.summary.scalar("rms", rms)
      if rlambda is not None:
        with tf.name_scope("regularizers"):
          r = rlambda * tf.nn.l2_loss(parameters[0])
          for params in parameters:
            r += rlambda * tf.nn.l2_loss(params)
          tf.summary.scalar("regularizer", r)
        loss = rms + r
        tf.summary.scalar("loss", loss)
      else:
        loss = rms

    # Setup the training rate decay
    batch = tf.Variable(0, dtype=tf.int32)
    learning_rate = tf.train.exponential_decay(
      start_learning_rate, tf.mul(batch, batch_size), TRAIN_SIZE, 0.9,
      staircase=True)

    # TODO: Setup the momentum decay

    # Use a simple momentum optimizer
    trainer = tf.train.MomentumOptimizer(learning_rate, 0.5)
    optimizer = trainer.minimize(loss, global_step=batch)

    # Start running operations on the Graph.
    print("-> Start training MBE-NN-M ...")
    print("")

    # Merge all the summaries and write them out to ./logs (by default).
    merged = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter(logdir=logdir, graph=graph)

    # Small utility function to evaluate a dataset by feeding batches of data to
    # {eval_dataset} and pulling the results from {eval_values}.
    # Saves memory and enables this to run on smaller GPUs.
    def eval_in_batches(data):
      """ Get all predictions for a dataset by running it in small batches."""
      size = data.shape[0]
      if size < batch_size:
        raise ValueError("batch size for evals larger than dataset: %d" % size)
      eval_values = np.ndarray(shape=(size, ), dtype=get_np_dtype())
      dtype = get_np_dtype()
      for i, inext in chunk_range(0, size, batch_size):
        eval_values[i: inext] = sess.run(
          estimates,
          feed_dict={input_dataset: data[i: inext, ...].astype(dtype)})
      return eval_values

    # Enter the training session
    with tf.Session(graph=graph) as sess:

      # Build an initialization operation.
      tf.global_variables_initializer().run()

      # Register a model saver
      saver = tf.train.Saver()

      print("Initialized!")
      print("")
      print("Training Samples      :", TRAIN_SIZE)
      print("Batch Size            :", batch_size)
      print("Number of Epochs      :", num_epochs)
      print("Log Frequency         :", log_frequency)
      print("Eval Frequency        :", eval_frequency)
      print("")

      tic = time.time()
      tstart = time.time()

      # Loop through training steps.
      for step in xrange(int(num_epochs * TRAIN_SIZE) // batch_size):
        # Compute the offset of the current minibatch in the data.
        # The dataset was already shuffled in assignment 1 so we do not need to
        # randomize it.
        offset = (step * batch_size) % (TRAIN_SIZE - batch_size)
        batch_dataset = train_dataset[offset: (offset + batch_size), ...]
        batch_targets = train_targets[offset: (offset + batch_size), ...]
        # Build the feed dict to feed previous defined placeholders.
        if FLAGS.precise:
          feed_dict = {input_dataset: batch_dataset.astype(get_np_dtype()),
                       input_targets: batch_targets.astype(get_np_dtype())}
        else:
          feed_dict = {input_dataset: batch_dataset,
                       input_targets: batch_targets}
        # Run the optimization session.
        sess.run([optimizer], feed_dict=feed_dict)
        # Save the training accuracy every 100 steps.
        if step % log_frequency == 0:
          summary, error = sess.run([merged, loss], feed_dict=feed_dict)
          elapsed_time = time.time() - tic
          tic = time.time()
          print("Step %6d (epoch %5.2f)" % (
            step, float(step) * batch_size / TRAIN_SIZE))
          print("Minibatch loss        : %.6f" % error)
          print("Minibatch time        : %.3f s" % elapsed_time)
          writer.add_summary(summary, step)
          # Every `eval_frequency` steps we shall take several extra operations,
          # including printing the validation accuracy and updating the learning
          # rate.
          if step % eval_frequency == 0:
            lr = sess.run(learning_rate)
            valid_error = np.mean(np.abs(
              valid_targets - eval_in_batches(valid_dataset)))
            print("Validation error      : %.6f" % scale_error(valid_error))
            print("learning rate         : %.3f" % lr)
            print("Time since beginning  : %.3f s" % (time.time() - tstart))
            print("")
          sys.stdout.flush()
        # Save the trained model every 1000 steps.
        if step % save_frequency == 0:
          saver.save(sess, save_path=save_path, global_step=batch)
      # Close the writer
      writer.close()
      # Finally the training is completed. Now let me see if this MBE model can
      # really estimate DFT energies.
      test_error = np.mean(np.abs(test_targets - eval_in_batches(test_dataset)))
      print("")
      print("-> Test error        : %.6f" % scale_error(test_error))
      print("")
      # Do not forget to save the model one last time!
      saver.save(sess, save_path=save_path, global_step=batch)


header = """

                ================

                    MBE-NN-M

                ================

Author: Xin Chen
Date: 2016-11-25
Version: 0.1

"""


def main(_):
  # Refresh the logdir before launching the training.
  if tf.gfile.Exists(FLAGS.logdir):
    tf.gfile.DeleteRecursively(FLAGS.logdir)
  tf.gfile.MakeDirs(FLAGS.logdir)

  print(header)

  run_mbenn(batch_size=FLAGS.batch_size, num_epochs=FLAGS.num_epochs,
            logdir=FLAGS.logdir, log_frequency=FLAGS.log_frequency,
            eval_frequency=FLAGS.eval_frequency, rlambda=FLAGS.rlambda)


if __name__ == "__main__":

  parser = argparse.ArgumentParser()

  parser.add_argument(
    "--precise",
    action="store_true",
    default=False,
    help="Use double-precision floating numbers. This makes the calculation "
         "much more expensive."
  )
  parser.add_argument(
    "-b", "--batch_size",
    type=int,
    default=200,
    help="The batch size."
  )
  parser.add_argument(
    "-e", "--num_epochs",
    type=int,
    default=100,
    help="The total number of epochs."
  )
  parser.add_argument(
    "-o", "--logdir",
    type=str,
    default="output",
    help="The directory for saving tensorflow summaries and models."
  )
  parser.add_argument(
    "--log_frequency",
    type=int,
    default=50,
    help="The frequency to output minibatch errors."
  )
  parser.add_argument(
    "--eval_frequency",
    type=int,
    default=500,
    help="The frequency to output validation errors."
  )
  parser.add_argument(
    "--save_frequency",
    type=int,
    default=2000,
    help="The frequency to save the trained model."
  )
  parser.add_argument(
    "--rlambda",
    type=float,
    default=None,
    help="The lambda for L2 loss of regularizers. Default to None."
  )

  FLAGS = parser.parse_args([
    "--log_frequency=10", "--eval_frequency=100", "--batch_size=500",
    "--num_epochs=40"])
  tf.app.run(main)
