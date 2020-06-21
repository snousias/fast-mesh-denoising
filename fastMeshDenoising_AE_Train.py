import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed
from tensorflow.python.platform import gfile
from tensorflow.python.util.deprecation import deprecated
import scipy.io
from six.moves import xrange
import json
import time
import random
from tensorflow.contrib.factorization import KMeans
import sklearn as sk
import fastMeshDenoising_Data_Utils_Train as dat
from fastMeshDenoising_Data_Utils_Train import *
from scipy.spatial.distance import pdist, squareform

# https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/3_NeuralNetworks/autoencoder.ipynb
#

params = {
    'keyword': '',
    'learning_rate': 0.000004,
    'training_epochs': 12,
    'batch_size': 128,
    'usePolar': False,
    'useWeights': False,
    'useHeatmap': False,
    'useFeature': True,
    'useFlat': True,
    'trainSeparateModels': False,
    'n_hidden_layers': [512,256,128,64],
}
learning_rate = params['learning_rate']
training_epochs = params['training_epochs']
batch_size = params['batch_size']
n_hidden_layers = params['n_hidden_layers']


class DataSet(object):
    """Container class for a dataset (deprecated).

    THIS CLASS IS DEPRECATED. See
    [contrib/learn/README.md](https://www.tensorflow.org/code/tensorflow/contrib/learn/README.md)
    for general migration instructions.
    """

    def __init__(self,
                 images,
                 labels,
                 fake_data=False,
                 one_hot=False,
                 dtype=dtypes.float32,
                 reshape=True,
                 seed=None):
        """Construct a DataSet.
        one_hot arg is used only if fake_data is true.  `dtype` can be either
        `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
        `[0, 1]`.  Seed arg provides for convenient deterministic testing.
        """
        seed1, seed2 = random_seed.get_seed(seed)
        # If op level seed is not set, use whatever graph level seed is returned
        np.random.seed(seed1 if seed is None else seed2)
        dtype = dtypes.as_dtype(dtype).base_dtype
        if dtype not in (dtypes.uint8, dtypes.float32):
            raise TypeError(
                'Invalid image dtype %r, expected uint8 or float32' % dtype)
        if fake_data:
            self._num_examples = 10000
            self.one_hot = one_hot
        else:
            assert images.shape[0] == labels.shape[0], (
                    'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
            self._num_examples = images.shape[0]

            # Convert shape from [num examples, rows, columns, depth]
            # to [num examples, rows*columns] (assuming depth == 1)
            if reshape:
                # assert images.shape[3] == 1
                images = images.reshape(images.shape[0], images.shape[1] * images.shape[2])  # order='F'
            if dtype == dtypes.float32:
                # Convert from [0, 255] -> [0.0, 1.0].
                images = images.astype(np.float32)
                # images = np.multiply(images, 1.0 / 255.0)
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, fake_data=False, shuffle=True):
        """Return the next `batch_size` examples from this data set."""
        if fake_data:
            fake_image = [1] * 784
            if self.one_hot:
                fake_label = [1] + [0] * 9
            else:
                fake_label = 0
            return [fake_image for _ in xrange(batch_size)], [
                fake_label for _ in xrange(batch_size)
            ]
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._images = self.images[perm0]
            self._labels = self.labels[perm0]
        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            images_rest_part = self._images[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            # Shuffle the data
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._images = self.images[perm]
                self._labels = self.labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            images_new_part = self._images[start:end]
            labels_new_part = self._labels[start:end]
            return np.concatenate((images_rest_part, images_new_part), axis=0), np.concatenate(
                (labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._images[start:end], self._labels[start:end]

    def fetch(self):
        return self._images, self._labels


def asCartesian(rthetaphi):
    # takes list rthetaphi (single coord)
    r = rthetaphi[0]
    theta = rthetaphi[1] * np.pi / 180  # to radian
    phi = rthetaphi[2] * np.pi / 180
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return [x, y, z]


def asSpherical(xyz):
    # takes list xyz (single coord)
    x = xyz[0]
    y = xyz[1]
    z = xyz[2]
    r = np.sqrt(x * x + y * y + z * z)
    theta = np.arccos(z / r) * 180 / np.pi  # to degrees
    phi = np.arctan2(y, x) * 180 / np.pi
    if phi < 0:
        phi = phi + 360
    if theta < 0:
        theta = theta + 360
    return [r, theta, phi]


def mat2Sph(M):
    for i in range(0, np.size(M, axis=1)):
        xyz = M[:, i]
        r, theta, phi = asSpherical(xyz)
        M[0, i] = r
        M[1, i] = theta
        M[2, i] = phi
    return M


def mat2Cartesian(M):
    for i in range(0, np.size(M, axis=1)):
        rthetaphi = M[:, i]
        x, y, z = asSpherical(rthetaphi)
        M[0, i] = x
        M[1, i] = y
        M[2, i] = z
    return M


def imputeSph(M, thetaImp, phiImp):
    for i in range(0, np.size(M, axis=1)):
        M[1, i] = M[1, i] + thetaImp
        M[2, i] = M[2, i] + phiImp
        if M[1, i] >= 360:
            M[1, i] = M[1, i] - int(M[1, i] / 360) * 360
        if M[2, i] >= 360:
            M[2, i] = M[2, i] - int(M[2, i] / 360) * 360
    return M


def encoder(X, n_hidden_layers, weights, bias):
    layers = [];
    for idx, val in enumerate(n_hidden_layers):
        if idx == 0:
            layers.append(
                tf.nn.sigmoid(tf.matmul(X, weights['encoder_w' + str(idx + 1)]) + bias['encoder_b' + str(idx + 1)])
            )
        if idx > 0:
            layers.append(
                tf.nn.sigmoid(tf.matmul(layers[idx - 1], weights['encoder_w' + str(idx + 1)]) + bias['encoder_b' + str(idx + 1)])
            )
    return layers[len(n_hidden_layers) - 1]


def decoder(x, n_hidden_layers, weights, bias):
    layers = [];
    for idx, val in enumerate(n_hidden_layers):
        if idx == 0:
            layers.append(
                tf.nn.sigmoid(tf.matmul(x, weights['decoder_w' + str(idx + 1)]) + bias['decoder_b' + str(idx + 1)])
            )
        if idx > 0:
            layers.append(
                tf.nn.sigmoid(tf.matmul(layers[idx - 1], weights['decoder_w' + str(idx + 1)]) + bias['decoder_b' + str(idx + 1)])
            )
    return layers[len(n_hidden_layers) - 1]


# Parameters
def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


def setW():
    weights = {}
    for idx, val in enumerate(n_hidden_layers):
        if idx == 0:
            weights.update(
                # {'encoder_w' + str(idx + 1): tf.Variable(tf.random_uniform([n_input, val]))}
                # {'encoder_w' + str(idx + 1): tf.Variable(tf.truncated_normal([n_input, val], mean=0.0, stddev=0.5))}
                {'encoder_w' + str(idx + 1): tf.Variable(xavier_init([n_input, val]))}
            )
        if idx > 0:
            weights.update(
                # {'encoder_w' + str(idx + 1): tf.Variable(tf.random_uniform([n_hidden_layers[idx - 1], n_hidden_layers[idx]]))}
                # {'encoder_w' + str(idx + 1): tf.Variable(tf.truncated_normal([n_hidden_layers[idx - 1], n_hidden_layers[idx]], mean=0.0, stddev=0.5))}
                {'encoder_w' + str(idx + 1): tf.Variable(xavier_init([n_hidden_layers[idx - 1], n_hidden_layers[idx]]))}
            )

    for idx, val in enumerate(n_hidden_layers):
        tot = len(n_hidden_layers) - 1
        j = tot - idx
        if idx < tot:
            weights.update(
                # {'decoder_w' + str(idx + 1): tf.Variable(tf.random_uniform([n_hidden_layers[j], n_hidden_layers[j - 1]]))}
                # {'decoder_w' + str(idx + 1): tf.Variable(tf.truncated_normal([n_hidden_layers[j], n_hidden_layers[j - 1]], mean=0.0, stddev=0.5))}
                {'decoder_w' + str(idx + 1): tf.Variable(xavier_init([n_hidden_layers[j], n_hidden_layers[j - 1]]))}
            )
        if idx == tot:
            weights.update(
                # {'decoder_w' + str(idx + 1): tf.Variable(tf.random_uniform([n_hidden_layers[j], n_input]))}
                # {'decoder_w' + str(idx + 1): tf.Variable(
                #     tf.truncated_normal([n_hidden_layers[j], n_input], mean=0.0, stddev=0.5))}
                {'decoder_w' + str(idx + 1): tf.Variable(xavier_init([n_hidden_layers[j], n_input]))}
            )
    return weights


def setB():
    bias = {}
    for idx, val in enumerate(n_hidden_layers):
        bias.update(
            {'encoder_b' + str(idx + 1): tf.Variable(tf.zeros([1, val]))}
            # {'encoder_b' + str(idx + 1): tf.Variable(tf.truncated_normal([1, val], mean=0.0, stddev=0.5))}
            # {'encoder_b' + str(idx + 1): tf.Variable(xavier_init(1, val))}
        )
    for idx, val in enumerate(n_hidden_layers):
        tot = len(n_hidden_layers) - 1
        j = tot - idx
        if idx < tot:
            bias.update(
                {'decoder_b' + str(idx + 1): tf.Variable(tf.zeros([1, n_hidden_layers[j - 1]]))}
                # {'decoder_b' + str(idx + 1): tf.Variable(
                #     tf.truncated_normal([1, n_hidden_layers[j - 1]], mean=0.0, stddev=0.5))}
                # {'decoder_b' + str(idx + 1): tf.Variable(xavier_init(1, n_hidden_layers[j - 1]))}
            )
        if idx == tot:
            bias.update(
                {'decoder_b' + str(idx + 1): tf.Variable(tf.zeros([1, n_input]))}
                # {'decoder_b' + str(idx + 1): tf.Variable(tf.truncated_normal([1, n_input], mean=0.0, stddev=0.5))}
                # {'decoder_b' + str(idx + 1): tf.Variable(xavier_init(1, n_input))}
            )
    return bias


n_input = np.size(train_images_original, axis=1) * np.size(train_images_original, axis=2)

options = dict(dtype=dtypes.float32, reshape=True, seed=None)
weights = setW()
bias = setB()

X = tf.placeholder(tf.float32, shape=(None, n_input))
X_noise = tf.placeholder(tf.float32, shape=(None, n_input))
encoder_op = encoder(X_noise, n_hidden_layers, weights, bias)
decoder_op = decoder(encoder_op, n_hidden_layers, weights, bias)
entropy = tf.losses.mean_squared_error(labels=X, predictions=decoder_op)
loss = tf.reduce_mean(entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize((tf.abs(loss)))
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize((tf.abs(loss)))


#############################################################################
#############################################################################
#############################################################################
val = DataSet(test_images_original, test_labels_updated, **options)
val_noisy = DataSet(test_images_noisy, test_labels_updated, **options)
train = DataSet(train_images_original, train_labels_updated, **options)
train_noisy = DataSet(train_images_noisy, train_labels_updated, **options)
#############################################################################
#############################################################################
#############################################################################


ts = int(time.time())
reportPath = dat.rootdir+'CVAE/sessions/AE/'
pred = decoder_op
if True:
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    num_batches = int(train.num_examples / batch_size)
    for epoch in range(training_epochs):
        total_loss = 0
        for batch in range(num_batches):
            x, _ = train.next_batch(batch_size)
            x_noise, _ = train_noisy.next_batch(batch_size)
            _, l = sess.run([optimizer, loss], feed_dict={X_noise: x_noise, X: x})
            total_loss += l
        print("Epoch {0}: {1}".format(epoch, total_loss))
        x_pred = sess.run(pred, feed_dict={X_noise: val_noisy.images})
        print("Epoch {0}: {1}".format(epoch, sk.metrics.mean_squared_error(x_pred, val.images)))
        if epoch>5 & epoch % 2 == 0:
            result = x_pred[:, 0:3]
            # Convert definition space
            result = 2.0 * result - 1.0 * np.ones(np.shape(result))
            result = np.asarray(result)
            mModelToProcessDenoised = copy.deepcopy(mModelToProcess)
            if doRotate:
                for r in range(0, np.size(result, axis=0)):
                    result[r, :] = rotate(result[r, :], mModelToProcessDenoised.faces[r].rotationAxis,
                                          -mModelToProcessDenoised.faces[r].theta)
            updateVerticesWithNormals(mModelToProcessDenoised, result, 20)
            exportObj(mModelToProcessDenoised, dat.rootdir+'Results-0/denoised_' + str(epoch) + '.obj')
            with open(dat.rootdir+'Results-0/result_' + str(epoch) + 'A.csv',
                      'w') as writeFile:
                for j in range(0, np.size(result, axis=0)):
                    line = str(result[j, 0]) + "," + str(result[j, 1]) + "," + str(result[j, 2])
                    writeFile.write(line)
                    writeFile.write('\n')
    exportObj(mModelToProcessDenoised,
              dat.rootdir+'Results-0/Comparisons/Denoised/AE/' + dat._modelName +dat.noiseLevelAsString+'_' + str(
                  dat.numOfElements) + '.obj')
    exportObj(mModelToProcess,
              dat.rootdir+'Results-0/Comparisons/Denoised/AE/' + dat._modelName + '_noisy'+dat.noiseLevelAsString +'.obj')
    with open(dat.rootdir+'Results-0/Comparisons/Denoised/AE/' + dat._modelName + '_normals'+dat.noiseLevelAsString+'_' + str(
                  dat.numOfElements) + '.csv','w') as writeFile:
        for j in range(0, np.size(result, axis=0)):
            line = str(result[j, 0]) + "," + str(result[j, 1]) + "," + str(result[j, 2])
            writeFile.write(line)
            writeFile.write('\n')





    saver = tf.train.Saver()
    save_path = saver.save(sess, reportPath +"model_"+str(numOfElements)+dat.noiseLevelAsString+".ckpt")
    print("Model saved in path: %s" % save_path)
    with open(reportPath + "readme.txt", 'w') as file:
        file.write(json.dumps(params))  # use `json.loads` to do the reverse

