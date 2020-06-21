import tensorflow as tf
import numpy as np
import random
import gzip
import tarfile
import pickle
import os
from six.moves import urllib
from CVAEplot import *
from commonReadModelV3 import *

import scipy.io
from tensorflow.contrib.factorization import KMeans
import sklearn as sk
from scipy.spatial.distance import pdist, squareform
import pickle
import tensorflow as tf
import numpy as np
from CVAEutils import *
from CVAEplot import *
from CVAE import *
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
from commonReadModelV3 import Geometry,Vertex,Face,Edge
from commonReadModelV3 import Vertex
from commonReadModelV3 import Face
from commonReadModelV3 import Geometry


# ===================== Configutation ============================ #
useGuided = False
doRotate = True
noiseLevel = 0.1
nClusters = 200
numberOfPermutations = 10
numOfElements = 20
selectedModel = 10
noiseLevelAsString = '_n1'
trainSet=range(0, 8)
bilateralIterations=1
bilateralSigma=0.25
bilateralNeighb=20
vertexUpdateIterations=20

doReadOBJ = False
doTrain = False
doTest = False
modelsOnly = True
# ===================== Initializations ============================ #

patchSizeGuided = numOfElements
keyTest = '_' + str(numOfElements)+noiseLevelAsString
#Add noise level
keyTrain = '_8x'+str(numberOfPermutations)+'_' + str(numOfElements) + '_simple_normals_normalized'+noiseLevelAsString
#keyTrain = '_scanned'+str(numberOfPermutations)+'_' + str(numOfElements) + '_simple_normals_normalized'+noiseLevelAsString
#keyTrain = '_scanned'+str(numberOfPermutations)+'_' + str(numOfElements) + '_simple_normals_normalized'+noiseLevelAsString
# distType='cosine'
distType = 'squared_euclidean'
rootdir= './'
root = rootdir+'meshes/GroundTruth/'
rootNoisy = rootdir+'meshes/Noisy/'
trainModels = ['block',
               'casting',
               'coverrear_Lp',
               'ccylinder',
               'eight',
               'joint',
               'part-Lp',
               'cad',
               'fandisk',
               'chinese-lion',
               'sculpt',
               'rockerarm',
               'smooth-feature',
               'trim-star',
               'gear',
               'boy01-scanned',
               'boy02-scanned',
               'pyramid-scanned',
               'girl-scanned',
               'cone-scanned',
               'sharp-sphere',
               'leg',
               'screwdriver',
               'carter100K',
               'pulley',
               'pulley-defects'
               ]
# ===================== Initializations ============================ #
train_images_original = []
train_images_noisy = []
test_images_original = []
test_images_noisy = []
NormalsNoisyTest = np.empty(shape=[0, numOfElements])
t = time.time()
# ===================== Read model ============================ #
_modelName = trainModels[selectedModel]
keyTest += '_' + _modelName
modelName = _modelName
print('Reading model ' + modelName)
mModelTest = []
mModelSrc = root + modelName + '.obj'
mModelSrcNoisy = rootNoisy + modelName + noiseLevelAsString + '.obj'
mModelToProcess = loadObj(mModelSrcNoisy)
updateGeometryAttibutes(mModelToProcess, useGuided=useGuided, numOfFacesForGuided=patchSizeGuided)
# ===================== Patches,test_images_noisy ============================ #
print('Read model complete', time.time() - t)
patches = []
for i in range(0, len(mModelToProcess.faces)):
    if i % 200 == 0:
        print('Extract patch information : ' + str(
            np.round((100 * i / len(mModelToProcess.faces)), decimals=2)) + ' ' + '%')
    p,r = neighboursByFace(mModelToProcess, i, numOfElements)
    patches.append(p)
NormalsNoisy = np.empty(shape=[len(mModelToProcess.faces), 3,numOfElements])
for idx,p in enumerate(patches):
    patchFacesNoisy = [mModelToProcess.faces[i] for i in p]
    normalsPatchFacesNoisy = np.asarray([pF.faceNormal for pF in patchFacesNoisy])
    if idx % 200 == 0:
        print('Rotate patch information : ' + str(
            np.round((100 * idx / len(mModelToProcess.faces)), decimals=2)) + ' ' + '%')
    # vec = np.mean(np.asarray([fnm.faceNormal for fnm in patchFacesNoisy]), axis=0)
    vec = np.mean(np.asarray([fnm.area * fnm.faceNormal for fnm in patchFacesNoisy]), axis=0)
    vec = vec / np.linalg.norm(vec)
    target = np.asarray([0.0, 1.0, 0.0])
    axis, theta = computeRotation(vec, target)
    mModelToProcess.faces[idx] = mModelToProcess.faces[idx]._replace(rotationAxis=axis, theta=theta)
    normalsPatchFacesNoisy = rotatePatch(normalsPatchFacesNoisy, axis, theta)
    normalsPatchFacesNoisy = normalsPatchFacesNoisy[np.newaxis, :, :]
    NormalsNoisy[idx,:,:]=normalsPatchFacesNoisy
test_images_noisy = (NormalsNoisy + 1.0 * np.ones(np.shape(NormalsNoisy))) / 2.0
test_images_noisy = np.round(test_images_noisy, decimals=6)
print('Process complete')
print('Time:' + str(time.time() - t))
# ===================== KMeans,test_labels_updated ============================ #
Xk = tf.placeholder(tf.float32, shape=[None, len(test_images_noisy[0].ravel())])
kmeans = KMeans(inputs=Xk, num_clusters=nClusters, distance_metric=distType, use_mini_batch=True)
training_graph = kmeans.training_graph()
(all_scores, cluster_idx, scores, cluster_centers_initialized, init_op, train_op) = training_graph
avg_distance = tf.reduce_mean(scores)
sesskmeans = tf.Session()
tf.train.Saver().restore(sesskmeans,
                         rootdir + 'sessions/KMeans/modelKMeans_' + str(numOfElements) + noiseLevelAsString + '.ckpt')
kMInputTest = np.reshape(test_images_noisy,(np.shape(test_images_noisy)[0],np.shape(test_images_noisy)[1]*np.shape(test_images_noisy)[2]),order='F').tolist()
_, d, idx = sesskmeans.run([train_op, avg_distance, cluster_idx],
                           feed_dict={Xk: kMInputTest})
sesskmeans.close()
mSizeTest = int((np.size(test_images_noisy, axis=0)))
test_labels_updated = np.zeros((mSizeTest, nClusters))
for i in range(0, mSizeTest):
    test_labels_updated[i, idx[0][i]] = 1.0


# ===================== Bilateral iterations ============================ #
class data_pipeline:
    def __init__(self, type):
        self.type = type
        self.debug = 0
        self.batch = 0

    def load_preprocess_data(self):
        self.train_images_original = train_images_original
        self.train_images_noisy = train_images_noisy
        self.test_images_original = test_images_original
        self.test_images_noisy = test_images_noisy
        self.train_images = train_images_original
        self.train_labels = []
        self.valid_images = test_images_original
        self.valid_labels = test_labels_updated
        self.test_images = test_images_original
        self.test_labels = test_labels_updated
        print("-" * 80)
        print("-" * 80)
        print("training size: ", np.shape(self.train_images), ", ", np.shape(self.train_labels))
        print("valid size:    ", np.shape(self.valid_images), ", ", np.shape(self.valid_labels))
        print("test size:     ", np.shape(self.test_images), ", ", np.shape(self.test_labels))
        return self.train_images, self.train_labels, self.valid_images, self.valid_labels, self.test_images, self.test_labels

    def next_batch(self, train_images_original,train_images_noisy, train_labels_updated, batch_size, make_noise=None):
        self.length = len(train_images_original) // batch_size
        batch_xs = train_images_original[self.batch * batch_size: self.batch * batch_size + batch_size, :, :]
        batch_noised_xs = train_images_noisy[self.batch * batch_size: self.batch * batch_size + batch_size, :, :]
        batch_ys = train_labels_updated[self.batch * batch_size: self.batch * batch_size + batch_size, :]
        self.batch += 1
        if self.batch == (self.length):
            self.batch = 0
        return batch_xs, batch_noised_xs, batch_ys

    def get_total_batch(self, images, batch_size):
        self.batch_size = batch_size
        return len(images) // self.batch_size

# ===================== Intialize CVAE ============================ #
if __name__ =="__main__":
    flag = tf.app.flags
    FLAGS = flag.FLAGS
    flag.DEFINE_float("learning_rate", 0.00002, "learning rate for training")
    flag.DEFINE_integer("n_epoch", 12, "number of Epoch")
    flag.DEFINE_integer("n_z", 2, "Dimension of latent variables")
    flag.DEFINE_float("keep_prob", 0.95,"Dropout rate")
    flag.DEFINE_float("decay_rate", 0.998,"learning rate decay rate")
    flag.DEFINE_integer("batch_size", 256, "Batch size for training")
    _, height,width = np.shape(test_images_noisy)
    n_cls = nClusters
    X = tf.placeholder(dtype = tf.float32, shape = [None, height, width], name ="Input")
    X_noised = tf.placeholder(dtype = tf.float32, shape = [None, height, width], name ="Input_noised")
    Y = tf.placeholder(dtype = tf.float32, shape = [None, n_cls], name = "labels")
    keep_prob = tf.placeholder(dtype = tf.float32, name = "drop_rate")
    CVAE = CVAE([_,height, width], n_cls, [1024, 2048, 4096, 4096], FLAGS.n_z, keep_prob)
    z, output, loss = CVAE.Conditional_Variational_AutoEncoder(X, X_noised, Y, keep_prob)
    global_step = tf.Variable(0, trainable=False)
    latent = tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.n_z], name="latent_input")

    # ===================== Load session ============================ #
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    tf.train.Saver().restore(sess, rootdir+'sessions/CVAE/model_' + str(numOfElements)+noiseLevelAsString+ '.ckpt')
    # ===================== Run inference ============================ #
    print("_" * 80)
    print("Processing")
    rotAx=[ mModelToProcess.faces[r].rotationAxis for r in range(0, np.size(test_images_noisy, axis=0))]
    thetas=[-mModelToProcess.faces[r].theta for r in range(0, np.size(test_images_noisy, axis=0))]
    rotAx=np.asarray(rotAx)
    thetas=np.asarray(thetas)
    nIterTest=1
    tottime=0
    for g in range(0,nIterTest):
        t = time.time()
        images = sess.run(output, feed_dict={X_noised: test_images_noisy,
                                             Y: test_labels_updated,
                                             keep_prob: 1.0})
        print("Step 1 complete "+str(time.time() - t))
        result = np.asarray(2.0 * images[:, 0:3] - 1.0 * np.ones(np.shape(images[:, 0:3])))
        print("Transform complete " + str(time.time() - t))
        tottime = tottime + (time.time() - t)
    tottime = tottime / nIterTest
    result=np.asarray([rotate(result[r, :], rotAx[r,:],thetas[r]) for r in range(0, np.size(result, axis=0))])
    print("Rotation complete "+str(time.time() - t))
    sess.close()
    print("Full time : "+str(tottime))

    # ===================== Run bilateral============================ #
    resultfiltered=BilateralNormalFiltering(mModelToProcess, result, sigma2=bilateralSigma,neighbours=bilateralNeighb,nIter=bilateralIterations)
    updateVerticesWithNormals(mModelToProcess, resultfiltered, vertexUpdateIterations)
    print("Step 2 complete "+str(time.time() - t))
    exportObj(mModelToProcess,
             rootdir+'results/Comparison/Denoised/CVAE/' + _modelName +  noiseLevelAsString + '_' + str(numOfElements) + '.obj')
    print("Export complete")
    print("Export normals complete "+str(time.time() - t))
