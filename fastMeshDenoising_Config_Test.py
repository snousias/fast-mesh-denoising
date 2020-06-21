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
from commonReadModelV3 import Geometry
import scipy.io
from tensorflow.contrib.factorization import KMeans
import sklearn as sk
from scipy.spatial.distance import pdist, squareform
import pickle
#=========Generic_configurations=========#
useGuided = False
doRotate = True
noiseLevel = 0.1
nClusters = 200
numberOfPermutations = 10
numOfElements = 20
selectedModel = 0
noiseLevelAsString = '_n1'
trainSet=range(0, 3)
#=========Actions========================#
doReadOBJ = True
doTrain = False
doTest = True
modelsOnly = False
#========================================#

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


