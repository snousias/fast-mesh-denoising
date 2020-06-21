from  fastMeshDenoising_Config_Test import  *
import fastMeshDenoising_Config_Test as conf


#
#
#
#
# noiseLevel = 0.1
# nClusters = 200
# numberOfPermutations = 10
# #Add number or permulations
# numOfElements = 20
# _selectedModel = 22
# patchSizeGuided = numOfElements
# noiseLevelAsString='_n1'
# keyTest = '_' + str(numOfElements)+noiseLevelAsString
# #Add noise level
# keyTrain = '_8x'+str(numberOfPermutations)+'_' + str(numOfElements) + '_simple_normals_normalized'+noiseLevelAsString
# #keyTrain = '_scanned'+str(numberOfPermutations)+'_' + str(numOfElements) + '_simple_normals_normalized'+noiseLevelAsString
#
#
# # distType='cosine'
# distType = 'squared_euclidean'
# rootdir= 'E:/_Groundwork/FastMeshDenoising/'
# root = rootdir+'Results-0/Comparisons/Original/'
# rootNoisy = rootdir+'Results-0/Comparisons/Noisy/'
# trainModels = [ 'block',
#                 'casting',
#                 'coverrear_Lp',
#                 'ccylinder',
#                 'eight',
#                 'joint',
#                 'part_Lp',
#                 'cad',
#                 'fandisk',
#                 'chinese-lion',
#                 'sculpt',
#                 'rockerarm',
#                 'smooth-feature',
#                 'trim-star',
#                 'gear',
#                 'conveyore',
#                 'boy01-scanned',
#                 'boy02-scanned',
#                 'pyramid-scanned',
#                 'girl-scanned',
#                 'cone-scanned',
#                 'pulley',
#                 'pulley_defects'
#                ]



_modelName = trainModels[conf.selectedModel];
keyTest += '_' + _modelName
######### Initializations ################
train_images_original = []
train_images_noisy = []
test_images_original = []
test_images_noisy = []
######### Initializations ################
t = time.time()

'''
if doTrain:
    for mIndex in range(0, 8):
        modelName = trainModels[mIndex]
        mModelSrc = root + modelName + '.obj'
        print(modelName)
        if doTrain:
            print('Initialize, read model', time.time() - t)
            mModel = loadObj(mModelSrc)
            updateGeometryAttibutes(mModel, useGuided=useGuided, numOfFacesForGuided=patchSizeGuided)
            print('Read model complete', time.time() - t)
            patches = []
            for i in range(0, len(mModel.faces)):
                if i % 20 == 0:
                    print('Extract patch information : ' + str(
                        np.round((100 * i / len(mModel.faces)), decimals=2)) + ' ' + '%')
                w, u, pv, p = neighboursByFace(mModel, i, numOfElements)
                patches.append(p)
            print('Initial model complete', time.time() - t)
        if doTrain:
            NormalsOriginalTrain = np.empty(shape=[0, numOfElements])
            NormalsNoisyTrain = np.empty(shape=[0, numOfElements])
            for repeat in range(0, numberOfPermutations):
                print('Progress:' + str(100 * (repeat + 1) / numberOfPermutations) + '%')
                mModelToProcess = copy.deepcopy(mModel)
                addNoise(mModelToProcess, noiseLevel)
                updateGeometryAttibutes(mModelToProcess, useGuided=useGuided, numOfFacesForGuided=patchSizeGuided)
                NormalsNoisy = np.empty(shape=[0, numOfElements])
                NormalsOriginal = np.empty(shape=[0, numOfElements])
                for p in patches:
                    patchFacesNoisy = [mModelToProcess.faces[i] for i in p]
                    patchFacesOriginal = [mModel.faces[i] for i in p]
                    normalsPatchFacesNoisy = []
                    normalsPatchFacesOriginal = []
                    if doRotate:
                        if useGuided:
                            vec = patchFacesNoisy[0].guidedNormal
                        else:
                            # vec = np.mean(np.asarray([fnm.faceNormal for fnm in patchFacesNoisy]), axis=0)
                            vec = np.mean(np.asarray([fnm.area * fnm.faceNormal for fnm in patchFacesNoisy]), axis=0)
                            vec = vec / np.linalg.norm(vec)
                        target = np.asarray([0.0, 1.0, 0.0])
                        axis, theta = computeRotation(vec, target)
                        for pF in patchFacesNoisy:
                            if useGuided:
                                normalsPatchFacesNoisy.append(rotate(pF.guidedNormal, axis, theta))
                            else:
                                normalsPatchFacesNoisy.append(rotate(pF.faceNormal, axis, theta))
                        for pF in patchFacesOriginal:
                            if useGuided:
                                normalsPatchFacesOriginal.append(rotate(pF.guidedNormal, axis, theta))
                            else:
                                normalsPatchFacesOriginal.append(rotate(pF.faceNormal, axis, theta))
                    else:
                        for pF in patchFacesNoisy:
                            if useGuided:
                                normalsPatchFacesNoisy.append(pF.guidedNormal)
                            else:
                                normalsPatchFacesNoisy.append(pF.faceNormal)
                        for pF in patchFacesOriginal:
                            if useGuided:
                                normalsPatchFacesOriginal.append(pF.guidedNormal)
                            else:
                                normalsPatchFacesOriginal.append(pF.faceNormal)

                    normalsPatchFacesNoisy = np.asarray(normalsPatchFacesNoisy)
                    normalsPatchFacesNoisy = np.transpose(normalsPatchFacesNoisy)

                    normalsPatchFacesOriginal = np.asarray(normalsPatchFacesOriginal)
                    normalsPatchFacesOriginal = np.transpose(normalsPatchFacesOriginal)

                    NormalsNoisy = np.concatenate((NormalsNoisy, normalsPatchFacesNoisy[:, 0:numOfElements]), axis=0)
                    NormalsOriginal = np.concatenate((NormalsOriginal, normalsPatchFacesOriginal[:, 0:numOfElements]),
                                                     axis=0)
                print('Complete', time.time() - t)
                NormalsOriginalTrain = np.concatenate((NormalsOriginalTrain, NormalsOriginal[:, 0:numOfElements]),
                                                      axis=0)
                NormalsNoisyTrain = np.concatenate((NormalsNoisyTrain, NormalsNoisy[:, 0:numOfElements]), axis=0)
            print('Process complete')

            fOrig = NormalsOriginalTrain
            fNoisy = NormalsNoisyTrain
            mSize = int((np.size(fOrig, axis=0) / 3))
            mSize = int((np.size(fNoisy, axis=0) / 3))
            for i in range(0, mSize):
                tStart = i * 3
                tEnd = i * 3 + 3
                toAppendΟ = fOrig[tStart:tEnd, 0:numOfElements]
                toAppendΟ = np.transpose(toAppendΟ)
                toAppendΟ = (toAppendΟ + 1.0 * np.ones(np.shape(toAppendΟ))) / 2.0
                toAppendN = fNoisy[tStart:tEnd, 0:numOfElements]
                toAppendN = np.transpose(toAppendN)
                toAppendN = (toAppendN + 1.0 * np.ones(np.shape(toAppendN))) / 2.0
                if True:
                    train_images_original.append(toAppendΟ)
                    train_images_noisy.append(toAppendN)
    ########################################################################################################################
    ########################################################################################################################
    ########################################################################################################################
    train_images_original = np.asarray(train_images_original)
    train_images_noisy = np.asarray(train_images_noisy)
    train_images_original = np.round(train_images_original, decimals=6)
    train_images_noisy = np.round(train_images_noisy, decimals=6)
    ########################################################################################################################
    ########################################################################################################################
    ########################################################################################################################
'''

if doTest:
    modelName = _modelName
    print('Reading model '+ modelName)
    mModelTest=[]
    if doReadOBJ:
        mModelSrc = root + modelName + '.obj'
        mModelSrcNoisy = rootNoisy + modelName + noiseLevelAsString + '.obj'
        mModelTest = loadObj(mModelSrc)
        mModelToProcess = loadObj(mModelSrcNoisy)
        # outputFile = root + modelName + '.data'
        # fw = open(outputFile, 'wb')
        # pickle.dump(mModelTest, fw)
        # pickle.dump(mModelToProcess, fw)
        # fw.close()
    # else:
    #     inputFile =  root + modelName + '.data'
    #     f = open(inputFile, 'rb')
    #     mModelTest = pickle.load(f)
    #     mModelToProcess = pickle.load(f)
    #     f.close()


    if True:
        NormalsOriginalTest = np.empty(shape=[0, numOfElements])
        NormalsNoisyTest = np.empty(shape=[0, numOfElements])
        # mModelToProcess = copy.deepcopy(mModel)
        # addNoise(mModelToProcess, noiseLevel)
        updateGeometryAttibutes(mModelTest, useGuided=useGuided, numOfFacesForGuided=patchSizeGuided)
        updateGeometryAttibutes(mModelToProcess, useGuided=useGuided, numOfFacesForGuided=patchSizeGuided)
        print('Read model complete', time.time() - t)
        patches = []
        for i in range(0, len(mModelTest.faces)):
            if i % 200 == 0:
                print('Extract patch information : ' + str(
                    np.round((100 * i / len(mModelTest.faces)), decimals=2)) + ' ' + '%')
            p,_ = neighboursByFace(mModelTest, i, numOfElements)
            patches.append(p)

        NormalsNoisy = np.empty(shape=[0, numOfElements])
        NormalsOriginal = np.empty(shape=[0, numOfElements])
        for p in patches:
            patchFacesNoisy = [mModelToProcess.faces[i] for i in p]
            patchFacesOriginal = [mModelTest.faces[i] for i in p]
            normalsPatchFacesNoisy = []
            normalsPatchFacesOriginal = []
            if doRotate:
                if useGuided:
                    vec = patchFacesNoisy[0].guidedNormal
                else:
                    # vec = np.mean(np.asarray([fnm.faceNormal for fnm in patchFacesNoisy]), axis=0)
                    vec = np.mean(np.asarray([fnm.area * fnm.faceNormal for fnm in patchFacesNoisy]), axis=0)
                    vec = vec / np.linalg.norm(vec)
                target = np.asarray([0.0, 1.0, 0.0])
                axis, theta = computeRotation(vec, target)
                idx = patchFacesNoisy[0].index
                mModelToProcess.faces[idx] = mModelToProcess.faces[idx]._replace(rotationAxis=axis, theta=theta)
                for pF in patchFacesNoisy:
                    if useGuided:
                        normalsPatchFacesNoisy.append(rotate(pF.guidedNormal, axis, theta))
                    else:
                        normalsPatchFacesNoisy.append(rotate(pF.faceNormal, axis, theta))

                idx = patchFacesOriginal[0].index
                mModelTest.faces[idx] = mModelTest.faces[idx]._replace(rotationAxis=axis, theta=theta)
                for pF in patchFacesOriginal:
                    if useGuided:
                        normalsPatchFacesOriginal.append(rotate(pF.guidedNormal, axis, theta))
                    else:
                        normalsPatchFacesOriginal.append(rotate(pF.faceNormal, axis, theta))
            else:
                for pF in patchFacesNoisy:
                    if useGuided:
                        normalsPatchFacesNoisy.append(pF.guidedNormal)
                    else:
                        normalsPatchFacesNoisy.append(pF.faceNormal)
                for pF in patchFacesOriginal:
                    if useGuided:
                        normalsPatchFacesOriginal.append(pF.guidedNormal)
                    else:
                        normalsPatchFacesOriginal.append(pF.faceNormal)

            normalsPatchFacesNoisy = np.asarray(normalsPatchFacesNoisy)
            normalsPatchFacesNoisy = np.transpose(normalsPatchFacesNoisy)

            normalsPatchFacesOriginal = np.asarray(normalsPatchFacesOriginal)
            normalsPatchFacesOriginal = np.transpose(normalsPatchFacesOriginal)

            #if np.size(normalsPatchFacesNoisy, axis=1) >= numOfElements:
            NormalsNoisy = np.concatenate((NormalsNoisy, normalsPatchFacesNoisy[:, 0:numOfElements]), axis=0)
            NormalsOriginal = np.concatenate((NormalsOriginal, normalsPatchFacesOriginal[:, 0:numOfElements]), axis=0)


        exportObj(mModelToProcess, rootdir+'mModelTest.obj')
        print('Time:' + str(time.time() - t))
        NormalsNoisyTest = np.concatenate((NormalsNoisyTest, NormalsNoisy[:, 0:numOfElements]), axis=0)
        NormalsOriginalTest = np.concatenate((NormalsOriginalTest, NormalsOriginal[:, 0:numOfElements]), axis=0)
    print('Process complete')

    fNoisyTest = NormalsNoisyTest
    fOrigTest = NormalsOriginalTest

    mSizeTest = int((np.size(fNoisyTest, axis=0) / 3))
    for i in range(0, mSizeTest):
        tStartTest = i * 3
        tEndTest = i * 3 + 3
        toAppendΟTest = fOrigTest[tStartTest:tEndTest, 0:numOfElements]
        toAppendΟTest = np.transpose(toAppendΟTest)
        toAppendΟTest = (toAppendΟTest + 1.0 * np.ones(np.shape(toAppendΟTest))) / 2.0
        toAppendNTest = fNoisyTest[tStartTest:tEndTest, 0:numOfElements]
        toAppendNTest = np.transpose(toAppendNTest)
        toAppendNTest = (toAppendNTest + 1.0 * np.ones(np.shape(toAppendNTest))) / 2.0
        if True:
            test_images_original.append(toAppendΟTest)
            test_images_noisy.append(toAppendNTest)

    test_images_original = np.asarray(test_images_original)
    test_images_noisy = np.asarray(test_images_noisy)

    test_images_original = np.round(test_images_original, decimals=6)
    test_images_noisy = np.round(test_images_noisy, decimals=6)

#############################################################################
#############################################################################
#############################################################################


Geometry = collections.namedtuple("G", "vertices, normals, faces, edges, adjacency")
Vertex = collections.namedtuple("V",
                                "index,position,normal,neighbouringFaceIndices,neighbouringVerticesIndices,rotationAxis,theta")
Face = collections.namedtuple("F",
                              "index,centroid,vertices,verticesIndices,faceNormal,area,edgeIndices,neighbouringFaceIndices,guidedNormal,rotationAxis,theta")
Edge = collections.namedtuple("E", "index,vertices,verticesIndices,length,facesIndices")

# if doTrain:
#     outputFile = rootdir+'data/storeTrain' + keyTrain + '.data'
#     fw = open(outputFile, 'wb')
#     pickle.dump(train_images_original, fw)
#     pickle.dump(train_images_noisy, fw)
#     fw.close()
# else:


if True:
    inputFile = rootdir+'data/storeTrain' + keyTrain + '.data'
    f = open(inputFile, 'rb')
    train_images_original = pickle.load(f)
    train_images_noisy = pickle.load(f)

    print('Trainset')
    print(np.size(train_images_original,axis=0))
    print(np.size(train_images_noisy, axis=0))
    f.close()

if doTest:
    outputFile = rootdir+'data/storeTest' + keyTest + '.data'
    fw = open(outputFile, 'wb')
    pickle.dump(test_images_original, fw)
    pickle.dump(test_images_noisy, fw)
    pickle.dump(mModelTest, fw)
    pickle.dump(mModelToProcess, fw)
    exportObj(mModelToProcess, rootdir+'mModelTest.obj')
    fw.close()
else:
    inputFile = rootdir+'data/storeTest' + keyTest + '.data'
    f = open(inputFile, 'rb')
    test_images_original = pickle.load(f)
    test_images_noisy = pickle.load(f)
    mModelTest = pickle.load(f)
    mModelToProcess = pickle.load(f)
    exportObj(mModelToProcess, rootdir+'mModelTest.obj')
    f.close()

if modelsOnly:
    exit(0)




if True:
    mSize = int((np.size(train_images_original, axis=0)))
    mSizeTest = int((np.size(test_images_original, axis=0)))
    kMInputTrain = []
    d = []
    for i in range(0, mSize):
        d = train_images_noisy[i].ravel()
        kMInputTrain.append(d)
    Xk = tf.placeholder(tf.float32, shape=[None, len(train_images_noisy[0].ravel())])
    # K-Means Parameters
    kmeans = KMeans(inputs=Xk, num_clusters=nClusters, distance_metric=distType,
                    use_mini_batch=True)
    training_graph = kmeans.training_graph()
    if len(training_graph) > 6:  # Tensorflow 1.4+
        (all_scores, cluster_idx, scores, cluster_centers_initialized,
         cluster_centers_var, init_op, train_op) = training_graph
    else:
        (all_scores, cluster_idx, scores, cluster_centers_initialized,
         init_op, train_op) = training_graph
    cluster_idx = cluster_idx[0]  # fix for cluster_idx being a tuple
    avg_distance = tf.reduce_mean(scores)
    # Initialize the variables (i.e. assign their default value)
    init_vars2 = tf.global_variables_initializer()
    # Start TensorFlow session
    sesskmeans = tf.Session()
    # Run the initializer
    sesskmeans.run(init_vars2, feed_dict={Xk: kMInputTrain})
    sesskmeans.run(init_op, feed_dict={Xk: kMInputTrain})


    tf.train.Saver().restore(sesskmeans, rootdir+'sessions/KMeans/modelKMeans_'+str(numOfElements)+noiseLevelAsString+'.ckpt')
    kMInputTest = [];
    for i in range(0, mSizeTest):
        d = test_images_noisy[i].ravel()
        kMInputTest.append(d)
    for i in range(1, 5 + 1):
        _, d, idx = sesskmeans.run([train_op, avg_distance, cluster_idx],
                                              feed_dict={Xk: kMInputTest})
    sesskmeans.close()
    test_labels_updated = np.zeros((mSizeTest, nClusters))
    for i in range(0, mSizeTest):
        test_labels_updated[i, idx[i]] = 1.0
    test_labels_updated = np.asarray(test_labels_updated)


#############################################################################
#############################################################################
#############################################################################
#############################################################################


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
