import tensorflow as tf
import numpy as np
import fastMeshDenoising_Data_Utils_Train as dat
from fastMeshDenoising_Data_Utils_Train import *
from CVAEutils import *
from CVAEplot import *
from CVAE import *

import time
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"
#os.environ["OMP_NUM_THREADS"] = "1"

if __name__ == "__main__":
    flag = tf.app.flags
    FLAGS = flag.FLAGS
    flag.DEFINE_float("learning_rate", 0.00003, "learning rate for training")
    flag.DEFINE_integer("n_epoch", 30, "number of Epoch")
    flag.DEFINE_integer("n_z", 2, "Dimension of latent variables")
    flag.DEFINE_float("keep_prob", 0.99, "Dropout rate")
    flag.DEFINE_float("decay_rate", 0.998, "learning rate decay rate")
    flag.DEFINE_integer("batch_size", 256, "Batch size for training")
    flag.DEFINE_bool("add_noise", False, "[True/False]")
    flag.DEFINE_bool("PMLR", False, "Boolean for plot manifold learning result")
    flag.DEFINE_bool("PARR", False, "Boolean for plot analogical reasoning result")
    data_pipeline = data_pipeline("Custom")
    train_xs, train_ys, valid_xs, valid_ys, test_xs, test_ys = data_pipeline.load_preprocess_data()
    _, height, width = np.shape(train_xs)
    n_cls = np.shape(train_ys)[1]
    X = tf.placeholder(dtype=tf.float32, shape=[None, height, width], name="Input")
    X_noised = tf.placeholder(dtype=tf.float32, shape=[None, height, width], name="Input_noised")
    Y = tf.placeholder(dtype=tf.float32, shape=[None, n_cls], name="labels")
    keep_prob = tf.placeholder(dtype=tf.float32, name="drop_rate")
    CVAE = CVAE([_, height, width], n_cls, [1024, 2048, 4096, 4096], FLAGS.n_z, keep_prob)
    z, output, loss = CVAE.Conditional_Variational_AutoEncoder(X, X_noised, Y, keep_prob)
    latent = tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.n_z], name="latent_input")
    global_step = tf.Variable(0, trainable=False)
    if FLAGS.PMLR is True:  # code for plot manifold learning Results
        assert FLAGS.n_z == 2, "n_z should be 2!"
        images_manifold = CVAE.conditional_bernoulli_decoder(latent, Y, keep_prob)
    if FLAGS.PARR is True:  # code for plot analogical reasoning result
        images_PARR = CVAE.conditional_bernoulli_decoder(latent, Y, keep_prob)
    total_batch = data_pipeline.get_total_batch(train_xs, FLAGS.batch_size)
    learning_rate_decayed = FLAGS.learning_rate * FLAGS.decay_rate ** (global_step / total_batch)
    optim_op = CVAE.optim_op(loss, learning_rate_decayed, global_step)

    # session_conf = tf.ConfigProto(
    #     intra_op_parallelism_threads=1,
    #     inter_op_parallelism_threads=1,
    #     device_count={'GPU': 0,"CPU": 1},
    #     allow_soft_placement=False
    # )
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    batch_v_xs, batch_vn_xs, batch_v_ys = data_pipeline.next_batch(valid_xs, valid_ys, 100, make_noise=FLAGS.add_noise)
    print("_" * 80)
    start_time = time.time()
    print("training started")
    for i in range(FLAGS.n_epoch):
        loss_val = 0
        for j in range(total_batch):
            batch_xs, batch_noised_xs, batch_ys = data_pipeline.next_batch(train_xs, train_ys, FLAGS.batch_size,
                                                                           make_noise=False)
            feed_dict = {X: batch_xs, X_noised: batch_noised_xs, Y: batch_ys, keep_prob: FLAGS.keep_prob}
            l, lr, op, g = sess.run([loss, learning_rate_decayed, optim_op, global_step], feed_dict=feed_dict)
            loss_val += l / total_batch
        if not math.isnan(loss_val) and (i % 2 == 0 or i == (FLAGS.n_epoch - 1)):
            images = sess.run(output, feed_dict={X: test_images_original,
                                                 X_noised: test_images_noisy,
                                                 Y: test_labels_updated,
                                                 keep_prob: 1.0})
            result = images[:, 0:3]
            # Convert definition space
            result = 2.0 * result - 1.0 * np.ones(np.shape(result))
            result = np.asarray(result)
            mModelToProcessDenoised = copy.deepcopy(mModelToProcess)
            if doRotate:
                for r in range(0, np.size(result, axis=0)):
                    result[r, :] = rotate(result[r, :], mModelToProcessDenoised.faces[r].rotationAxis,
                                          -mModelToProcessDenoised.faces[r].theta)
            updateVerticesWithNormals(mModelToProcessDenoised, result, 20)
            exportObj(mModelToProcessDenoised, dat.rootdir+'meshes/denoised_'+str(dat.numOfElements)+'_'+ str(dat.nClusters) +'_' + '.obj')
            #exportObj(mModelToProcessDenoised, dat.rootdir + 'Results-0/Comparison/Denoised/CVAE/'+str(dat.modelNameNoisy)+'_'+str(dat.numOfElements)+ '.obj')


            #+ str(i)
            #with open(dat.rootdir+'Results/result_' + str(i) + 'C.csv',
            with open(dat.rootdir+'meshes/result_' +str(dat.numOfElements)+'_'+ str(dat.nClusters) + '.csv',
                      'w') as writeFile:
                for j in range(0, np.size(result, axis=0)):
                    line = str(result[j, 0]) + "," + str(result[j, 1]) + "," + str(result[j, 2])
                    writeFile.write(line)
                    writeFile.write('\n')
        hour = int((time.time() - start_time) / 3600)
        min = int(((time.time() - start_time) - 3600 * hour) / 60)
        sec = int((time.time() - start_time) - 3600 * hour - 60 * min)
        print("Epoch: %.3d   loss: %.5f   lr: %f   Time: %d hour %d min %d sec\n" % (i, loss_val, lr, hour, min, sec))
        saver = tf.train.Saver()
        save_path = saver.save(sess,
                               dat.rootdir+'sessions/CVAE/model_' + str(numOfElements)+noiseLevelAsString+ '.ckpt')

    #Exports
    # exportObj(mModelToProcess,
    #           dat.rootdir+'meshes/Denoised/CVAE/' + dat._modelName + '_noisy' + noiseLevelAsString +  '.obj')
    #
    # exportObj(mModelToProcessDenoised,
    #           dat.rootdir+'meshes/Denoised/CVAE/' + dat._modelName + noiseLevelAsString + '_' + str(
    #               dat.numOfElements) + '.obj')
    #
    # with open(
    #         dat.rootdir+'meshes/Denoised/CVAE/' + dat._modelName + '_normals' + noiseLevelAsString + '_' + str(
    #                 dat.numOfElements) + '.csv', 'w') as writeFile:
    #     for j in range(0, np.size(result, axis=0)):
    #         line = str(result[j, 0]) + "," + str(result[j, 1]) + "," + str(result[j, 2])
    #         writeFile.write(line)
    #         writeFile.write('\n')

    sess.close()
