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
    flag.DEFINE_float("learning_rate", 0.00002, "learning rate for training")
    flag.DEFINE_integer("n_epoch", 40, "number of Epoch")
    flag.DEFINE_integer("n_z", 2, "Dimension of latent variables")
    flag.DEFINE_float("keep_prob", 0.95, "Dropout rate")
    flag.DEFINE_float("decay_rate", 0.998, "learning rate decay rate")
    flag.DEFINE_integer("batch_size", 256, "Batch size for training")
    flag.DEFINE_bool("add_noise", False, "[True/False]")
    flag.DEFINE_bool("PMLR", False, "Boolean for plot manifold learning result")
    flag.DEFINE_bool("PARR", False, "Boolean for plot analogical reasoning result")





    # FLAGS.batch_size=128
    # FLAGS.keep_prob=0.8
    # FLAGS.learning_rate=0.00004
    # FLAGS.decay_rate=0.998
    # E1=256
    # E2=256
    # D1=256
    # D2=256

    with open('F:/_Groundwork/FastMeshDenoisingProduction/results/hopresults.json', 'w+') as writeFile:
        writeFile.write('[')
        settingID=1
        for FLAGS.batch_size in [512]:
            for FLAGS.keep_prob in [0.99]:
                for FLAGS.learning_rate in [0.00003]:
                    for FLAGS.decay_rate in [0.998]:
                        for E1 in [1024,2048]:
                            for E2 in [1024,2048,4096]:
                                for D1 in [1024,2048,4096]:
                                    for D2 in [1024,2048,4096]:
                                        tf.reset_default_graph()
                                        if settingID>1:
                                            writeFile.write(',')
                                        writeFile.write('{')
                                        writeFile.write(
                                            '\"NUMEL\":' + str(dat.numOfElements) + ','+ '\n'+
                                            '\"CL\":' + str(dat.nClusters) + ','+ '\n'+
                                            '\"BS\":' + str(FLAGS.batch_size) + ','+'\n'+
                                            '\"KP\":' + str(FLAGS.keep_prob) + ','+'\n'+
                                            '\"LR\":' + str(FLAGS.learning_rate) + ','+'\n'+
                                            '\"DR\":' + str(FLAGS.decay_rate) + ','+'\n'+
                                            '\"E1\":' + str(E1) + ','+'\n'+
                                            '\"E2\":' + str(E2) + ','+'\n'+
                                            '\"D1\":' + str(D1) + ','+'\n'+
                                            '\"D2\":' + str(D2) + ','+'\n'+
                                            '\"loss\":['
                                            )

                                        _data_pipeline = data_pipeline("Custom")
                                        train_xs, train_ys, valid_xs, valid_ys, test_xs, test_ys = _data_pipeline.load_preprocess_data()
                                        _, height, width = np.shape(train_xs)
                                        n_cls = np.shape(train_ys)[1]
                                        X = tf.placeholder(dtype=tf.float32, shape=[None, height, width], name="Input")
                                        X_noised = tf.placeholder(dtype=tf.float32, shape=[None, height, width], name="Input_noised")
                                        Y = tf.placeholder(dtype=tf.float32, shape=[None, n_cls], name="labels")
                                        keep_prob = tf.placeholder(dtype=tf.float32, name="drop_rate")

                                        _CVAE=None
                                        _CVAE = CVAE([_, height, width], n_cls, [E1, E2, D1, D2], FLAGS.n_z, keep_prob)
                                        z, output, loss = _CVAE.Conditional_Variational_AutoEncoder(X, X_noised, Y, keep_prob)
                                        latent = tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.n_z], name="latent_input")
                                        global_step = tf.Variable(0, trainable=False)
                                        if FLAGS.PMLR is True:  # code for plot manifold learning Results
                                            assert FLAGS.n_z == 2, "n_z should be 2!"
                                            images_manifold = _CVAE.conditional_bernoulli_decoder(latent, Y, keep_prob)
                                        if FLAGS.PARR is True:  # code for plot analogical reasoning result
                                            images_PARR = _CVAE.conditional_bernoulli_decoder(latent, Y, keep_prob)
                                        total_batch = _data_pipeline.get_total_batch(train_xs, FLAGS.batch_size)
                                        learning_rate_decayed = FLAGS.learning_rate * FLAGS.decay_rate ** (global_step / total_batch)
                                        _optim_op = _CVAE.optim_op(loss, learning_rate_decayed, global_step)
                                        #_optim_op=tf.compat.v1.train.AdamOptimizer(learning_rate = learning_rate_decayed).minimize(loss, global_step = global_step)

                                        # session_conf = tf.ConfigProto(
                                        #     intra_op_parallelism_threads=1,
                                        #     inter_op_parallelism_threads=1,
                                        #     device_count={'GPU': 0,"CPU": 1},
                                        #     allow_soft_placement=False
                                        # )
                                        sess = tf.Session()
                                        sess.run(tf.compat.v1.global_variables_initializer())
                                        batch_v_xs, batch_vn_xs, batch_v_ys = _data_pipeline.next_batch(valid_xs, valid_ys, 100, make_noise=FLAGS.add_noise)
                                        print("_" * 80)
                                        start_time = time.time()
                                        print("training started")
                                        for i in range(FLAGS.n_epoch):
                                            loss_val = 0
                                            for j in range(total_batch):
                                                batch_xs, batch_noised_xs, batch_ys = _data_pipeline.next_batch(train_xs, train_ys, FLAGS.batch_size,
                                                                                                               make_noise=False)
                                                feed_dict = {X: batch_xs, X_noised: batch_noised_xs, Y: batch_ys, keep_prob: FLAGS.keep_prob}
                                                l, lr, op, g = sess.run([loss, learning_rate_decayed, _optim_op, global_step], feed_dict=feed_dict)
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


                                                # mModelToProcessDenoised = copy.deepcopy(mModelToProcess)
                                                # if doRotate:
                                                #     for r in range(0, np.size(result, axis=0)):
                                                #         result[r, :] = rotate(result[r, :], mModelToProcessDenoised.faces[r].rotationAxis,
                                                #                               -mModelToProcessDenoised.faces[r].theta)
                                                # updateVerticesWithNormals(mModelToProcessDenoised, result, 20)
                                                # exportObj(mModelToProcessDenoised, dat.rootdir+'meshes/denoised'+
                                                #           '_NUMEL' + str(dat.numOfElements) +
                                                #           '_CL' + str(dat.nClusters) +
                                                #           '_BS' + str(FLAGS.batch_size) +
                                                #           '_KP' + str(FLAGS.keep_prob) +
                                                #           '_LR'+str(FLAGS.learning_rate)+
                                                #           '_DR'+ str(FLAGS.decay_rate) +
                                                #           '_D1' + str(D1) +
                                                #           '_D2' + str(D2) +
                                                #           '_E1' + str(E1) +
                                                #           '_E2' + str(E2) +
                                                #           '_' + '.obj')



                                            writeFile.write(str(loss_val))
                                            if i!=(FLAGS.n_epoch-1):
                                                writeFile.write(',')
                                            hour = int((time.time() - start_time) / 3600)
                                            min = int(((time.time() - start_time) - 3600 * hour) / 60)
                                            sec = int((time.time() - start_time) - 3600 * hour - 60 * min)
                                            print("Epoch: %.3d   loss: %.5f   lr: %f   Time: %d hour %d min %d sec\n" % (i, loss_val, lr, hour, min, sec))
                                        sess.close()
                                        writeFile.write(']')
                                        writeFile.write('}')
                                        settingID=settingID+1

        writeFile.write(']')



