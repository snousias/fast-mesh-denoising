import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
basefolder='E:/_Groundwork/_Python/CVAE'

def plot_2d_scatter(x,y,test_labels):
    plt.figure(figsize = (8,6))
    plt.scatter(x,y, c = np.argmax(test_labels,1), marker ='.', edgecolor = 'none', cmap = discrete_cmap('jet'))
    plt.colorbar()
    plt.grid()
    if not tf.gfile.Exists(basefolder+'./Scatter'):
        tf.gfile.MakeDirs(basefolder+'./Scatter')
    plt.savefig(basefolder+'./Scatter/2D_latent_space.png')
    plt.close()

def discrete_cmap(base_cmap =None):
    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0,1,10))
    cmap_name = base.name + str(10)
    return base.from_list(cmap_name,color_list,10)

def plot_manifold_canvas(images, n, type, name):
    assert images.shape[0] == n**2, "n**2 should be number of images"
    height = images.shape[1]
    width = images.shape[2] # width = height
    x = np.linspace(-2, 2, n)
    y = np.linspace(-2, 2, n)

    if type is "MNIST":
        canvas = np.empty((n * height, n * height))
        for i, yi in enumerate(x):
            for j, xi in enumerate(y):
                canvas[height*i: height*i + height, width*j: width*j + width] = np.reshape(images[n*i + j], [height, width])
        plt.figure(figsize=(8, 8))
        plt.imshow(canvas, cmap="gray")
    else:
        canvas = np.empty((n * height, n * height, 3))
        for i, yi in enumerate(x):
            for j, xi in enumerate(y):
                canvas[height*i: height*i + height, width*j: width*j + width,:] = images[n*i + j]
        plt.figure(figsize=(8, 8))
        plt.imshow(canvas)

    if not tf.gfile.Exists(basefolder+'./plot'):
        tf.gfile.MakeDirs(basefolder+'./plot')
    if not tf.gfile.Exists(basefolder+'./plot/PMLR'):
        tf.gfile.MakeDirs(basefolder+'./plot/PMLR')
    if not tf.gfile.Exists(basefolder+'./plot/PARR'):
        tf.gfile.MakeDirs(basefolder+'./plot/PARR')

    name = name + ".png"
    path = os.path.join(basefolder+'./plot', name)
    plt.savefig(path)
    print("saving location: %s" % (path))
    plt.close()