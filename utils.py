import glob
import numpy as np
from PIL import Image
import scipy

def load_data_gray():
    filelist = glob.glob('Pokemon_Grey/*.png')
    x = np.array([np.array(Image.open(fname)) for fname in filelist])
    #x = np.array([np.array(scipy.misc.imresize(Image.open(fname), (28,28), interp='bilinear', mode=None)) for fname in filelist])
    print(x.shape)
    return x

def load_data():
    filelist = glob.glob('Pokemon/*.jpg')
    x = np.array([np.array(Image.open(fname)) for fname in filelist])
    #x = np.array([np.array(scipy.misc.imresize(Image.open(fname), (28,28), interp='bilinear', mode=None)) for fname in filelist])
    print(x.shape)
    return x

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))

    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        #print(j,h,i,w)
        img[j*h:j*h+h, i*w:i*w+w, :] = image

    return img
