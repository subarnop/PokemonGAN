import glob
import numpy as np
from PIL import Image
import scipy

def load_data():
    filelist = glob.glob('Pokemon_Grey/*.png')
    x = np.array([np.array(scipy.misc.imresize(Image.open(fname), (28,28), interp='bilinear', mode=None)) for fname in filelist])
    print(x.shape)

    return x
