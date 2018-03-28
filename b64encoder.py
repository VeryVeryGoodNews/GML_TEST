
# coding: utf-8

# In[ ]:

import numpy as np
from scipy.misc import imread, imsave, imresize
import scipy.ndimage
import matplotlib
matplotlib.use('TkAgg') # choose appropriate rendering backend
from matplotlib import pyplot as plt
import base64

img = imread('0.jpg')

img = imresize(img,(224,224))

#b64str = base64.b64encode(img)
b64str = base64.b64encode(img,'-_')

np.save('0.jpg',b64str)

print b64str

