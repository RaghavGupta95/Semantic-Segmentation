import infer
from matplotlib import pyplot as plt
import numpy as np
import cPickle as pickle
import PIL
import os

img = pickle.load(open("out.pkl",'rb'))
print np.unique(img)
class_names = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
'diningtable', 'dog', 'horse', 'motorbike', 'person',
'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
print np.array(class_names)[np.unique(img)]
image = plt.imshow(img)
plt.show()
plt.draw()
plt.imshow(PIL.Image.open("train.jpg"))
