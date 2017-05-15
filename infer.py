import numpy as np
from PIL import Image
import cPickle as pickle
import caffe
import os, sys


# load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
direct = sys.path[0]
direct_2 = '/img/'
im = Image.open(direct + direct_2 + 'img.jpg')
in_ = np.array(im, dtype=np.float32)
in_ = in_[:,:,::-1]
in_ -= np.array((104.00698793,116.66876762,122.67891434))
in_ = in_.transpose((2,0,1))

# load net
net = caffe.Net('voc-fcn8s/deploy.prototxt', 'voc-fcn8s/fcn8s-heavy-pascal.caffemodel', caffe.TEST)
# shape for input (data blob is N x C x H x W), set data
net.blobs['data'].reshape(1, *in_.shape)
net.blobs['data'].data[...] = in_
# run net and take argmax for prediction
net.forward()
out = net.blobs['score'].data[0].argmax(axis=0) 
pickle.dump(out,open("out.pkl",'wb'))
