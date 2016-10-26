import numpy as np
from PIL import Image
import os, sys, pickle, re, time
import matplotlib.pyplot as plt
os.environ['GLOG_minloglevel'] = '2'
caffe_root = '/home/mo/caffe/'
sys.path.append(caffe_root+'python')
import caffe
caffe.set_mode_gpu()
from myconf import *

def bboxes(imsize):
	bbs = []
	# global scales
	for scale in scales:
		rows_l 	= (0,int(scale*imsize[0]))
		rows_c 	= (int((1-scale)*imsize[0]/2),int((1+scale)*imsize[0]/2))
		rows_r 	= (int((1-scale)*imsize[0]),imsize[0])
		cols_l 	= (0,int(scale*imsize[1]))
		cols_c 	= (int((1-scale)*imsize[1]/2),int((1+scale)*imsize[1]/2))
		cols_r 	= (int((1-scale)*imsize[1]),imsize[1])
		bbs.append( ( rows_l , cols_l ) )
		bbs.append( ( rows_l , cols_c ) )
		bbs.append( ( rows_l , cols_r ) )
		bbs.append( ( rows_c , cols_l ) )
		bbs.append( ( rows_c , cols_c ) )
		bbs.append( ( rows_c , cols_r ) )
		bbs.append( ( rows_r , cols_l ) )
		bbs.append( ( rows_r , cols_c ) )
		bbs.append( ( rows_r , cols_r ) )
	bbs.append( ( (0,imsize[0]) , (0,imsize[1]) ) )
	return bbs

if not os.path.exists(featpath): os.mkdir(featpath)
# layers = ['data','pool1','pool2','conv3','conv4','conv5','fc6','fc7']
layers = ['conv3','fc6']
K = len(bboxes((100,100))) #number of patches per image
batch_size = K /K

images = [f for f in os.listdir(campath) if f.endswith('.png')]
images.sort()
N = len(images)

""" load net weights and allocate memory """
model_def = '/home/mo/caffe/models/places/places205CNN_deploy.prototxt'
W = '/home/mo/caffe/models/places/places205CNN_iter_300000.caffemodel'
net = caffe.Net(model_def, W, caffe.TEST)
net.blobs['data'].reshape(batch_size*K, 3, 227, 227)

""" pre-processing for input """
blob = caffe.proto.caffe_pb2.BlobProto()
mu_f = open(caffe_root+'models/places/places205CNN_mean.binaryproto','rb')
blob.ParseFromString(mu_f.read())

""" forward step """
for i in range(N / batch_size + 1):
	for j in range(batch_size):
		n = batch_size*i + j
		if (n>=N): break
		print 'CPU: working on %s (%.2f%%)'%(images[n],(n+1.)/N*100)
		image = Image.open(campath+images[n])
		bbs = bboxes(image.size[::-1])
		with open(featpath+images[n][:-4]+'.meta','wb') as fbb:
			pickle.dump(bbs,fbb)
		for k, bb in enumerate(bbs):
			print '...patch number %d'%(k+j*K)
			patch = image.crop((bb[1][0],bb[0][0],bb[1][1],bb[0][1]))
			patch = patch.resize((227, 227), Image.ANTIALIAS)
			patch = np.asarray(patch)
			patch = patch-patch.mean()
			patch = patch.reshape(1,227,227)
			patch = np.concatenate((patch,)*3,axis=0)
			net.blobs['data'].data[k+j*K] = patch
	print 'Titan enters'
	t1 = time.time()
	output = net.forward()
	t2 = time.time()
	print 'Titan leaves (%f ms / patch)' %((t2-t1)*1000/batch_size/K)
	print 'CPU: writing to disk'
	for j in range(batch_size):
		n = batch_size*i + j
		if (n>=N): break
		for layer in layers:
			with open(featpath+images[n][:-4]+'.'+layer,'wb') as fw:
				pickle.dump(net.blobs[layer].data[range(j*K,j*K+K)],fw)
