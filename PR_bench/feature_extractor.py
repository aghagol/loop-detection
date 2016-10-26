import numpy as np
import os, sys, pickle
import matplotlib.pyplot as plt
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
os.environ['GLOG_minloglevel'] = '2'
import caffe
caffe_root = '/home/mo/caffe/'
caffe.set_mode_gpu()

# dataset = 'day_left'
dataset = sys.argv[1]
prepath = 'data/'+dataset+'/'
images = [f for f in os.listdir(prepath) if f.endswith('.jpg')]
images.sort()

""" load net weights and allocate memory """
model_def = '/home/mo/caffe/models/places/places205CNN_deploy.prototxt'
W = '/home/mo/caffe/models/places/places205CNN_iter_300000.caffemodel'
net = caffe.Net(model_def, W, caffe.TEST)
net.blobs['data'].reshape(len(images), 3, 227, 227)

""" pre-processing for input """
blob = caffe.proto.caffe_pb2.BlobProto()
mu_f = open(caffe_root+'models/places/places205CNN_mean.binaryproto','rb')
blob.ParseFromString(mu_f.read())
mu = np.array(caffe.io.blobproto_to_array(blob))[0]
BGRmean = mu.mean(1).mean(1)
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', BGRmean)
transformer.set_raw_scale('data', 255)
transformer.set_channel_swap('data', (2,1,0))

""" forward step """
for i in range(len(images)):
	image = caffe.io.load_image(prepath+images[i])
	net.blobs['data'].data[i] = transformer.preprocess('data', image)
print "finished loading the net... moving forward"
output = net.forward()
print "writing features to output file..."

""" write stuff """
with open('net_'+dataset+'_data','w') as fw:
	pickle.dump(net.blobs['data'].data,fw)

# with open('net_'+dataset+'_pool1','w') as fw:
# 	pickle.dump(net.blobs['pool1'].data,fw)

# with open('net_'+dataset+'_pool2','w') as fw:
# 	pickle.dump(net.blobs['pool2'].data,fw)

# with open('net_'+dataset+'_conv3','w') as fw:
# 	pickle.dump(net.blobs['conv3'].data,fw)

# with open('net_'+dataset+'_conv4','w') as fw:
# 	pickle.dump(net.blobs['conv4'].data,fw)

# with open('net_'+dataset+'_conv5','w') as fw:
# 	pickle.dump(net.blobs['conv5'].data,fw)

# with open('net_'+dataset+'_fc6','w') as fw:
# 	pickle.dump(net.blobs['fc6'].data,fw)

# with open('net_'+dataset+'_fc7','w') as fw:
# 	pickle.dump(net.blobs['fc7'].data,fw)
