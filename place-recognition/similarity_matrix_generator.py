import pickle, os
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10, 10) 
plt.rcParams['image.interpolation'] = 'nearest'
from myconf import *

camsub = 'up' #subpath for lm data
# layers = ['data','pool1','pool2','conv3','conv4','conv5','fc6','fc7']
layers = ['conv3']

images = [f for f in os.listdir(campath) if f.endswith('.jpg')]
images.sort()
N = len(images)

for layer in layers:
	
	with open(lmfeatpath+'lmfeat.'+camsub+'.'+layer) as fr:
		X = pickle.load(fr)
	X = X.reshape((X.shape[0],-1)).T
	X /= np.sqrt(np.power(X,2).sum(axis=0))

	for image in images:
		with open(featpath+image[:-4]+'.'+layer,'rb') as fr:
			Y = pickle.load(fr)
		Y = Y.reshape((Y.shape[0],-1)).T
		Y /= np.sqrt(np.power(Y,2).sum(axis=0))
		D = (Y.T).dot(X)
		with open(featpath+image[:-4]+'.'+layer+'.'+camsub,'wb') as fw:
			pickle.dump(D,fw)
