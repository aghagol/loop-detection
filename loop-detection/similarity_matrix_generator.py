import pickle, os
import numpy as np
import matplotlib.pyplot as plt
from myconf import *

layers = ['fc6']

images = [f for f in os.listdir(campath) if f.endswith('.png')]
images.sort()
N = len(images)

for layer in layers:
	with open(featpath+images[0][:-4]+'.'+layer,'rb') as fr:
		X = pickle.load(fr)
	X = X.reshape((X.shape[0],-1)).T
	X /= np.sqrt(np.power(X,2).sum(axis=0))
	for i in range(len(images)):
		print 'working on image %s (%2.2f%%)' %(images[i],i*100./len(images))
		pd = []
		with open(featpath+images[i][:-4]+'.'+layer,'rb') as fr:
			Y = pickle.load(fr)
		Y = Y.reshape((Y.shape[0],-1)).T
		Y /= np.sqrt(np.power(Y,2).sum(axis=0))
		# for j in range(i):
		# 	with open(featpath+images[j][:-4]+'.'+layer,'rb') as fr:
		# 		X = pickle.load(fr)
		# 	X = X.reshape((X.shape[0],-1)).T
		# 	X /= np.sqrt(np.power(X,2).sum(axis=0))
		D = (Y.T).dot(X)
		# pd.append((D.max(),D.argmax()))
		with open(featpath+images[i][:-4]+'.'+layer+'.D','wb') as fw:
			pickle.dump(D,fw)
