import pickle
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10, 10) 
plt.rcParams['image.interpolation'] = 'nearest'

# layers = ['data','pool1','pool2','conv3','conv4','conv5','fc6','fc7']
layers = ['data']
for layer in layers:
	
	with open('net_day_left_'+layer) as fr:
		Xdl = pickle.load(fr).reshape((200,-1))

	with open('net_day_right_'+layer) as fr:
		Xdr = pickle.load(fr).reshape((200,-1))

	with open('net_night_right_'+layer) as fr:
		Xnr = pickle.load(fr).reshape((200,-1))

	Xdl /= np.sqrt(np.sum(np.power(Xdl,2),axis=1)).reshape(200,1)
	Xdr /= np.sqrt(np.sum(np.power(Xdr,2),axis=1)).reshape(200,1)
	Xnr /= np.sqrt(np.sum(np.power(Xnr,2),axis=1)).reshape(200,1)

	D_nr_dr = Xnr.dot(Xdr.T)
	with open('D_nr_dr_'+layer, 'w') as fw:
		pickle.dump(D_nr_dr,fw)
	# plt.imshow(D_nr_dr)
	# plt.colorbar()
	# plt.title('appearance changes (night right vs day right)')
	# plt.savefig('conv3_cosine_nr_dr.png', bbox_inches='tight')
	# plt.show()

	D_dl_dr = Xdl.dot(Xdr.T)
	with open('D_dl_dr_'+layer, 'w') as fw:
		pickle.dump(D_dl_dr,fw)
	# plt.imshow(D_dl_dr)
	# plt.colorbar()
	# plt.title('viewpoint changes (day left vs day right)')
	# plt.savefig('conv3_cosine_dl_dr.png', bbox_inches='tight')
	# plt.show()

	D_nr_dl = Xnr.dot(Xdl.T)
	with open('D_nr_dl_'+layer, 'w') as fw:
		pickle.dump(D_nr_dl,fw)
	# plt.imshow(D_nr_dl)
	# plt.colorbar()
	# plt.title('appearance+view changes (night right vs day left)')
	# plt.savefig('conv3_cosine_nr_dl.png', bbox_inches='tight')
	# plt.show()



