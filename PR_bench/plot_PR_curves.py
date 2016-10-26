import pickle
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10, 10) 
plt.rcParams['image.interpolation'] = 'nearest'

e = 3 #true-positive region: +/-e frames
r_range = np.arange(1,2,.001) #threshold for PR curve

# layers = ['data','pool1','pool2','conv3','conv4','conv5','fc6','fc7']
layers = ['data']
for layer in layers:	

	with open('D_nr_dr_'+layer) as f:
		D_nr_dr = pickle.load(f)

	with open('D_dl_dr_'+layer) as f:
		D_dl_dr = pickle.load(f)

	with open('D_nr_dl_'+layer) as f:
		D_nr_dl = pickle.load(f)

	n = D_nr_dr.shape[0]
	AUC = []
	for D in [D_nr_dr,D_dl_dr,D_nr_dl]:
		TP = np.zeros_like(r_range)
		FP = np.zeros_like(r_range)
		FN = np.zeros_like(r_range)
		for i in range(len(r_range)):
			r = r_range[i]
			for row in range(n):
				col2, col1 = np.argsort(D[row])[-2:]
				if ( D[row,col1] / D[row,col2] ) >= r:
					if (row <= col1+e) and (row >= col1-e):
						TP[i] +=1
					else:
						FP[i] +=1
				else:
					FN[i] +=1
		IX = (TP+FP) > 0 #retain only the valid indices
		R = TP[IX] / ( TP[IX] + FN[IX] )
		P = TP[IX] / ( TP[IX] + FP[IX] )
		AUC.append(np.trapz(P[::-1],R[::-1]))
		plt.plot(R, P)
		# F1 = 2 * (P*R) / (P+R)

	labels = [	'AUC=%.3f (night-right vs day-right)' 	% AUC[0],
				'AUC=%.3f (day-left vs day-right)' 		% AUC[1],
				'AUC=%.3f (night-right vs day-left)' 	% AUC[2]	]

	p = np.arange(.001,1,.001)
	for f in np.arange(.1,1,.1):
		r = np.array([f*v/(2*v-f) if 2*v!=f else -1 for v in p])
		p_cut = p[np.logical_and(r>=0,r<=1)]
		r_cut = r[np.logical_and(r>=0,r<=1)]
		plt.plot(r_cut, p_cut, "--", color='gray')
		plt.annotate(r"$F_1=%.1f$" % f, xy=(r_cut[0], p_cut[0]),
			xytext=(.9, p_cut[0]), size="small", color="gray")

	plt.legend(labels, loc='lower left')
	plt.xlim([-0.01,1.01])
	plt.xlabel('Recall')
	plt.ylim([-0.01,1.01])
	plt.ylabel('Precision')
	plt.title('Precision-Recall curve for layer: {0}'.format(layer))
	plt.savefig('PR_'+layer+'.png', bbox_inches='tight')
	plt.show()















# plt.plot(FP.flatten()/n,TP.flatten()/n)
# plt.xlim([-0.01,1.01])
# plt.xlabel('False Positive')
# plt.ylim([-0.01,1.01])
# plt.ylabel('True Positive')
# plt.title('ROC curve')
# plt.show()
