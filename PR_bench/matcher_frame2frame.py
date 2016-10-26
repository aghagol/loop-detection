import pickle
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10, 10) 
plt.rcParams['image.interpolation'] = 'nearest'

with open('D_nr_dr') as f:
	D_nr_dr = pickle.load(f)

with open('D_dl_dr') as f:
	D_dl_dr = pickle.load(f)

with open('D_nr_dl') as f:
	D_nr_dl = pickle.load(f)

with open('matching_with_appearance_changes.txt','w') as f:
	f.write('night right --> day right\n')
	for row in range(D_nr_dr.shape[0]):
		col = np.argsort(D_nr_dr[row])[-1]
		if row in np.argsort(D_nr_dr.T[col])[-3:]:
			f.write('frame %03d --> frame %03d\n' % (row,col))
		else:
			f.write('frame %03d --> no match\n' % (row))

with open('matching_with_view_changes.txt','w') as f:
	f.write('day left --> day right\n')
	for row in range(D_dl_dr.shape[0]):
		col = np.argsort(D_dl_dr[row])[-1]
		if row in np.argsort(D_dl_dr.T[col])[-3:]:
			f.write('frame %03d --> frame %03d\n' % (row,col))
		else:
			f.write('frame %03d --> no match\n' % (row))

with open('matching_with_appearance_and_view_changes.txt','w') as f:
	f.write('night right --> day left\n')
	for row in range(D_nr_dl.shape[0]):
		col = np.argsort(D_nr_dl[row])[-1]
		if row in np.argsort(D_nr_dl.T[col])[-3:]:
			f.write('frame %03d --> frame %03d\n' % (row,col))
		else:
			f.write('frame %03d --> no match\n' % (row))
