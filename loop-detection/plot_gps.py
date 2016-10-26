import os
import matplotlib.pyplot as plt

def make_tree(C,T, parent_depth):
	if not C: return (T,C)
	elif C[0]['depth'] <= parent_depth:
		return (T,C)
	elif len(C)==1: #last entry
		T[C[0]['key']] = C[0]['val']
		return (T,C[1:])
	elif C[0]['depth'] > C[1]['depth']: #step out
		T[C[0]['key']] = C[0]['val']
		return (T,C[1:])
	elif C[0]['depth'] == C[1]['depth']: #stay
		T[C[0]['key']] = C[0]['val']
		return make_tree(C[1:],T, parent_depth)
	elif C[0]['depth'] <  C[1]['depth']: #step in
		T[C[0]['key']],C = make_tree(C[1:],{},C[0]['depth'])
		T,C = make_tree(C,T, parent_depth)
	return (T,C)

datapath = '../Soonhac/left'
gps_files = [f for f in os.listdir(datapath) if f.endswith('.gps')]
gps_files.sort()

T = {}
for gpsf in gps_files:
	C = []
	with open(datapath+'/'+gpsf) as f:
		for ln in f:
			s = dict(zip(['key','val'],[i.strip() for i in ln.split(':')]))
			s['depth'] = len(ln)-len(ln.lstrip())
			C.append(s)
	T[gpsf[:-4]] = make_tree(C,{},-1)[0]

fig, ax = plt.subplots(1,1)
ax.set_xlim([-20,70])
ax.set_ylim([-20,70])
for gpsf in gps_files:
	x = float(T[gpsf[:-4]]['pose']['pose']['position']['x'])
	y = float(T[gpsf[:-4]]['pose']['pose']['position']['y'])
	ax.plot(x,y,'go')
	plt.pause(.1)


