import pickle, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle
import PIL.Image
from operator import itemgetter
import GPX
from myconf import *

layer = 'conv3'

camsub = ['up','dn'] #subpath for lm data
gps = {}
for csub in camsub:
	gps[csub] = GPX.read_from_srt(lmpath+csub+'/'+'geotag')

def deg2dec(d): #GPS: degree to decimal convertor
	return 1.*d[0][0]/d[0][1]+1.*d[1][0]/d[1][1]/60+1.*d[2][0]/d[2][1]/3600

def coor(bb): #extract coordinates for patch bbox
	cors = ( bb[0][0] , bb[1][0] )
	rows = bb[0][1] - bb[0][0]
	cols = bb[1][1] - bb[1][0]
	return (cors,rows,cols)

background = mpimg.imread(mappath+'map.jpg')
bb = [45.591834,45.595397,-122.462629,-122.456612]
nr, nc, n3 = background.shape

pano = {} #dictionary containing test images' data and exif
dist = {} #dictionary of similarity matrices for each test image
mach = {} #dictionary of matches for each test image
meta = {} #dictionary of metadata for each test image (patches, ...)

fpoints = open('points','r')

fig1, ax1 = plt.subplots(5,8,figsize=(20,11))
if not os.path.exists(outpath): os.mkdir(outpath)
for line_num, line in enumerate(fpoints):

	# if line_num<9: continue
	# if line_num>0: break

	if line_num==6: continue #point has missing views
	if line_num==7: continue #point has missing views

	fig1.suptitle('Test point # %02d' %(line_num+1))
	map(lambda x: x.cla(), ax1.flatten().tolist())
	# map(lambda x: x.axis('off'), ax1.flatten().tolist())
	ax1[0,0].set_ylabel('Test Image')
	ax1[1,0].set_ylabel('gr=landmark, bl=test')
	ax1[2,0].set_ylabel('Landmark')
	ax1[3,0].set_ylabel('Landmark (patch)')
	ax1[4,0].set_ylabel('Test (patch)')


	images = [line.split()[i] for i in [6,7,0,1,2,3,4,5]]

	dist.clear()
	pano.clear()
	x_gt = [] #ground-truth GPS (longitude) from exif
	y_gt = [] #ground-truth GPS (latitude) from exif

	for idx,image in enumerate(images):

		pano[image] = PIL.Image.open(campath+image+'.jpg')
		# imsize = pano[image].size[::-1]
		ax1[0,idx].imshow(np.array(pano[image]))
		if idx==2: ax1[0,idx].set_title('center (up)')
		if idx==6: ax1[0,idx].set_title('center (dn)')
		lat = deg2dec(pano[image]._getexif()[34853][2])
		lon = deg2dec(pano[image]._getexif()[34853][4])*-1

		with open(featpath+image+'.meta','rb') as fmeta:
			meta[image] = pickle.load(fmeta)

		for csub in camsub:
			with open(featpath+image+'.'+layer+'.'+csub,'rb') as fd:
				dist.setdefault(image,{})[csub] = pickle.load(fd)

		for csub in camsub:
			cols = dist[image][csub].argsort()
			d1 = dist[image][csub][tuple(range(cols.shape[0])),tuple(cols[:,-1])]
			d2 = dist[image][csub][tuple(range(cols.shape[0])),tuple(cols[:,-2])]
			mach.setdefault(image,[]).append((
				csub, 
				(d1.max() / d2[d1.argmax()])>1.05, 
				cols[d1.argmax(),-1], 
				d1.max(), 
				d1.argmax() ))
		
		match = max(mach[image],key=itemgetter(3))

		ax1[1,idx].imshow(background)
		ax1[1,idx].set_xlim([0, nc-1])
		ax1[1,idx].set_ylim([nr-1, 0])
		ax1[1,idx].set_title(str(match[3]))

		x_gt.append(0  + (lon-bb[2])/(bb[3]-bb[2])*nc)
		y_gt.append(nr - (lat-bb[0])/(bb[1]-bb[0])*nr)
		ax1[1,idx].plot(x_gt[idx],y_gt[idx],'bo')

		x1 = 0  + (gps[match[0]][match[2]].lon-bb[2])/(bb[3]-bb[2])*nc
		y1 = nr - (gps[match[0]][match[2]].lat-bb[0])/(bb[1]-bb[0])*nr

		if True: #match[1]:

			ax1[1,idx].plot(x1,y1,'go')
			frame = mpimg.imread(lmpath+match[0]+'/out%03d.png'%(match[2]+1))
			ax1[2,idx].imshow(frame)
			ax1[2,idx].set_title(match[0])
			ax1[2,idx].add_patch(Rectangle((febb_vid[1][0],febb_vid[0][0]), 
				febb_vid[1][1]-febb_vid[1][0],
				febb_vid[0][1]-febb_vid[0][0],
				fill=False, linewidth=1, edgecolor="red"))
			ax1[3,idx].imshow(frame[
				febb_vid[0][0]:febb_vid[0][1],
				febb_vid[1][0]:febb_vid[1][1],:])

			bbb = meta[image][match[4]]
			bbcors, bbrows, bbcols = coor(bbb)
			bbcors = (bbcors[0]+febb_stl[0][0], bbcors[1]+febb_stl[1][0])
			ax1[0,idx].add_patch(Rectangle(bbcors[::-1], bbcols, bbrows,
				fill=False, linewidth=1, edgecolor="red"))
			ax1[4,idx].imshow(np.array(pano[image])[
				bbb[0][0]+febb_stl[0][0]:bbb[0][1]+febb_stl[0][0],
				bbb[1][0]+febb_stl[1][0]:bbb[1][1]+febb_stl[1][0],:])

	fig1.tight_layout()
	plt.pause(1)
	plt.savefig(outpath+'point%02d.png'%(line_num+1))

fpoints.close()