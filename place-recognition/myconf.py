#path to landmarks directory
lmpath = '../cam/vid_3_28/'

#path to background map directory
mappath = '../localize/'

#path to camera data directory
campath = '../cam/still_3_28/'

#path to features directory for test data
featpath = './features/'

#path to features directory for landmarks
lmfeatpath = './'

#output directory
outpath = './out/'

#fish-eye crop ( ( row1 , row2 ) , ( col1 , col2 ) )
febb_stl = ( ( 250 , 2250 ) , ( 600 , 2850 ) )
febb_vid = ( (  50 , 450  ) , ( 100 ,  550 ) )

#scales at which patches are extracted
scales = [.5, .65, .8, .9]
