#path to landmarks directory
lmpath = '../Soonhac/left/'

#path to background map directory
mappath = '../localize/'

#path to camera data directory
campath = '../Soonhac/left/'

#path to features directory for test data
featpath = './features/'

#path to features directory for landmarks
lmfeatpath = './'

#output directory
outpath = './out/'

# #fish-eye crop ( ( row1 , row2 ) , ( col1 , col2 ) )
febb_stl = ( ( 250 , 2250 ) , ( 600 , 2850 ) )
febb_vid = ( (  50 , 450  ) , ( 100 ,  550 ) )

#scales at which patches are extracted
scales = [.8]
