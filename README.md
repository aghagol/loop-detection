# place-recognition
[AlexNet](https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet) feature vectors for place recognition and loop closing

It has been proposed that feature vectors from AlexNet are robust with respect to appearance changes:

N. Sunderhauf, F. Dayoub, S. Shirazi, B. Upcroft, and M. Milford, [On the performance of convnet features for place recognition](https://arxiv.org/abs/1501.04158), arXiv preprint arXiv:1501.04158, 2015.

Look in the folder `PR_bench` for our implementation of the above paper over the [Gardens Point dataset](https://wiki.qut.edu.au/display/cyphy/Day+and+Night+with+Lateral+Pose+Change+Datasets).

#### Demo:

We use feature vectors from the `conv3` layer of AlexNet to match each frame of the testing video with a frame in the reference video that is captured on a different day. Videos were captured using a Google Nexus 5 phone. GPS is recorded simultanously using a dashcam Andriod app. I put a cheap ($5) fisheye lens on Nexus 5 to increase its field of view.

 - Matching video frames from a cloudy day to video frames of a sunny day and predicting the location. On the map, big blue circle corresponds to the GPS location that is recorded over each test video frame (ground-truth) and the green star corresponds to the GPS location of the reference video frame (prediction).

<p align="center">
  <a href="https://www.youtube.com/watch?v=ESLLPeJJJ2I&feature=youtu.be" target="_blank"><img src="https://img.youtube.com/vi/ESLLPeJJJ2I/0.jpg" alt="IMAGE ALT TEXT HERE" width="480" height="360" border="10" /></a>
</p>
