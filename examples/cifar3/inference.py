import caffe
import lmdb
import numpy as np
from caffe.proto import caffe_pb2
import scipy
from scipy import io
import csv


mean_blob = caffe_pb2.BlobProto()
with open('/Users/riya/caffe/examples/cifar3/mean.binaryproto') as f:
    mean_blob.ParseFromString(f.read())

mean_array = np.asarray(mean_blob.data, dtype=np.uint8).reshape(
    (mean_blob.channels, mean_blob.height, mean_blob.width))

print mean_array.shape


net = caffe.Net('/Users/riya/caffe/examples/cifar3/deploy.prototxt', '/Users/riya/caffe/examples/cifar3/cifar10_quick_iter_5000.caffemodel.h5', caffe.TEST)

print net.blobs['data'].data.shape

path = '/Users/riya/Downloads/data_mat/test_data.mat'
test_data = scipy.io.loadmat(path)['data']
N = test_data.shape[0]
ch = 3
h_in = 32
w_in = 32

with open('results.csv', 'wb') as csvfile:
	filewriter = csv.writer(csvfile, delimiter=',')

	for i in xrange(N):
		img = X = test_data[i].reshape((ch, h_in, w_in)) - mean_array

		net.blobs['data'].data[...] = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
		out = net.forward()

		pred_probas = out['prob']
		filewriter.writerow([pred_probas.argmax()])