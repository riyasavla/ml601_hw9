import caffe
import lmdb
import numpy as np
from caffe.proto import caffe_pb2
import scipy
from scipy import io
import csv
import sys, getopt

# mean_file = '/Users/riya/Downloads/mean.binaryproto' #'/Users/riya/caffe/examples/cifar3/mean.binaryproto'
# model = "/Users/riya/Downloads/cifar10_full_iter_60000.caffemodel.h5" #'/Users/riya/caffe/examples/cifar3/cifar10_quick_iter_5000.caffemodel.h5'
# deploy_net = '/Users/riya/caffe/examples/cifar3/deploy.prototxt'

ch = 3
h_in = 32
w_in = 32

# Per channel ZCA whitening
def my_zca(data, cov_dict):
    eps = 0.000000001
    new_data = np.zeros(data.shape)
    for c in xrange(ch):
        channel = data[:, c, :]
        #print "channel" , channel.shape
        cov = cov_dict[str(c)]
        U,S,V = np.linalg.svd(cov)
        channel_rot = np.dot(channel, U)
        channel_rot_cov = (np.dot(channel_rot.T, channel_rot)/ channel_rot.shape[0])
        #plt.show(imshow(np.dot(channel_rot.T, channel_rot)/ channel_rot.shape[0]))
        #print "CHECK DIAG ", np.array_equal(np.argmax(channel_rot_cov, axis =1), np.argmax(channel_rot_cov, axis =1))
        channel_white = channel_rot / (np.sqrt(S) + eps)
        channel_white = np.dot(channel_white, U.T)
        new_data[:, c, :] = channel_white
    return new_data

def main(argv):

	mean_file = ''
	zca_file = ''
	model = ''
	deploy_net = ''

	opts, args = getopt.getopt(argv,"n:z:m:d:",["mean=", "zca=", "model=","deploy="])
	for opt, arg in opts:
		if opt in ("-n", "--mean"):
			mean_file = str(arg)
		elif opt in ("-z", "--zca"):
			zca_file = str(arg)
		elif opt in ("-m", "--model"):
			model = str(arg)
		elif opt in ("-d",  "--deploy"):
			deploy_net = str(arg)

	ch = 3
	h_in = 32
	w_in = 32

	mean = (scipy.io.loadmat(mean_file)['data'])
	cov_dict = scipy.io.loadmat(zca_file)

	net = caffe.Net(deploy_net, model, caffe.TEST)

	print net.blobs['data'].data.shape

	path = '/Users/riya/Downloads/data_mat/test_data.mat'
	data = scipy.io.loadmat(path)['data']

	N = data.shape[0]

	test_data = np.zeros((N, ch*h_in*w_in))
	for i in xrange(N):
		test_data[i] = data[i]

	test_data -= mean
	test_data = test_data.reshape((N, ch, h_in*w_in))
	# Per channel ZCA whitening
	test_data = my_zca(test_data, cov_dict)

	with open('results.csv', 'wb') as csvfile:
		filewriter = csv.writer(csvfile, delimiter=',')

		for i in xrange(N):
			img = X = test_data[i].reshape((ch, h_in, w_in)) 

			net.blobs['data'].data[...] = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
			out = net.forward()

			pred_probas = out['prob']
			filewriter.writerow([pred_probas.argmax()])

if __name__ == "__main__":
   main(sys.argv[1:])