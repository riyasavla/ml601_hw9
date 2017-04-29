import caffe
import lmdb
import numpy as np
import matplotlib.pyplot as plt
from caffe.proto import caffe_pb2
from caffe.io import datum_to_array, array_to_datum
import scipy
from scipy import io
from skimage.io import imshow
from PIL import Image


path = '/Users/riya/Downloads/data_mat/data_batch.mat'
data_labels = scipy.io.loadmat(path)
data = data_labels['data']
labels = data_labels['labels']
print data.shape
print labels.shape

N = data.shape[0]
N_train = 10000
ch = 3
h_in = 32
w_in = 32

map_size = data.nbytes * 2
env = lmdb.open('cifar3_train_lmdb', map_size=map_size)

with env.begin(write=True) as txn:
    for i in xrange(N_train):
        X = data[i].reshape((3, h_in, w_in))
        # X_reshuffled = np.moveaxis(X, 0, 2)
        # im = Image.fromarray(X_reshuffled)
        # im.show()
        datum = caffe.proto.caffe_pb2.Datum()
        datum.channels = X.shape[0]
        datum.height = X.shape[1]
        datum.width = X.shape[2]
        datum.data = X.tobytes()
        datum.label = int(labels[i])
        str_id = '{:08}'.format(i)
        txn.put(str_id.encode('ascii'), datum.SerializeToString())

# path = '/Users/riya/Downloads/data_mat/test_data.mat'
# data_labels = scipy.io.loadmat(path)
# data = data_labels['data']
# labels = data_labels['labels']
# print data.shape
# print labels.shape

N_test = 2000

map_size = data.nbytes * 2
env = lmdb.open('cifar3_test_lmdb', map_size=map_size)

with env.begin(write=True) as txn:
    for i in xrange(N_test):
        X = data[N_train + i].reshape((3, h_in, w_in))
        # X_reshuffled = np.moveaxis(X, 0, 2)
        # im = Image.fromarray(X_reshuffled)
        # im.show()
        datum = caffe.proto.caffe_pb2.Datum()
        datum.channels = X.shape[0]
        datum.height = X.shape[1]
        datum.width = X.shape[2]
        datum.data = X.tobytes()
        datum.label = int(labels[N_train + i])
        str_id = '{:08}'.format(i)
        txn.put(str_id.encode('ascii'), datum.SerializeToString())

    
    