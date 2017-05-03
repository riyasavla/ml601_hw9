import caffe
import lmdb
import numpy as np
import random
import matplotlib.pyplot as plt
from caffe.proto import caffe_pb2
from caffe.io import datum_to_array, array_to_datum
import scipy
from scipy import io
from skimage.io import imshow
from PIL import Image
from numpy import linalg
from skimage.io import imshow


path = '/Users/riya/Downloads/data_mat/data_batch.mat'
data_labels = scipy.io.loadmat(path)
data = data_labels['data']
labels = data_labels['labels']
print data.shape
print labels.shape

N = data.shape[0]
N_train = 9000
N_test = 12000 - N_train
ch = 3
h_in = 32
w_in = 32

def compute_train_cov(data):
    cov_dict = {}
    for c in xrange(ch):
        channel = data[:, c, :]
        cov = np.dot(channel.T, channel) / channel.shape[0]
        cov_dict[str(c)] = cov
    scipy.io.savemat('zca.mat', cov_dict)
    return  cov_dict

# Per channel ZCA whitening
def my_zca(data, cov_dict):
    eps = 0.000000001
    new_data = np.zeros(data.shape)
    for c in xrange(ch):
        channel = data[:, c, :]
        print "channel" , channel.shape
        cov = cov_dict[str(c)]
        U,S,V = np.linalg.svd(cov)
        channel_rot = np.dot(channel, U)
        channel_rot_cov = (np.dot(channel_rot.T, channel_rot)/ channel_rot.shape[0])
        #plt.show(imshow(np.dot(channel_rot.T, channel_rot)/ channel_rot.shape[0]))
        print "CHECK DIAG ", np.array_equal(np.argmax(channel_rot_cov, axis =1), np.argmax(channel_rot_cov, axis =1))
        channel_white = channel_rot / (np.sqrt(S) + eps)
        channel_white = np.dot(channel_white, U.T)
        new_data[:, c, :] = channel_white
    return new_data


map_size = data.nbytes * 10
env = lmdb.open('cifar3_train_lmdb', map_size=map_size)

indices = range(N)
#random.shuffle(indices)

train_indices = indices[:N_train]
train_data = np.zeros((N_train, ch*h_in*w_in))
for i in range(N_train):
    #plt.show(imshow(data[i].reshape((3, h_in, w_in))))
    #imshow(np.moveaxis(data[i].reshape((3, h_in, w_in)), 0, 2))
    train_data[i] = data[train_indices[i]]

#mean 
### SAVE IT as .mat file for inference !!!
mean = np.mean(train_data, axis = 0)
mean_dict = {}
mean_dict['data'] = mean
scipy.io.savemat('mean.mat', mean_dict)
# mean_file = scipy.io.loadmat('mean.mat')['data']
# equal = np.allclose(mean, mean_file)
# print equal
# print mean
# print
# print mean_file

# Zero center
train_data -= mean
train_data = train_data.reshape(N_train, ch, h_in*w_in)
# Per channel ZCA whitening
cov_dict = compute_train_cov(train_data)
train_data = my_zca(train_data, cov_dict)

def vis_square(data):
    """Take an array of shape (n, height, width) or (n, height, width, 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""
    
    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
               (0, 1), (0, 1))                 # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    
    plt.show(imshow(data)); plt.axis('off')


def vis_few_whitened_images():
    X_all = np.zeros((16, h_in, w_in, ch))
    for i in xrange(16):
    #for i in xrange(5):
        X = train_data[i].reshape((ch, h_in, w_in))
        X = np.moveaxis(X, 0, 2)
        max_im = X.max()
        min_im = X.min()
        X_vis = (X + min_im) / (max_im - min_im)
        X_all[i] = X_vis
    print X_all.shape
    vis_square(X_all)

#vis_few_whitened_images()


with env.begin(write=True) as txn:
    for i in xrange(N_train):
    #for i in xrange(5):
        X = train_data[i].reshape((ch, h_in, w_in))
        X_reshuffled = np.moveaxis(X, 0, 2)
        
        max_im = X_reshuffled.max()
        min_im = X_reshuffled.min()
        X_vis = (X_reshuffled + min_im) / (max_im - min_im)
        #plt.show(imshow(X_vis))

        #datum = caffe.proto.caffe_pb2.Datum()
        # datum.channels = X.shape[0]
        # datum.height = X.shape[1]
        # datum.width = X.shape[2]
        # datum.data = X.tobytes()
        #datum.label = int(labels[i])

        datum = caffe.io.array_to_datum(X, int(labels[i]))
        str_id = '{:08}'.format(i)
        txn.put(str_id.encode('ascii'), datum.SerializeToString())

# path = '/Users/riya/Downloads/data_mat/test_data.mat'
# data_labels = scipy.io.loadmat(path)
# data = data_labels['data']
# labels = data_labels['labels']
# print data.shape
# print labels.shape

test_indices = indices[N_train:]
test_data = np.zeros((N_test, ch*h_in*w_in))
for i in range(N_test):
    test_data[i] = data[test_indices[i]]

#zero center test data as well
test_data -= mean
test_data = test_data.reshape((N_test, ch, h_in*w_in))
# Per channel ZCA whitening
test_data = my_zca(test_data, cov_dict)

map_size = data.nbytes * 10
env = lmdb.open('cifar3_test_lmdb', map_size=map_size)

with env.begin(write=True) as txn:
    for i in xrange(N_test):
        X = test_data[i].reshape((3, h_in, w_in))
        # X_reshuffled = np.moveaxis(X, 0, 2)
        # im = Image.fromarray(X_reshuffled)
        # im.show()
        # datum = caffe.proto.caffe_pb2.Datum()
        # datum.channels = X.shape[0]
        # datum.height = X.shape[1]
        # datum.width = X.shape[2]
        # datum.data = X.tobytes()
        # datum.label = int(labels[N_train + i])

        datum = caffe.io.array_to_datum(X, int(labels[N_train + i]))
        str_id = '{:08}'.format(i)
        txn.put(str_id.encode('ascii'), datum.SerializeToString())
































def standford_zca(x):
    # STEP 0: Load data
    # x = np.loadtxt('pcaData.txt')
    # plt.figure()
    # plt.plot(x[0,:], x[1,:], 'o', mec='blue', mfc='none')
    # plt.title('Raw data')
    # plt.show()

    # STEP 1a: Implement PCA to obtain the rotation matrix, U, which is
    # the eigenbases sigma.

    sigma = x.dot(x.T) / x.shape[1]
    U, S, Vh = linalg.svd(sigma)

    # STEP 1b: Compute xRot, the projection on to the eigenbasis

    xRot = U.T.dot(x)

    # STEP 2: Reduce the number of dimensions from 2 to 1

    k = 1
    xRot = U[:,0:k].T.dot(x)
    xHat = U[:,0:k].dot(xRot)

    # STEP 3: PCA Whitening

    epsilon = 1e-5
    xPCAWhite = np.diag(1.0 / np.sqrt(S + epsilon)).dot(U.T).dot(x)

    # STEP 4: ZCA Whitening

    xZCAWhite = U.dot(xPCAWhite)
    return xZCAWhite

def zca_whiten(X):
    """
    Applies ZCA whitening to the data (X)
    http://xcorr.net/2011/05/27/whiten-a-matrix-matlab-code/
    X: numpy 2d array
        input data, rows are data points, columns are features
    Returns: ZCA whitened 2d array
    """
    assert(X.ndim == 2)
    EPS = 10e-5
    #   covariance matrix
    cov = np.dot(X.T, X)
    #   d = (lambda1, lambda2, ..., lambdaN)
    d, E = np.linalg.eigh(cov)
    #   D = diag(d) ^ (-1/2)
    D = np.diag(1. / np.sqrt(d + EPS))
    #   W_zca = E * D * E.T
    W = np.dot(np.dot(E, D), E.T)
    X_white = np.dot(X, W)
    return X_white

    
    