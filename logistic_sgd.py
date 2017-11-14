import cPickle
import gzip
import os
import sys
import time
import h5py
import numpy
import theano
import theano.tensor as T
import nlinalg

from Data_process2 import overlapping_patches, overlapping_patches_strides

#################################################
#          Logistic Regression Class            #
#################################################

class LogisticRegression(object):

    def __init__(self, input, n_in, n_out,W=None, b=None):

        if not W :
           self.W = theano.shared(
               value=numpy.zeros(
                   (n_in, n_out),
                   dtype=theano.config.floatX
               ),
               name='W',
               borrow=True
           )

        if not b:
           self.b = theano.shared(
               value=numpy.zeros(
                   (n_out,),
                   dtype=theano.config.floatX
               ),
               name='b',
               borrow=True
           )
       
        self.p_y_given_x = T.nnet.sigmoid(T.dot(input, self.W) + self.b)
        self.y_pred = self.p_y_given_x
        self.params = [self.W, self.b]
            
    def image_norm(self, y, obj):

	y_diff = (y - self.y_pred)
	l2norm = (T.sqrt((y_diff**2).sum(axis=1,keepdims=False))**2)
	lambda_reg = 0.00001
	weights = 0
	for i in xrange(obj.n_layers):
            weight = (T.sqrt((obj.dA_layers[i].W ** 2).sum())**2)
            #weight = (nlinalg.trace(T.dot(obj.dA_layers[i].W.T, obj.dA_layers[i].W)))**2 #Frobenius norm
            weights = weights + weight
	regterm = T.sum(weights,keepdims=False)
	
	return T.mean(l2norm) + lambda_reg/2 *regterm
	
    def image_norm_noreg(self, y):

	y_diff = (y - self.y_pred)
	l2norm = (T.sqrt((y_diff**2).sum(axis=1,keepdims=False))**2)
	
	return T.mean(l2norm)

#################################################
#          Loading Data for Training            #
#################################################

def load_data(dataset):

    print '... loading h5py mat data'

    f = h5py.File(dataset)

    train_set_x = numpy.transpose(f['train_set_x'])
    valid_set_x = numpy.transpose(f['valid_set_x'])
    test_set_x = numpy.transpose(f['test_set_x'])
    train_set_y = numpy.transpose(f['train_set_y'])
    valid_set_y = numpy.transpose(f['valid_set_y'])
    test_set_y = numpy.transpose(f['test_set_y'])
    
    train_set = train_set_x, train_set_y
    valid_set = valid_set_x, valid_set_y
    test_set = test_set_x, test_set_y

    def shared_dataset(data_xy, borrow=True):

        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                       dtype=theano.config.floatX),
                       borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                       dtype=theano.config.floatX),
                       borrow=borrow)

        return shared_x, shared_y #T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval
    
#################################################
#  Loading Data for Testing with Full Overlap   #
#################################################

def load_data_overlapped(te_dataset, patch_size):

    test_set_, te_h, te_w = overlapping_patches(path=te_dataset, patch_size = patch_size)

    def shared_dataset(data_x, borrow=True):
        shared_data = theano.shared(numpy.asarray(data_x,
                          dtype=theano.config.floatX),
                          borrow=borrow)
        return shared_data

    test_set_ = shared_dataset(test_set_)
    rval = test_set_
    return rval, te_h, te_w
    
#################################################
# Loading Data for Testing with Overlap Strides #
#################################################
    
def load_data_overlapped_strides(te_dataset, patch_size, strides):

    test_set_, te_h, te_w = overlapping_patches_strides(path=te_dataset, patch_size = patch_size, strides=strides)

    def shared_dataset(data_x, borrow=True):
        shared_data = theano.shared(numpy.asarray(data_x,
                          dtype=theano.config.floatX),
                          borrow=borrow)
        return shared_data

    test_set_ = shared_dataset(test_set_)
    rval = test_set_
    return rval, te_h, te_w
