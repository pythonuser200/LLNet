import scipy
import gzip,cPickle
import correlation
import os,pdb,glob
import theano 
import skimage
import sklearn
import PIL.Image
import pylab

import scipy.ndimage as ndi
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm 
import matplotlib.font_manager as font_manager

from sklearn import preprocessing, cross_validation
from sklearn.feature_extraction import image
from sklearn.decomposition import PCA
from scipy import io as sp_io
from numpy.random import RandomState
from theano import tensor as T
from scipy import misc
from PIL import Image, ImageEnhance
from pylab import *
from theano.tensor.nnet import conv
from scipy.misc import lena
from itertools import product

#################################################
#      Generating fully overlapping patches     #
#################################################

def   overlapping_patches(path, patch_size):
      
      Ols_images = Image.open (path).convert('L')
      height, width = np.asarray(Ols_images).shape

      if height < 512:
         Ols_images = Ols_images.resize((512, 512), Image.BICUBIC)  
         print '... image resized to 512,512'
      elif height > 512:
         Ols_images = Ols_images.resize((512, 512), Image.ANTIALIAS) 
         print '... image resized to 512,512'  
      
      Ols_images = np.asarray(Ols_images,dtype = 'float')
      Ols_images = correlation.normalizeArray(np.asarray(Ols_images))

      image_height = np.asarray(Ols_images).shape[0]
      image_width = np.asarray(Ols_images).shape[1]

      Ols_patche = image.extract_patches_2d(image=Ols_images, patch_size=patch_size,max_patches=None)
      Ols_patches = np.reshape(Ols_patche,(Ols_patche.shape[0], -1))
      n_patches, nvis = Ols_patches.shape
      rval = (Ols_patches, image_height, image_width)
      return rval
      
#################################################
#  Generating overlapping patches with strides  #
#################################################
      
def   overlapping_patches_strides(path, patch_size, strides):
      
      Ols_images = Image.open (path).convert('L')
      height, width = np.asarray(Ols_images).shape
 
      '''if height < 512:
         Ols_images = Ols_images.resize((512, 512), Image.BICUBIC)  
         print '... image resized to 512,512'
      elif height > 512:
         Ols_images = Ols_images.resize((512, 512), Image.ANTIALIAS) 
         print '... image resized to 512,512'  
         '''
      
      #nrow = 512
      #ncol = 512 #767 #768 for 28x281 767 for 17x17
      
      # ROC Pic
      nrow = height*1
      ncol = width*1
      
      #print '    ... Initial image dimensions: ', nrow, ncol
      
      Up = (nrow-patch_size[0]-strides[0])/strides[0]
      Vp = (ncol-patch_size[1]-strides[1])/strides[1]
      
      #print '    ... Initial patches: ', '%.2f'%(Up), '%.2f'%(Vp)

      Up = np.floor((nrow-patch_size[0]-strides[0])/strides[0])
      Vp = np.floor((ncol-patch_size[1]-strides[1])/strides[1])
      
      #print '    ... Generated patches: ', '%.2f'%(Up), '%.2f'%(Vp)
      
      nrow = np.int(Up*strides[0] + strides[0] + patch_size[0])
      ncol = np.int(Vp*strides[1] + strides[1] + patch_size[1])
      
      #print '    ... Resized image dimensions: ', nrow, ncol
      
      Ols_images = Ols_images.resize((ncol, nrow), Image.BICUBIC)  
      
      Ols_images = np.asarray(Ols_images,dtype = 'float')
      Ols_images = correlation.normalizeArray(np.asarray(Ols_images))
      U = (nrow-patch_size[0]-strides[0])/strides[0]
      V = (ncol-patch_size[1]-strides[1])/strides[1]

      image_height = np.asarray(Ols_images).shape[0]
      image_width = np.asarray(Ols_images).shape[1]

      Ols_patche = image.extract_patches(Ols_images, patch_shape=patch_size, extraction_step=strides)
      Ols_patches = np.reshape(Ols_patche,(Ols_patche.shape[0]*Ols_patche.shape[1], -1))
     
      n_patches, nvis = Ols_patches.shape
      rval = (Ols_patches, image_height, image_width)
      return rval
      
#################################################
#      Reconstructing pathes with strides       #
#################################################
      
def reconstruct_from_patches_with_strides_2d(patches, image_size, strides):

    i_stride = strides[0]
    j_stride = strides[1]
    i_h, i_w = image_size[:2]
    p_h, p_w = patches.shape[1:3]
    img = np.zeros(image_size)
    img1 = np.zeros(image_size)
    n_h = int((i_h - p_h + i_stride)/i_stride)
    n_w = int((i_w - p_w + j_stride)/j_stride)
    for p, (i, j) in zip(patches, product(range(n_h), range(n_w))):
        img[i*i_stride:i*i_stride + p_h, j*j_stride:j*j_stride + p_w] +=p
        img1[i*i_stride:i*i_stride + p_h, j*j_stride:j*j_stride + p_w] +=np.ones(p.shape)
    return img/img1
    
    
