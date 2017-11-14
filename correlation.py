"""
correlation.py
Compute the correlation between two, single-channel, grayscale input images.
The second image must be smaller than the first.

Author: Brad Montgomery
        http://bradmontgomery.net

This code has been placed in the public domain by the author.

USAGE: python correlation <image file> <match file>

"""
#import Image
import numpy
import math
import sys
import timeit

def normalizeArray(a):
    """ 
    Normalize the given array to values between 0 and 1.
    Return a numpy array of floats (of the same shape as given) 
    """
    w,h = a.shape
    minval = a.min()
    if minval < 0: # shift to positive...
        a = a + abs(minval)
    maxval = a.max() # THEN, get max value!
    new_a = numpy.zeros(a.shape, 'd')
    for x in range(0,w):
        for y in range(0,h):
            new_a[x,y] = float(a[x,y])/maxval
    return new_a

def pil2array(im):
    """ Convert a 1-channel grayscale PIL image to a numpy ndarray """
    data = list(im.getdata())
    w,h = im.size
    A = numpy.zeros((w*h), 'd')
    i=0
    for val in data:
        A[i] = val
        i=i+1
    A=A.reshape(w,h)
    return A

def array2pil(A,mode='L'):
    """ 
    Convert a numpy ndarray to a PIL image.
    Only grayscale images (PIL mode 'L') are supported.
    """
    w,h = A.shape
    # make sure the array only contains values from 0-255
    # if not... fix them.
    if A.max() > 255 or A.min() < 0: 
        A = normalizeArray(A) # normalize between 0-1
        A = A * 255 # shift values to range 0-255
    if A.min() >= 0.0 and A.max() <= 1.0: # values are already between 0-1
        A = A * 255 # shift values to range 0-255
    A = A.flatten()
    data = []
    for val in A:
        if val is numpy.nan: val = 0 
        data.append(int(val)) # make sure they're all int's
    im = Image.new(mode, (w,h))
    im.putdata(data)
    return im

def correlation(input, match):
    """ 
    Calculate the correlation coefficients between the given pixel arrays.

    input - an input (numpy) matrix representing an image 
    match - the (numpy) matrix representing the image for which we are looking
    
    """
    t = timeit.Timer()
    assert match.shape < input.shape, "Match Template must be Smaller than the input"
    c = numpy.zeros(input.shape) # store the coefficients...
    mfmean = match.mean()
    iw, ih = input.shape # get input image width and height
    mw, mh = match.shape # get match image width and height
    
    print "Computing Correleation Coefficients..."
    start_time = t.timer()

    for i in range(0, iw):
        for j in range(0, ih):

            # find the left, right, top 
            # and bottom of the sub-image
            if i-mw/2 <= 0:
                left = 0
            elif iw - i < mw:
                left = iw - mw
            else:
                left = i
                
            right = left + mw 

            if j - mh/2 <= 0:
                top = 0
            elif ih - j < mh:
                top = ih - mh
            else:
                top = j

            bottom = top + mh

            # take a slice of the input image as a sub image
            sub = input[left:right, top:bottom]
            assert sub.shape == match.shape, "SubImages must be same size!"
            localmean = sub.mean()
            temp = (sub - localmean) * (match - mfmean)
            s1 = temp.sum()
            temp = (sub - localmean) * (sub - localmean)
            s2 = temp.sum()
            temp = (match - mfmean) * (match - mfmean)
            s3 = temp.sum() 
            denom = s2*s3
            if denom == 0: 
                temp = 0
            else: 
                temp = s1 / math.sqrt(denom)
            
            c[i,j] = temp
            
    end_time = t.timer()
    print "=> Correlation computed in: ", end_time - start_time
    print '\tMax: ', c.max()
    print '\tMin: ', c.min()
    print '\tMean: ', c.mean()
    return c

def main(f1, f2, output_file="CORRELATION.jpg"):
    """ open the image files, and compute their correlation """
    im1 = f1.convert('L')
    im2 = f2.convert('L')
    # Better way to do PIL-Numpy conversion
    f = numpy.asarray(im1) # was f = pil2array(im1)
    w = numpy.asarray(im2) # was w = pil2array(im2)
    corr = correlation(f,w) # was c = array2pil(correlation(f,w))
    c = Image.fromarray(numpy.uint8(normalizeArray(corr) * 255))
    
    print "Saving as: %s" % output_file
    c.save(output_file)

if __name__ == "__main__":
    if len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    else:
        print 'USAGE: python correlation <image file> <match file>'

