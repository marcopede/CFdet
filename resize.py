#!/usr/bin/python 

import platform
from numpy import ctypeslib,empty,array,exp,ascontiguousarray,zeros,asfortranarray
from ctypes import c_float,c_double,c_int
from time import time

def resize(img,scale):
    sdims=img.shape
    datatype=c_double
    if img.dtype!=datatype:
       print "Error the image must be of doubles!"
       raise RuntimeError

    if scale>1.0:
       print "Invalid scaling factor!"
       raise RuntimeError  
    
    #img = ascontiguousarray(img,c_double) # make array continguous
    img = asfortranarray(img,c_double) # make array continguous
    
    try:
        mresize = ctypeslib.load_library("libresize.so",".") 
        #mresize = ctypeslib.load_library("libchi2.so",".") # adapt path to library
    except:
        print "Unable to load resize library"
        raise RuntimeError
        
    fresize = mresize.resize1dtran
    fresize.restype = None
    #fresize.argtypes = [ ctypeslib.ndpointer(dtype=datatype, ndim=3, flags='CONTIGUOUS'), c_int,ctypeslib.ndpointer(dtype=datatype, ndim=3, flags='CONTIGUOUS'), c_int, c_int , c_int ]
    fresize.argtypes = [ ctypeslib.ndpointer(dtype=datatype, ndim=3), c_int,ctypeslib.ndpointer(dtype=datatype, ndim=3), c_int, c_int , c_int ]
    ddims = [int(round(sdims[0]*scale)),int(round(sdims[1]*scale)),sdims[2]];
    mxdst = zeros((ddims), dtype=datatype)
    tmp = zeros((ddims[0],sdims[1],sdims[2]), dtype=datatype)
    img1=img#array(img,order="F")
    #fresize(img, sdims[0], tmp, ddims[0], sdims[1], sdims[2]);
    t1=time()
    fresize(img1, sdims[0], tmp, ddims[0], sdims[1], sdims[2]);
    #print ddims[1],sdims[0],sdims[2]
    #fresize(tmp, sdims[1], mxdst, ddims[1], ddims[0], sdims[2]);
    #import pylab
    #pylab.imshow(tmp)
    #tmp.shape
    fresize(tmp, sdims[1], mxdst, ddims[1], ddims[0], sdims[2]);
    t2=time()
    #print "Resize:",t2-t1
    #return ascontiguousarray(mxdst.reshape(sdims[2],ddims[1],ddims[0]).T),0#,tmp
    return mxdst.reshape(ddims[2],ddims[1],ddims[0]).T


#def chi2_kernel(X,Y=None,K=None,oldmeanK=None):
#    K,meanK = chi2_dist(X,Y,K)
#    if (oldmeanK == None):
#        K = exp(-0.5*K/meanK)
#    else:    
#        K = exp(-0.5*K/oldmeanK)
#   return K,meanK


if __name__ == "__main__":
    from numpy.random import random_integers
    from time import time
    from pylab import imread,figure,imshow
    from ctypes import c_float,c_double,c_int
    
    img=imread("test.png").astype(c_double)
    imshow(img)
    img1=resize(img,0.25)
    figure()
    imshow(img1)
    #figure()
    #imshow(tmp)

    

