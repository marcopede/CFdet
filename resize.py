#!/usr/bin/python 

import platform
from numpy import ctypeslib,empty,array,exp,ascontiguousarray,zeros,asfortranarray
from ctypes import c_float,c_double,c_int
from time import time

def resize(img,scale):
    """
        downsample img to scale 
    """
    sdims=img.shape
    datatype=c_double
    if img.dtype!=datatype:
       print "Error the image must be of doubles!"
       raise RuntimeError

    if scale>1.0:
       print "Invalid scaling factor!"
       raise RuntimeError  
    
    img = asfortranarray(img,c_double) # make array continguous
    
    try:
        mresize = ctypeslib.load_library("libresize.so",".") 
    except:
        print "Unable to load resize library"
        raise RuntimeError
        
    #use two times the 1d resize to get a 2d resize
    fresize = mresize.resize1dtran
    fresize.restype = None
    fresize.argtypes = [ ctypeslib.ndpointer(dtype=datatype, ndim=3), c_int,ctypeslib.ndpointer(dtype=datatype, ndim=3), c_int, c_int , c_int ]
    ddims = [int(round(sdims[0]*scale)),int(round(sdims[1]*scale)),sdims[2]];
    mxdst = zeros((ddims), dtype=datatype)
    tmp = zeros((ddims[0],sdims[1],sdims[2]), dtype=datatype)
    img1=img
    t1=time()
    fresize(img1, sdims[0], tmp, ddims[0], sdims[1], sdims[2]);
    fresize(tmp, sdims[1], mxdst, ddims[1], ddims[0], sdims[2]);
    t2=time()
    return mxdst.reshape(ddims[2],ddims[1],ddims[0]).T


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

    

