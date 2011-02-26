import ctypes
import numpy
from numpy import ctypeslib
from ctypes import c_int,c_double,c_float

ctypes.cdll.LoadLibrary("./libhog.so")
lhog= ctypes.CDLL("libhog.so")
lhog.process.argtypes=[
    numpy.ctypeslib.ndpointer(dtype=c_float,ndim=3,flags="F_CONTIGUOUS")#im
    ,c_int #dimy
    ,c_int #dimx
    ,c_int #sbin
    ,numpy.ctypeslib.ndpointer(dtype=c_float,ndim=3,flags="F_CONTIGUOUS")# *hog (round(dimy/float(sbin))-2,round(dimx/float(sbin))-2,31)
    ,c_int #hy
    ,c_int #hx
    ,c_int #hz
    ]


def hog(img,sbin=8):
    """
    Compute the HOG descriptor of an image
    """
    if type(img)!=numpy.ndarray:
        raise "img must be a ndarray"
    if img.ndim!=3:
        raise "img must have 3 dimensions"
    hy=int(round(img.shape[0]/float(sbin)))-2
    hx=int(round(img.shape[1]/float(sbin)))-2
    mtype=c_float
    hog=numpy.zeros((hy,hx,31),dtype=mtype,order="f")
    lhog.process(numpy.asfortranarray(img,dtype=mtype),img.shape[0],img.shape[1],sbin,hog,hy,hx,31)
    return hog;#mfeatures.mfeatures(img , sbin);

def hogflip(feat,obin=9):
    """    
    returns the orizontally flipped version of the HOG features
    """
    #feature shape
    #[9 not oriented][18 oriented][4 normalization]
    aux=feat[:,::-1,:]
    last=obin+obin*2
    noriented=numpy.concatenate((aux[:,:,0].reshape(aux.shape[0],aux.shape[1],1),aux[:,:,obin-1:0:-1]),2)
    oriented=numpy.concatenate((aux[:,:,obin].reshape(aux.shape[0],aux.shape[1],1),aux[:,:,last-1:obin:-1]),2)
    norm1=aux[:,:,last+2].reshape(aux.shape[0],aux.shape[1],1)
    norm2=aux[:,:,last+3].reshape(aux.shape[0],aux.shape[1],1)
    norm3=aux[:,:,last].reshape(aux.shape[0],aux.shape[1],1)
    norm4=aux[:,:,last+1].reshape(aux.shape[0],aux.shape[1],1)
    aux=numpy.concatenate((oriented,noriented,norm1,norm2,norm3,norm4),2)
    return aux
