
from ctypes import *
import numpy as np

cdll.LoadLibrary("./libexcorr.so")
ff= CDLL("libexcorr.so")

image=np.zeros((50,100,10))
mask=2*np.ones((10,5,10))

image[25,25,0]=1

ff.corr3d.restype=c_double
ff.corr3d.argtypes=[np.ctypeslib.ndpointer(dtype=c_double,ndim=3,flags="C_CONTIGUOUS"),c_int,c_int,np.ctypeslib.ndpointer(dtype=c_double,ndim=3,flags="C_CONTIGUOUS"),c_int,c_int,c_int,c_int,c_int]

ff.refine.restype=c_double
ff.refine.argtypes=[np.ctypeslib.ndpointer(dtype=c_double,ndim=3,flags="C_CONTIGUOUS"),c_int,c_int,np.ctypeslib.ndpointer(dtype=c_double,ndim=3,flags="C_CONTIGUOUS"),c_int,c_int,c_int,c_int,c_int,POINTER(c_int)]

ff.scan.argtypes=[np.ctypeslib.ndpointer(dtype=c_double,ndim=3,flags="C_CONTIGUOUS"),c_int,c_int,np.ctypeslib.ndpointer(dtype=c_double,ndim=3,flags="C_CONTIGUOUS"),c_int,c_int,c_int,np.ctypeslib.ndpointer(dtype=c_int,ndim=1,flags="C_CONTIGUOUS"),np.ctypeslib.ndpointer(dtype=c_int,ndim=1,flags="C_CONTIGUOUS"),np.ctypeslib.ndpointer(dtype=c_double,ndim=1,flags="C_CONTIGUOUS"),np.ctypeslib.ndpointer(dtype=c_int,ndim=1,flags="C_CONTIGUOUS"),c_int]

res=ff.corr3d(image,image.shape[0],image.shape[1],mask,mask.shape[0],mask.shape[1],image.shape[2],25,25)
print res

p=c_int(0)
res=ff.refine(image,image.shape[0],image.shape[1],mask,mask.shape[0],mask.shape[1],image.shape[2],2,2,p)

val=np.zeros(4);pose=np.zeros(4,dtype=c_int)
sampley=np.array([2,3,4,5],dtype=c_int)
samplex=np.array([2,3,4,5],dtype=c_int)
ff.scan(image,image.shape[0],image.shape[1],mask,mask.shape[0],mask.shape[1],image.shape[2],sampley,samplex,val,pose,4)
print val,pose

