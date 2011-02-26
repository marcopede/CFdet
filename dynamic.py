# test unit for dinamic programming
import numpy
import ctypes
from ctypes import c_int,c_double,c_float

ctypes.cdll.LoadLibrary("./libdynamic.so")
ldin= ctypes.CDLL("libdynamic.so")
ldin.filltables.argtypes=[
    numpy.ctypeslib.ndpointer(dtype=c_float,ndim=2,flags="C_CONTIGUOUS"),#scores
    numpy.ctypeslib.ndpointer(dtype=c_float,ndim=3,flags="C_CONTIGUOUS"),#deform
    c_int, #num nodes
    c_int #nlables=(radius*2+1)^2
    ]

ldin.maxmarginal.argtypes=[
    c_int, #node
    c_int, #label
    numpy.ctypeslib.ndpointer(dtype=c_int,ndim=2,flags="C_CONTIGUOUS")#best positions    
    ]
ldin.maxmarginal.restype=c_float

ldin.map.argtypes=[
    numpy.ctypeslib.ndpointer(dtype=c_int,ndim=1,flags="C_CONTIGUOUS"),#best path    
    c_int,#node
    numpy.ctypeslib.ndpointer(dtype=c_float,ndim=1,flags="C_CONTIGUOUS"),#max marginals
    ]
ldin.map.restype=c_float

#       l0 l1 l2 l3
#
#   n0  .  .  .  .
#
#   n1  .  .  .  .
#
#   n2  .  .  .  .
#

nodes=4
numlab=4
scr=numpy.array([[1,0,4,1],
                [1,5,3,2],
                [3,5,3,2],
                [0,0,13,0],
                [3,1,2,0]],dtype=numpy.float32);

#pott model
#             node i
#           [. . . .]
#  node i-1 [. . . .]
#           [. . . .]
#           [. . . .]
df=-numpy.array([[[0,1,1,1],
                [1,0,1,1],
                [1,1,0,1],
                [1,1,1,0]],
                
                [[0,1,1,1],
                [1,0,1,1],
                [1,1,0,1],
                [1,1,1,0]],
                
                [[0,1,1,1],
                [1,0,1,1],
                [1,1,0,1],
                [1,1,1,0]],
                
                [[0,1,1,1],
                [1,0,1,1],
                [1,1,0,1],
                [1,1,1,0]]],dtype=numpy.float32);
    
pos=numpy.zeros((nodes,numlab),dtype=numpy.int32);

ldin.filltables(scr,df,nodes,numlab)
mmax=ldin.maxmarginal(1,0,pos)
print pos
print mmax

import time
t=time.time()
print "MAP"
ldin.filltables(scr,df,nodes,numlab)
apos=numpy.zeros((numlab,nodes,numlab),dtype=numpy.int32);
mmap=numpy.zeros(numlab)
for l in range(numlab):
    mmap[l]=ldin.maxmarginal(0,l,apos[l])
print mmap.max()
print "All"
print apos[0]
print mmap[0]
print apos[1]
print mmap[1]
print apos[2]
print mmap[2]
print apos[3]
print mmap[3]
print "Best"
print apos[mmap.argmax()]
print "Time",time.time()-t


t=time.time()
avrg=c_float(0)
print "MAP2"
ldin.filltables(scr,df,nodes,numlab)
path=numpy.zeros(nodes,dtype=c_int)
maxm=numpy.zeros(numlab,dtype=c_float)
mmap=ldin.map(path,0,maxm)
print "MAP",mmap
print "PATH",path
print "MAXM",maxm
print "AVRG",numpy.mean(maxm)
print "Time",time.time()-t

#print
#pos=-numpy.ones((nodes,numlab),dtype=numpy.int32);
#ldin.filltables(scr,df,nodes,numlab)
#mmax=ldin.maxmarginal(1,3,pos)
#print pos
#print mmax

