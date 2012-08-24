import ctypes
import numpy
from numpy import ctypeslib
from ctypes import c_int,c_double,c_float
import time

#compute(int width,int height,dtype *data,int *result)
ctypes.cdll.LoadLibrary("./libcrf2.so")
lib= ctypes.CDLL("libcrf2.so")
lib.compute.argtypes=[
    c_int,# w
    c_int,# h
    numpy.ctypeslib.ndpointer(dtype=c_float,ndim=3,flags="C_CONTIGUOUS"),#image scores
    numpy.ctypeslib.ndpointer(dtype=c_int,ndim=2,flags="C_CONTIGUOUS"),#image labels
    ]
lib.compute.restype=ctypes.c_float

#dtype compute_graph(int *connect,int num_parts,dtype *costs,int num_labels,dtype *data,int *reslab)
ctypes.cdll.LoadLibrary("./libcrf2.so")
lib= ctypes.CDLL("libcrf2.so")
lib.compute_graph.argtypes=[
    #numpy.ctypeslib.ndpointer(dtype=c_int,ndim=2,flags="C_CONTIGUOUS"),# topolgy of the graph
    c_int,c_int,# num parts
    #numpy.ctypeslib.ndpointer(dtype=c_float,ndim=3,flags="C_CONTIGUOUS"),# costs for each edge
    numpy.ctypeslib.ndpointer(dtype=c_float,ndim=3,flags="C_CONTIGUOUS"),# costs for each edge
    c_int,# num_lab_y
    c_int,# num_lab_x
    numpy.ctypeslib.ndpointer(dtype=c_float,ndim=2,flags="C_CONTIGUOUS"),#observations
    numpy.ctypeslib.ndpointer(dtype=c_int,ndim=2,flags="C_CONTIGUOUS"),#labels
    ]
lib.compute_graph.restype=ctypes.c_float

#fill_cache(dtype* cache,int m_height,int m_width,int m_nLabels,int num_parts_y,int num_parts_x,dtype *costs)
#lib.fill_cache.argtypes=[
#    numpy.ctypeslib.ndpointer(dtype=c_float,ndim=4,flags="C_CONTIGUOUS"),# 
#    c_int,c_int,# num parts
#    c_int,c_int,c_int,# num_labels
#    numpy.ctypeslib.ndpointer(dtype=c_float,ndim=3,flags="C_CONTIGUOUS"),#observations
#    ]
#lib.compute_graph.restype=ctypes.c_float


crf=lib.compute
crfgr=lib.compute_graph
#fill_cache=lib.fill_cache

def cost(numy,numx,movy,movx,c=0.001,ch=None,cv=None):
    np=numy*numx
    nl=movy*movx
    cc=numpy.zeros((2,np,nl,nl),dtype=numpy.float32)
    if ch==None:
        ch=numpy.ones((numy,numx),dtype=numpy.float32)
    if cv==None:
        cv=numpy.ones((numy,numx),dtype=numpy.float32)
    for p in range(np):
        for l1 in range(nl):
            for l2 in range(nl):
                x1=l1%(movx*2-1)/2;
                y1=l1/(movx*2-1)/2;
                x2=l2%(movx*2-1)/2;    
                y2=l2/(movx*2-1)/2;
                cc[0,p,l1,l2]=ch[p/numx,p%numx]*abs(x1-x2)
                cc[1,p,l1,l2]=cv[p/numx,p%numx]*abs(y1-y2)
    cc[0,:,:,-1]=0
    cc[1,:,-1,:]=0
    return cc

def costV(numy,numx,movy,movx,c=0.001):
    np=numy*numx
    nl=movy*movx
    cc=numpy.zeros((nl,nl),dtype=numpy.float32)
    for l1 in range(nl):
        for l2 in range(nl):
            x1=l1%(movx*2-1)/2;
            y1=l1/(movx*2-1)/2;
            x2=l2%(movx*2-1)/2;    
            y2=l2/(movx*2-1)/2;
            cc[l1,l2]=abs(x1-x2)+abs(y1-y2)
            #cc[l2,l1]=abs(x1-x2)+abs(y1-y2)
    return cc


def match_slow(m1,m2,cost,padvalue=0,pad=0,feat=True,show=True):
    t=time.time()
    blk1=numpy.concatenate((m1[:-1:,:-1],m1[:-1,1:],m1[1:,:-1],m1[1:,1:]),2)
    blk2=numpy.concatenate((m2[:-1:,:-1],m2[:-1,1:],m2[1:,:-1],m2[1:,1:]),2)
    p1=blk1[::2,::2]
    p2=blk2[::2,::2]
    numy=p1.shape[0]
    numx=p1.shape[1]
    #numlab=blk1.shape[0]*blk1.shape[1]
    bb1=blk1.reshape((blk1.shape[0]*blk1.shape[1],-1))
    pp1=p1.reshape((p1.shape[0]*p1.shape[1],-1))
    bb2=blk2.reshape((blk2.shape[0]*blk2.shape[1],-1)).T
    #pp2=p2.reshape((p2.shape[0]*p2.shape[1],-1))

    movy=blk1.shape[0]/2
    movx=blk1.shape[1]/2
    numlab=(movy*2+1)*(movx*2+1)
    blk2pad=padvalue*numpy.ones((blk1.shape[0]+2*movy,blk1.shape[1]+2*movx,blk1.shape[2]))
    blk2pad[movy-pad:-movy+pad,movx-pad:-movx+pad]=blk2
    #data=numpy.zeros((numy,numx,(movy*2+1),(movx*2+1)),dtype=numpy.float32)
    data=numpy.zeros((numy,numx,(movy*2+1)*(movx*2+1)),dtype=numpy.float32)
    for px in range(p1.shape[1]):
        for py in range(p1.shape[0]):
            data[py,px]=-numpy.sum(p1[py,px]*blk2pad[2*py:2*py+(2*movy+1),2*px:2*px+(2*movx+1)],2).flatten()
    #print "time hog",time.time()-t
    rdata=data.reshape((data.shape[0]*data.shape[1],-1))
    rmin=rdata.min()
    rdata=rdata-rmin

#        data1=-numpy.dot(pp1,bb2)
#        t=time.time()
#        data2=-numpy.dot(pp1,bb1.T)
#        print "time hog",time.time()-t
    res=numpy.zeros((numy,numx),dtype=c_int)
    #rdata=numpy.ascontiguousarray((rdata.T-rdata.min(1).reshape((1,-1))).T)
    #sdf
    t=time.time()
    #print "before"
    scr=crfgr(numy,numx,cost,movy*2+1,movx*2+1,rdata,res)  
    #print "after"
    scr=scr-rmin*numy*numx
    res2=numpy.zeros((2,res.shape[0],res.shape[1]),dtype=c_int)
    res2[0]=(res/(movx*2+1)-movy)
    res2[1]=(res%(movx*2+1)-movx)

    if show:    
        print "time config",time.time()-t
        print scr,res
        import pylab
        pylab.figure(10)
        pylab.clf()
        pylab.ion()
        #pylab.axis("image")
        aa=pylab.axis()
        pylab.axis([aa[0],aa[1],aa[3],aa[2]])
        import util
        for px in range(res.shape[1]):
            for py in range(res.shape[0]):
                util.box(py*20+(res[py,px]/(movy*2+1)-movy)*10, px*20+(res[py,px]%(movx*2+1)-movx)*10, py*20+(res[py,px]/(movy*2+1)-movy)*10+20, px*20+(res[py,px]%(movx*2+1)-movx)*10+20, col='b', lw=2)   
                pylab.text(px*20+(res[py,px]%(movx*2+1)-movx)*10,py*20+(res[py,px]/(movy*2+1)-movy)*10,"(%d,%d)"%(py,px))
           
    if feat:
        dfeat=numpy.zeros(m1.shape,dtype=numpy.float32)
        for px in range(p1.shape[1]):
            for py in range(p1.shape[0]):
                dfeat[py*2:py*2+2,px*2:px*2+2]=blk2pad[py*2+res[py,px]/(movx*2+1),px*2+res[py,px]%(movx*2+1)].reshape(2,2,31)
        edge=numpy.zeros(res2.shape,dtype=numpy.float32)
        #edge[0,:-1,:]=(res2[0,:-1]-res2[0,1:])
        #edge[1,:,:-1]=(res2[1,:,:-1]-res2[1,:,1:])
        edge[0,:-1,:]=abs(res2[0,:-1,:]-res2[0,1:,:])+abs(res2[1,:-1,:]-res2[1,1:,:])
        edge[1,:,:-1]=abs(res2[0,:,:-1]-res2[0,:,1:])+abs(res2[1,:,:-1]-res2[1,:,1:])
        #edge[0,:-1,:]=abs(dy)/(movx*2+1)+abs(dy)%(movx*2+1)
        #edge[1,:,:-1]=abs(dx)/(movx*2+1)+abs(dx)%(movx*2+1)
        return scr,res2,dfeat,-edge

    return scr,res

from ctypes import c_int,c_double,c_float
import ctypes
ctypes.cdll.LoadLibrary("./libexcorr.so")
ff= ctypes.CDLL("libexcorr.so")
ff.scaneigh.argtypes=[numpy.ctypeslib.ndpointer(dtype=c_float,ndim=3,flags="C_CONTIGUOUS"),c_int,c_int,numpy.ctypeslib.ndpointer(dtype=c_float,flags="C_CONTIGUOUS"),c_int,c_int,c_int,numpy.ctypeslib.ndpointer(dtype=c_int,flags="C_CONTIGUOUS"),numpy.ctypeslib.ndpointer(dtype=c_int,flags="C_CONTIGUOUS"),numpy.ctypeslib.ndpointer(dtype=c_float,flags="C_CONTIGUOUS"),numpy.ctypeslib.ndpointer(dtype=c_int,flags="C_CONTIGUOUS"),numpy.ctypeslib.ndpointer
(dtype=c_int,flags="C_CONTIGUOUS"),c_int,c_int,c_int,c_int]

#inline ftype refineighfull(ftype *img,int imgy,int imgx,ftype *mask,int masky,int maskx,int dimz,ftype dy,ftype dx,int posy,int posx,int rady,int radx,ftype *scr,int *rdy,int *rdx,ftype *prec,int pady,int padx,int occl)
ff.refineighfull.argtypes=[
numpy.ctypeslib.ndpointer(dtype=c_float,ndim=3,flags="C_CONTIGUOUS"),c_int,c_int,
numpy.ctypeslib.ndpointer(dtype=c_float,flags="C_CONTIGUOUS"),c_int,c_int,c_int,
c_float,c_float,c_int,c_int,c_int,c_int,
numpy.ctypeslib.ndpointer(dtype=c_float,ndim=1,flags="C_CONTIGUOUS"),
numpy.ctypeslib.ndpointer(dtype=c_int,ndim=1,flags="C_CONTIGUOUS"),
numpy.ctypeslib.ndpointer(dtype=c_int,ndim=1,flags="C_CONTIGUOUS"),
ctypes.POINTER(c_float),c_int,c_int,c_int]
ff.refineighfull.restype=c_float

def match_wrong(m1,m2,cost,movy=None,movx=None,padvalue=0,pad=0,feat=True,show=True):
    t=time.time()
    blk1=numpy.concatenate((m1[:-1:,:-1],m1[:-1,1:],m1[1:,:-1],m1[1:,1:]),2)
    blk2=numpy.concatenate((m2[:-1:,:-1],m2[:-1,1:],m2[1:,:-1],m2[1:,1:]),2)
    p1=blk1[::2,::2]
    numy=p1.shape[0]
    numx=p1.shape[1]
    bb2=blk2.reshape((blk2.shape[0]*blk2.shape[1],-1)).T

    if movy==None:
        movy=blk1.shape[0]/2
    if movx==None:
        movx=blk1.shape[1]/2
    numlab=(movy*2+1)*(movx*2+1)
    #blk2pad=padvalue*numpy.ones((blk1.shape[0]+2*movy,blk1.shape[1]+2*movx,blk1.shape[2]))
    #blk2pad[movy-pad:-movy+pad,movx-pad:-movx+pad]=blk2
    data=numpy.zeros((numy,numx,(movy*2+1)*(movx*2+1)),dtype=numpy.float32)
    #print "time preparation",time.time()-t
    t=time.time()
#    t1=time.time()
#    for px in range(p1.shape[1]):
#        for py in range(p1.shape[0]):
#            data[py,px]=-numpy.sum(p1[py,px]*blk2pad[2*py:2*py+(2*movy+1),2*px:2*px+(2*movx+1)],2).flatten()
#    print "Time mode1",time.time()-t1

    #data1=numpy.zeros((numy,numx,(movy*2+1)*(movx*2+1)),dtype=numpy.float32)
    mmax=numpy.zeros(2,dtype=c_int)
    #t1=time.time()
    p1=numpy.ascontiguousarray(p1)
    blk2=numpy.ascontiguousarray(blk2)
    for px in range(p1.shape[1]):
        for py in range(p1.shape[0]):
            ff.refineighfull(blk2,blk2.shape[0],blk2.shape[1],p1[py,px],
                1,1,p1.shape[2],0,0,py*2+pad,px*2+pad,movy,movx,data[py,px],
                mmax,mmax,ctypes.POINTER(c_float)(),0,0,0)
    #print "Time mode1",time.time()-t1

    #print "time hog",time.time()-t
    rdata=data.reshape((data.shape[0]*data.shape[1],-1))
    rmin=rdata.min()
    rdata=rdata-rmin

    res=numpy.zeros((numy,numx),dtype=c_int)
    #sdf
    #print "time matching",time.time()-t
    t=time.time()
    #print "before"
    scr=crfgr(numy,numx,cost,movy*2+1,movx*2+1,rdata,res)  
    #print "after"
    scr=scr-rmin*numy*numx
    res2=numpy.zeros((2,res.shape[0],res.shape[1]),dtype=c_int)
    res2[0]=(res/(movx*2+1)-movy)
    res2[1]=(res%(movx*2+1)-movx)
    #print "time config",time.time()-t

    if show:    
        print scr,res
        import pylab
        pylab.figure(10)
        pylab.clf()
        pylab.ion()
        #pylab.axis("image")
        aa=pylab.axis()
        pylab.axis([aa[0],aa[1],aa[3],aa[2]])
        import util
        for px in range(res.shape[1]):
            for py in range(res.shape[0]):
                util.box(py*20+(res[py,px]/(movy*2+1)-movy)*10, px*20+(res[py,px]%(movx*2+1)-movx)*10, py*20+(res[py,px]/(movy*2+1)-movy)*10+20, px*20+(res[py,px]%(movx*2+1)-movx)*10+20, col='b', lw=2)   
                pylab.text(px*20+(res[py,px]%(movx*2+1)-movx)*10,py*20+(res[py,px]/(movy*2+1)-movy)*10,"(%d,%d)"%(py,px))
           
    if feat:
        t=time.time()
        dfeat=numpy.zeros(m1.shape,dtype=numpy.float32)
        for px in range(p1.shape[1]):
            for py in range(p1.shape[0]):
                dfeat[py*2:py*2+2,px*2:px*2+2]=blk2pad[py*2+res[py,px]/(movx*2+1),px*2+res[py,px]%(movx*2+1)].reshape(2,2,31)
        edge=numpy.zeros(res2.shape,dtype=numpy.float32)
        #edge[0,:-1,:]=(res2[0,:-1]-res2[0,1:])
        #edge[1,:,:-1]=(res2[1,:,:-1]-res2[1,:,1:])
        edge[0,:-1,:]=abs(res2[0,:-1,:]-res2[0,1:,:])+abs(res2[1,:-1,:]-res2[1,1:,:])
        edge[1,:,:-1]=abs(res2[0,:,:-1]-res2[0,:,1:])+abs(res2[1,:,:-1]-res2[1,:,1:])
        #edge[0,:-1,:]=abs(dy)/(movx*2+1)+abs(dy)%(movx*2+1)
        #edge[1,:,:-1]=abs(dx)/(movx*2+1)+abs(dx)%(movx*2+1)
        #print "time feat",time.time()-t
        return scr,res2,dfeat,-edge

    return scr,res2

def match(m1,m2,cost,movy=None,movx=None,padvalue=0,pad=0,feat=True,show=True,rotate=False):
    t=time.time()
    #blk1=numpy.concatenate((m1[:-1:,:-1],m1[:-1,1:],m1[1:,:-1],m1[1:,1:]),2)
    #blk2=numpy.concatenate((m2[:-1:,:-1],m2[:-1,1:],m2[1:,:-1],m2[1:,1:]),2)
    #p1=blk1[::2,::2]
    numy=m1.shape[0]/2#p1.shape[0]
    numx=m1.shape[1]/2#p1.shape[1]
    #bb2=blk2.reshape((blk2.shape[0]*blk2.shape[1],-1)).T

    if movy==None:
        movy=((m1.shape[0]-2*pad)-1)/2
    if movx==None:
        movx=((m1.shape[1]-2*pad)-1)/2
    numlab=(movy*2+1)*(movx*2+1)
    #blk2pad=padvalue*numpy.ones((blk1.shape[0]+2*movy,blk1.shape[1]+2*movx,blk1.shape[2]))
    #blk2pad[movy-pad:-movy+pad,movx-pad:-movx+pad]=blk2
    data=numpy.zeros((numy,numx,(movy*2+1)*(movx*2+1)),dtype=numpy.float32)
    #print "time preparation",time.time()-t
    t=time.time()
#    t1=time.time()
#    for px in range(p1.shape[1]):
#        for py in range(p1.shape[0]):
#            data[py,px]=-numpy.sum(p1[py,px]*blk2pad[2*py:2*py+(2*movy+1),2*px:2*px+(2*movx+1)],2).flatten()
#    print "Time mode1",time.time()-t1

    #data1=numpy.zeros((numy,numx,(movy*2+1)*(movx*2+1)),dtype=numpy.float32)
    mmax=numpy.zeros(2,dtype=c_int)
    #t1=time.time()
    #m1=numpy.ascontiguousarray(m1)
    m2=numpy.ascontiguousarray(m2)
    for px in range(numx):
        for py in range(numy):
            ff.refineighfull(m2,m2.shape[0],m2.shape[1],numpy.ascontiguousarray(m1[2*py:2*(py+1),2*px:2*(px+1)]),
                2,2,m1.shape[2],0,0,py*2+pad,px*2+pad,movy,movx,data[py,px],
                mmax,mmax,ctypes.POINTER(c_float)(),0,0,0)
    #print "Time mode1",time.time()-t1
    
    #print "time hog",time.time()-t
    rdata=data.reshape((data.shape[0]*data.shape[1],-1))
    rmin=rdata.min()
    rdata=rdata-rmin

    res=numpy.zeros((numy,numx),dtype=c_int)
    #print "time matching",time.time()-t
    t=time.time()
    #print "before"
    scr=crfgr(numy,numx,cost,movy*2+1,movx*2+1,rdata,res)  
    #print "after"
    scr=scr-rmin*numy*numx
    res2=numpy.zeros((2,res.shape[0],res.shape[1]),dtype=c_int)
    res2[0]=(res/(movx*2+1)-movy)
    res2[1]=(res%(movx*2+1)-movx)
    #print "time config",time.time()-t

    if show:    
        print scr,res
        import pylab
        pylab.figure(10)
        pylab.clf()
        pylab.ion()
        #pylab.axis("image")
        aa=pylab.axis()
        pylab.axis([aa[0],aa[1],aa[3],aa[2]])
        import util
        for px in range(res.shape[1]):
            for py in range(res.shape[0]):
                util.box(py*20+(res[py,px]/(movy*2+1)-movy)*10, px*20+(res[py,px]%(movx*2+1)-movx)*10, py*20+(res[py,px]/(movy*2+1)-movy)*10+20, px*20+(res[py,px]%(movx*2+1)-movx)*10+20, col='b', lw=2)   
                pylab.text(px*20+(res[py,px]%(movx*2+1)-movx)*10,py*20+(res[py,px]/(movy*2+1)-movy)*10,"(%d,%d)"%(py,px))
           
    if feat:
        t=time.time()
        dfeat=numpy.zeros(m1.shape,dtype=numpy.float32)
        m2pad=numpy.zeros((m2.shape[0]+2*movy-2*pad,m2.shape[1]+2*movx-2*pad,m2.shape[2]),dtype=numpy.float32)
        m2pad[movy-pad:-movy+pad,movx-pad:-movx+pad]=m2
        for px in range(numx):
            for py in range(numy):
                #dfeat[py*2:py*2+2,px*2:px*2+2]=blk2pad[py*2+res[py,px]/(movx*2+1),px*2+res[py,px]%(movx*2+1)].reshape(2,2,31)
                cpy=py*2+res2[0,py,px]+movy
                cpx=px*2+res2[1,py,px]+movx    
                dfeat[py*2:py*2+2,px*2:px*2+2]=m2pad[cpy:cpy+2,cpx:cpx+2]
        edge=numpy.zeros(res2.shape,dtype=numpy.float32)
        #edge[0,:-1,:]=(res2[0,:-1]-res2[0,1:])
        #edge[1,:,:-1]=(res2[1,:,:-1]-res2[1,:,1:])
        edge[0,:-1,:]=abs(res2[0,:-1,:]-res2[0,1:,:])+abs(res2[1,:-1,:]-res2[1,1:,:])
        edge[1,:,:-1]=abs(res2[0,:,:-1]-res2[0,:,1:])+abs(res2[1,:,:-1]-res2[1,:,1:])
        #edge[0,:-1,:]=abs(dy)/(movx*2+1)+abs(dy)%(movx*2+1)
        #edge[1,:,:-1]=abs(dx)/(movx*2+1)+abs(dx)%(movx*2+1)
        #print "time feat",time.time()-t
        return scr,res2,dfeat,-edge

    return scr,res2

def rotate(hog,shift=1):
    """
    rotate each hog cell of a certain shift
    """
    if shift==0:
        return hog
    hbin=9
    rhog=numpy.zeros(hog.shape)
    rhog[:,:,:18]=hog[:,:,numpy.mod(numpy.arange(shift,hbin*2+shift),hbin*2)]
    rhog[:,:,18:27]=hog[:,:,numpy.mod(numpy.arange(shift,hbin+shift),hbin)+18]
    rhog[:,:,27:]=hog[:,:,27:]
    return rhog

def match_new(m1,m2,cost,movy=None,movx=None,padvalue=0,pad=0,feat=True,show=True,rotate=False):
    t=time.time()
    numy=m1.shape[0]/2#p1.shape[0]
    numx=m1.shape[1]/2#p1.shape[1]

    if movy==None:
        movy=((m1.shape[0]-2*pad)-1)/2
    if movx==None:
        movx=((m1.shape[1]-2*pad)-1)/2
    numlab=(movy*2+1)*(movx*2+1)
    data=numpy.zeros((numy,numx,(movy*2+1)*(movx*2+1)),dtype=numpy.float32)
    auxdata=numpy.zeros((3,numy,numx,(movy*2+1)*(movx*2+1)),dtype=numpy.float32)
    #print "time preparation",time.time()-t
    t=time.time()
    mmax=numpy.zeros(2,dtype=c_int)
    #original model
    m2=numpy.ascontiguousarray(m2)
    for px in range(numx):
        for py in range(numy):
            ff.refineighfull(m2,m2.shape[0],m2.shape[1],numpy.ascontiguousarray(m1[2*py:2*(py+1),2*px:2*(px+1)]),
                2,2,m1.shape[2],0,0,py*2+pad,px*2+pad,movy,movx,auxdata[1,py,px],
                mmax,mmax,ctypes.POINTER(c_float)(),0,0,0)
    if rotate:
        #rotate +1
        m2=numpy.ascontiguousarray(rotate(m2,shift=1))
        for px in range(numx):
            for py in range(numy):
                ff.refineighfull(m2,m2.shape[0],m2.shape[1],numpy.ascontiguousarray(m1[2*py:2*(py+1),2*px:2*(px+1)]),
                    2,2,m1.shape[2],0,0,py*2+pad,px*2+pad,movy,movx,auxdata[2,py,px],
                    mmax,mmax,ctypes.POINTER(c_float)(),0,0,0)
        #rotate -1
        m2=numpy.ascontiguousarray(rotate(m2,shift=-1))
        for px in range(numx):
            for py in range(numy):
                ff.refineighfull(m2,m2.shape[0],m2.shape[1],numpy.ascontiguousarray(m1[2*py:2*(py+1),2*px:2*(px+1)]),
                    2,2,m1.shape[2],0,0,py*2+pad,px*2+pad,movy,movx,auxdata[0,py,px],
                    mmax,mmax,ctypes.POINTER(c_float)(),0,0,0)

        #print "time hog",time.time()-t
        data=numpy.max(auxdata,0)
        rot=numpy.argmax(auxdata,0)
        rdata=data.reshape((data.shape[0]*data.shape[1],-1))
        rrot=rot.reshape((rot.shape[0]*rot.shape[1]))
    else:
        data=auxdata[1]
        rdata=data.reshape((data.shape[0]*data.shape[1],-1))
    rmin=rdata.min()
    rdata=rdata-rmin

    res=numpy.zeros((numy,numx),dtype=c_int)
    #print "time matching",time.time()-t
    t=time.time()
    #print "before"
    scr=crfgr(numy,numx,cost,movy*2+1,movx*2+1,rdata,res)  
    #print "after"
    scr=scr-rmin*numy*numx
    res2=numpy.zeros((2,res.shape[0],res.shape[1]),dtype=c_int)
    res2[0]=(res/(movx*2+1)-movy)
    res2[1]=(res%(movx*2+1)-movx)
    #print "time config",time.time()-t

    if show:    
        print scr,res
        import pylab
        pylab.figure(10)
        pylab.clf()
        pylab.ion()
        #pylab.axis("image")
        aa=pylab.axis()
        pylab.axis([aa[0],aa[1],aa[3],aa[2]])
        import util
        for px in range(res.shape[1]):
            for py in range(res.shape[0]):
                util.box(py*20+(res[py,px]/(movy*2+1)-movy)*10, px*20+(res[py,px]%(movx*2+1)-movx)*10, py*20+(res[py,px]/(movy*2+1)-movy)*10+20, px*20+(res[py,px]%(movx*2+1)-movx)*10+20, col='b', lw=2)   
                pylab.text(px*20+(res[py,px]%(movx*2+1)-movx)*10,py*20+(res[py,px]/(movy*2+1)-movy)*10,"(%d,%d)"%(py,px))
           
    if feat:
        t=time.time()
        dfeat=numpy.zeros(m1.shape,dtype=numpy.float32)
        m2pad=numpy.zeros((m2.shape[0]+2*movy-2*pad,m2.shape[1]+2*movx-2*pad,m2.shape[2]),dtype=numpy.float32)
        m2pad[movy-pad:-movy+pad,movx-pad:-movx+pad]=m2
        for px in range(numx):
            for py in range(numy):
                #dfeat[py*2:py*2+2,px*2:px*2+2]=blk2pad[py*2+res[py,px]/(movx*2+1),px*2+res[py,px]%(movx*2+1)].reshape(2,2,31)
                cpy=py*2+res2[0,py,px]+movy
                cpx=px*2+res2[1,py,px]+movx    
                dfeat[py*2:py*2+2,px*2:px*2+2]=m2pad[cpy:cpy+2,cpx:cpx+2]
        edge=numpy.zeros(res2.shape,dtype=numpy.float32)
        #edge[0,:-1,:]=(res2[0,:-1]-res2[0,1:])
        #edge[1,:,:-1]=(res2[1,:,:-1]-res2[1,:,1:])
        edge[0,:-1,:]=abs(res2[0,:-1,:]-res2[0,1:,:])+abs(res2[1,:-1,:]-res2[1,1:,:])
        edge[1,:,:-1]=abs(res2[0,:,:-1]-res2[0,:,1:])+abs(res2[1,:,:-1]-res2[1,:,1:])
        #edge[0,:-1,:]=abs(dy)/(movx*2+1)+abs(dy)%(movx*2+1)
        #edge[1,:,:-1]=abs(dx)/(movx*2+1)+abs(dx)%(movx*2+1)
        #print "time feat",time.time()-t
        return scr,res2,dfeat,-edge

    return scr,res2



def getfeat(m1,pad,res2,movy=None,movx=None,mode="Best"):
    if movy==None:
        movy=((m1.shape[0]-2*pad)-1)/2
    if movx==None:
        movx=((m1.shape[1]-2*pad)-1)/2
    dfeat=numpy.zeros((m1.shape[0]-2*pad,m1.shape[1]-2*pad,m1.shape[2]),dtype=numpy.float32)
    m1pad=numpy.zeros((m1.shape[0]+2*movy-2*pad,m1.shape[1]+2*movx-2*pad,m1.shape[2]),dtype=numpy.float32)
    m1pad[movy-pad:-movy+pad,movx-pad:-movx+pad]=m1
    for px in range(res2.shape[2]):
        for py in range(res2.shape[1]):
            #dfeat[py*2:py*2+2,px*2:px*2+2]=blk2pad[py*2+res[py,px]/(movx*2+1),px*2+res[py,px]%(movx*2+1)].reshape(2,2,31)
            cpy=py*2+res2[0,py,px]+movy
            cpx=px*2+res2[1,py,px]+movx    
            dfeat[py*2:py*2+2,px*2:px*2+2]=m1pad[cpy:cpy+2,cpx:cpx+2]
    edge=numpy.zeros((res2.shape[0]*2,res2.shape[1],res2.shape[2]),dtype=numpy.float32)
    #edge[0,:-1,:]=(res2[0,:-1]-res2[0,1:])
    #edge[1,:,:-1]=(res2[1,:,:-1]-res2[1,:,1:])
    if mode=="Old":
        edge[0,:-1,:]=abs(res2[0,:-1,:]-res2[0,1:,:])+abs(res2[1,:-1,:]-res2[1,1:,:])
        edge[1,:,:-1]=abs(res2[0,:,:-1]-res2[0,:,1:])+abs(res2[1,:,:-1]-res2[1,:,1:])
    elif mode=="New":
        edge[0,:-1,:]=abs(res2[0,:-1,:]-res2[0,1:,:])
        edge[1,:,:-1]=abs(res2[1,:,:-1]-res2[1,:,1:])
    elif mode=="Best":
        edge[0,:-1,:]=abs(res2[0,:-1,:]-res2[0,1:,:])
        edge[1,:-1,:]=abs(res2[1,:-1,:]-res2[1,1:,:])
        edge[2,:,:-1]=abs(res2[0,:,:-1]-res2[0,:,1:])
        edge[3,:,:-1]=abs(res2[1,:,:-1]-res2[1,:,1:])
    return dfeat,-edge    

def getfeat_new(m1,pad,res2,movy=None,movx=None,mode="Best",rot=None):
    if movy==None:
        movy=((m1.shape[0]-2*pad)-1)/2
    if movx==None:
        movx=((m1.shape[1]-2*pad)-1)/2
    dfeat=numpy.zeros((m1.shape[0]-2*pad,m1.shape[1]-2*pad,m1.shape[2]),dtype=numpy.float32)
    m1pad=numpy.zeros((m1.shape[0]+2*movy-2*pad,m1.shape[1]+2*movx-2*pad,m1.shape[2]),dtype=numpy.float32)
    m1pad[movy-pad:-movy+pad,movx-pad:-movx+pad]=m1
    for px in range(res2.shape[2]):
        for py in range(res2.shape[1]):
            #dfeat[py*2:py*2+2,px*2:px*2+2]=blk2pad[py*2+res[py,px]/(movx*2+1),px*2+res[py,px]%(movx*2+1)].reshape(2,2,31)
            cpy=py*2+res2[0,py,px]+movy
            cpx=px*2+res2[1,py,px]+movx    
            if rot==None:
                dfeat[py*2:py*2+2,px*2:px*2+2]=m1pad[cpy:cpy+2,cpx:cpx+2]
            else:
                dfeat[py*2:py*2+2,px*2:px*2+2]=rotate(m1pad[cpy:cpy+2,cpx:cpx+2],rot[py,px])
    edge=numpy.zeros((res2.shape[0]*2,res2.shape[1],res2.shape[2]),dtype=numpy.float32)
    #edge[0,:-1,:]=(res2[0,:-1]-res2[0,1:])
    #edge[1,:,:-1]=(res2[1,:,:-1]-res2[1,:,1:])
    if mode=="Old":
        edge[0,:-1,:]=abs(res2[0,:-1,:]-res2[0,1:,:])+abs(res2[1,:-1,:]-res2[1,1:,:])
        edge[1,:,:-1]=abs(res2[0,:,:-1]-res2[0,:,1:])+abs(res2[1,:,:-1]-res2[1,:,1:])
    elif mode=="New":
        edge[0,:-1,:]=abs(res2[0,:-1,:]-res2[0,1:,:])
        edge[1,:,:-1]=abs(res2[1,:,:-1]-res2[1,:,1:])
    elif mode=="Best":
        edge[0,:-1,:]=abs(res2[0,:-1,:]-res2[0,1:,:])
        edge[1,:-1,:]=abs(res2[1,:-1,:]-res2[1,1:,:])
        edge[2,:,:-1]=abs(res2[0,:,:-1]-res2[0,:,1:])
        edge[3,:,:-1]=abs(res2[1,:,:-1]-res2[1,:,1:])
    return dfeat,-edge    


if __name__ == "__main__":

    if 1:#example with HOG
        import util
        #model1=util.load("./data/CRF/12_04_27/bicycle2_NoCRF9.model")
        #model2=util.load("./data/CRF/12_04_27/bicycle2_NoCRFNoDef9.model")
        model1=util.load("./data/rigid/12_08_17/bicycle3_complete8.model")
        model2=util.load("data/CF/12_08_15/bicycle3_newcache2.model")
        m1=model1[0]["ww"][2]
        m2=model2[0]["ww"][2]    
        pad=2
        m3=numpy.zeros((m2.shape[0]+2*pad,m2.shape[1]+2*pad,m2.shape[2]),dtype=numpy.float32)
        m3[pad:-pad,pad:-pad]=m2
        m2=m3

        numy=m1.shape[0]/2
        numx=m1.shape[1]/2
        movy=(numy*2-1)/2
        movx=(numx*2-1)/2
        mcost=0.01*numpy.ones((4,numy,numx),dtype=c_float)
        #mcost[0,-1,:]=0#vertical 
        #mcost[0,0,:]=0#added v
        #mcost[1,:,-1]=0#horizontal
        #mcost[1,:,0]=0#added o
        mcost[0]=mcost[0]*1
        mcost[1]=mcost[1]*1
        #mcost[1,0,:]=1
        #costV=costV(numy,numx,movy,movx,c=0.001,ch=mcost[0],cv=mcost[1])
        t=time.time()
        #scr,res,dfeat,edge=match(m1,m2,mcost,movy=m1.shape[0]/4,movx=m1.shape[1]/4,pad=pad,show=False)
        scr,res=match(m1,m2,mcost,movy=m1.shape[0]/4,movx=m1.shape[1]/4,pad=pad,show=False,feat=False)
        print "Total time",time.time()-t
        print "Score",scr
        dfeat,edge=getfeat(m2,pad,res)
        print "Error",numpy.sum(m1*dfeat)+numpy.sum(edge*mcost)-scr
        import pylab
        import drawHOG
        img=drawHOG.drawHOG(dfeat)
        pylab.figure(11);pylab.imshow(img)
        pylab.show()
        #img2=drawHOG.drawHOG(dfeat2)
        #pylab.figure(12);pylab.imshow(img2)


