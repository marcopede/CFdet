#class used to manage the pyramid HOG features
import resize
import util2 as util
import numpy
import math
import pylab
#import scipy.misc.pilutils
import cPickle
import time

SMALL=100 #coefficient to multiply resolution features
DENSE=0 #number of levels to use a dense scan instead of a Ctf
#BOW=False
K=1.0 #0.3 #coefficient for the deformation featres

from numpy import ctypeslib
from ctypes import c_int,c_double,c_float
import ctypes

#library to compute HOGs
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
#lhog.process.argtypes=[
#    numpy.ctypeslib.ndpointer(dtype=c_double,ndim=3,flags="F_CONTIGUOUS")#im
#    ,c_int #dimy
#    ,c_int #dimx
#    ,c_int #sbin
#    ,numpy.ctypeslib.ndpointer(dtype=c_double,ndim=3,flags="F_CONTIGUOUS")# *hog (round(dimy/float(sbin))-2,round(dimx/float(sbin))-2,31)
#    ,c_int #hy
#    ,c_int #hx
#    ,c_int #hz
#    ]

#library to compute correlation between object model and HOGs
ctypes.cdll.LoadLibrary("./libexcorr.so")
ff= ctypes.CDLL("libexcorr.so")
###
#corr3dpadbow(ftype *img,int imgy,int imgx,ftype *mask,int masky,int maskx,int dimz,int posy,int posx,ftype *prec,int pady,int padx,int occl,int sizevoc,int numvoc,ftype *voc,ftype *mhist
ff.corr3dpadbow.argtypes=[numpy.ctypeslib.ndpointer(dtype=c_float,ndim=3,flags="C_CONTIGUOUS"),c_int,c_int,numpy.ctypeslib.ndpointer(dtype=c_float,ndim=3,flags="C_CONTIGUOUS"),c_int,c_int,c_int,
c_int,c_int,ctypes.POINTER(c_float),c_int,c_int,c_int,c_int,c_int,
numpy.ctypeslib.ndpointer(dtype=c_int,flags="C_CONTIGUOUS"),c_int,numpy.ctypeslib.ndpointer
(dtype=c_float,flags="C_CONTIGUOUS")]
ff.corr3dpadbow.restype=ctypes.c_float
###
ff.scaneigh.argtypes=[numpy.ctypeslib.ndpointer(dtype=c_float,ndim=3,flags="C_CONTIGUOUS"),c_int,c_int,numpy.ctypeslib.ndpointer(dtype=c_float,ndim=3,flags="C_CONTIGUOUS"),c_int,c_int,c_int,numpy.ctypeslib.ndpointer(dtype=c_int,flags="C_CONTIGUOUS"),numpy.ctypeslib.ndpointer(dtype=c_int,flags="C_CONTIGUOUS"),numpy.ctypeslib.ndpointer(dtype=c_float,flags="C_CONTIGUOUS"),numpy.ctypeslib.ndpointer(dtype=c_int,flags="C_CONTIGUOUS"),numpy.ctypeslib.ndpointer
(dtype=c_int,flags="C_CONTIGUOUS"),c_int,c_int,c_int,c_int]

ff.scaneighbow.argtypes=[numpy.ctypeslib.ndpointer(dtype=c_float,ndim=3,flags="C_CONTIGUOUS"),c_int,c_int,numpy.ctypeslib.ndpointer(dtype=c_float,ndim=3,flags="C_CONTIGUOUS"),c_int,c_int,c_int,numpy.ctypeslib.ndpointer(dtype=c_int,flags="C_CONTIGUOUS"),numpy.ctypeslib.ndpointer(dtype=c_int,flags="C_CONTIGUOUS"),numpy.ctypeslib.ndpointer(dtype=c_float,flags="C_CONTIGUOUS"),numpy.ctypeslib.ndpointer(dtype=c_int,flags="C_CONTIGUOUS"),numpy.ctypeslib.ndpointer
(dtype=c_int,flags="C_CONTIGUOUS"),c_int,c_int,c_int,c_int,
c_int,c_int,numpy.ctypeslib.ndpointer(dtype=c_float,flags="C_CONTIGUOUS"),numpy.ctypeslib.ndpointer(dtype=c_float,flags="C_CONTIGUOUS")]


ff.scanDef2.argtypes = [
    ctypeslib.ndpointer(c_float),ctypeslib.ndpointer(c_float),ctypeslib.ndpointer(c_float),ctypeslib.ndpointer(c_float),
    c_int,c_int,c_int,
    ctypeslib.ndpointer(c_float),ctypeslib.ndpointer(c_float),ctypeslib.ndpointer(c_float),ctypeslib.ndpointer(c_float),
    ctypeslib.ndpointer(c_float),
    c_int,c_int,ctypeslib.ndpointer(c_int),ctypeslib.ndpointer(c_int),
    ctypeslib.ndpointer(c_int),
    ctypeslib.ndpointer(c_float),
    c_int,c_int,c_int,ctypeslib.ndpointer(c_float),c_int,
    ctypes.POINTER(c_float),c_int,c_int,c_int]
ff.scanDef2.restype=ctypes.c_float

ff.scanDefbow.argtypes = [
    ctypeslib.ndpointer(c_float),ctypeslib.ndpointer(c_float),ctypeslib.ndpointer(c_float),ctypeslib.ndpointer(c_float),
    c_int,c_int,c_int,
    ctypeslib.ndpointer(c_float),ctypeslib.ndpointer(c_float),ctypeslib.ndpointer(c_float),ctypeslib.ndpointer(c_float),
    ctypeslib.ndpointer(c_float),
    c_int,c_int,ctypeslib.ndpointer(c_int),ctypeslib.ndpointer(c_int),
    ctypeslib.ndpointer(c_int),
    ctypeslib.ndpointer(c_float),
    c_int,c_int,c_int,ctypeslib.ndpointer(c_float),c_int,
    ctypes.POINTER(c_float),c_int,c_int,c_int,
    c_int,c_int,ctypeslib.ndpointer(c_float),ctypeslib.ndpointer(c_float)]
ff.scanDef2.restype=ctypes.c_float


ff.setK.argtypes = [c_float]
ff.setK(K)

ff.getK.argtypes = []
ff.getK.restype = ctypes.c_float
ff.setK(K)

def setK(pk):
    ff.setK(pk)    

def decompose(model,l=0,py=0,px=0,ppy=0,ppx=0):
        if len(model["ww"])<=l:
            return []
        model2={}
#        if l==0:
#            model2["ww"]=model["ww"][0]
#            model2["df"]=model["df"][0]
#            model2["parts"]=[decompose(model,1,0,0),decompose(model,1,0,1),decompose(model,1,1,0),decompose(model,1,1,1)]
#        else:
        dy=model["ww"][0].shape[0]
        dx=model["ww"][0].shape[1]
        model2["ww"]=model["ww"][l][(py+ppy)*dy:(py+ppy+1)*dy,(px+ppx)*dx:(px+ppx+1)*dx].copy()
        model2["df"]=model["df"][l][py+ppy:(py+ppy+1),px+ppx:(px+ppx+1)].copy()
        #model2["rho"]=model["rho"]
        model2["base"]=[dy,dx]
        model2["len"]=len(model["ww"])
        model2["parts"]=[decompose(model,l+1,ppy*2,ppx*2,0,0),decompose(model,l+1,ppy*2,ppx*2,0,1),decompose(model,l+1,ppy*2,ppx*2,1,0),decompose(model,l+1,ppy*2,ppx*2,1,1)]
        return model2

def getfeat(a,y1,y2,x1,x2,trunc=0):
    """
        returns the hog features at the given position and 
        zeros in case the coordiantes are outside the borders
        """
    dimy=a.shape[0]
    dimx=a.shape[1]
    py1=y1;py2=y2;px1=x1;px2=x2
    dy1=0;dy2=0;dx1=0;dx2=0
    #if trunc>0:
    b=numpy.zeros((abs(y2-y1),abs(x2-x1),a.shape[2]+trunc))
    if trunc>0:
        b[:,:,-trunc]=1
    #else:
    #    b=numpy.zeros((abs(y2-y1),abs(x2-x1),a.shape[2]))
    if py1<0:
        py1=0
        dy1=py1-y1
    if py2>=dimy:
        py2=dimy
        dy2=y2-py2
    if px1<0:
        px1=0
        dx1=px1-x1
    if px2>=dimx:
        px2=dimx
        dx2=x2-px2
    if numpy.array(a[py1:py2,px1:px2].shape).min()==0 or numpy.array(b[dy1:y2-y1-dy2,dx1:x2-x1-dx2].shape).min()==0:
        pass
    else:
        if trunc==1:
            b[dy1:y2-y1-dy2,dx1:x2-x1-dx2,:-1]=a[py1:py2,px1:px2]
            #b[:,:,-1]=1
            b[dy1:y2-y1-dy2,dx1:x2-x1-dx2,-1]=0
        else:
            b[dy1:y2-y1-dy2,dx1:x2-x1-dx2]=a[py1:py2,px1:px2]
    return b


def hog2bow_5(feat,bin=5,siftsize=2):
    selbin=numpy.array([0,2,4,5,7])
    hist=numpy.zeros(bin**(siftsize**2))
    hog=feat[:,:,18:27].copy()
    import PySegment
    bow=PySegment.hogtosift(hog,siftsize,geom=False)
    #print len(bow)
    i=0
    for ll in bow:
        bow1=ll.reshape((4,9)).astype(numpy.float32)
        val=2*bow1[:,selbin]+bow1[:,selbin+1]+bow1[:,(selbin+8)%9]
        pp=numpy.sum(val.argmax(1)*numpy.array([1,bin,bin**2,bin**3]))
        hist[pp]=1.0#new value
        #print i,pp,val.argmax(1),val.max(1)
        i+=1
    hist=hist/numpy.sqrt(numpy.sum(hist**2))
    return hist.astype(numpy.float32)

def hog2bow_old(feat,bin=6,siftsize=2,pr=False,code=False):
    selbin=numpy.array([0,2,4,5,7])
    hist=numpy.zeros(bin**(siftsize**2))
    hog=feat[:,:,18:27].copy()
    import PySegment
    bow=PySegment.hogtosift(hog,siftsize,geom=False)
    #print len(bow)
    i=0
    bcode=numpy.zeros((numpy.array(feat.shape[:2])-siftsize+1),dtype=numpy.int)
    for ll in bow:
        bow1=ll.reshape((4,9)).astype(numpy.float32)
        val=2*bow1[:,selbin]+bow1[:,selbin+1]+bow1[:,(selbin+8)%9]
        mmax=val.argmax(1)
        mmax[val.max(1)==0.0]=bin-1
        pp=numpy.sum(mmax*numpy.array([1,bin,bin**2,bin**3]))
        hist[pp]=1.0#new value
        bcode[i/bcode.shape[1],i%bcode.shape[1]]=pp
        if pr:
            print i,pp,mmax,val.max(1)
        i+=1
    hist=hist/numpy.sqrt(numpy.sum(hist**2))
    if code:
        return bcode.astype(numpy.int32)
    return hist.astype(numpy.float32)

#inline ftype hog2bow(ftype *img,int imgx,int imgy,ftype *mask,int masky,int maskx,int dimz,int posy,int posx,int sizevoc,int *code)
ff.hog2bow.argtypes=[numpy.ctypeslib.ndpointer(dtype=c_float,ndim=3,flags="C_CONTIGUOUS"),c_int,c_int,numpy.ctypeslib.ndpointer(dtype=c_float,flags="C_CONTIGUOUS"),c_int,c_int,c_int,c_int,c_int,c_int,
numpy.ctypeslib.ndpointer(dtype=c_int,flags="C_CONTIGUOUS")]
#ff.hog2bow.restype=ctypes.c_float

def hog2bow(feat,bin=6,siftsize=2,pr=False,code=False,db=False):
    hist=numpy.zeros(bin**(siftsize**2),dtype=numpy.float32)
    shy=feat.shape[0]
    shx=feat.shape[1]
    bcode=numpy.zeros((shy-1,shx-1),dtype=numpy.int32)
    feat1=feat.astype(numpy.float32)#.copy()
    numvoc=ctypes.c_int(bin**(siftsize**2))
    mask=numpy.zeros((10,10),dtype=numpy.float32)
    if db:
        mask[:]=10.0
    ff.hog2bow(feat1,shx,shy,mask,shy,shx,31,0,0,2,bcode)
    for l in bcode.flatten():
        hist[l]=1.0
    hist=10.0*hist/numpy.sqrt(numpy.sum(hist**2))
    if code:
        return bcode
    return hist

def hog2bow_slow(feat,bin=6,siftsize=2,pr=False,code=False):
    selbin=numpy.array([0,2,4,5,7])
    nrm=numpy.array([3,1,2,0],dtype=numpy.int)
    hist=numpy.zeros(bin**(siftsize**2))
    hog=feat[:,:,18:].astype(numpy.float32)#copy()
    import PySegment
    bow=PySegment.hogtosift(hog,siftsize,geom=False)
    #print len(bow)
    i=0
    bcode=numpy.zeros((numpy.array(feat.shape[:2])-siftsize+1),dtype=numpy.int)
    for ll in bow:
        bow1=ll.reshape((4,13)).astype(numpy.float32)
        val=2*bow1[:,selbin]+bow1[:,selbin+1]+bow1[:,(selbin+8)%9]
        mmax=val.argmax(1)
        mmax[val.max(1)==0.0]=bin-1
        mmax[bow1[numpy.arange(4),9+nrm]<0.1]=bin-1
        pp=numpy.sum(mmax*numpy.array([1,bin,bin**2,bin**3]))
        hist[pp]=1.0#new value
        bcode[i%bcode.shape[0],i/bcode.shape[0]]=pp
        if pr:
            print i,pp,mmax,val.max(1)
        i+=1
    hist=hist/numpy.sqrt(numpy.sum(hist**2))
    if code:
        return bcode.astype(numpy.int32)
    return hist.astype(numpy.float32)


def hogtosift(hog,siftsize,geom=False):
    him=hog#numpy.ascontiguousarray(hog)
    #print hog.flags
    #raw_input()
    himy=him.shape[0]-siftsize+1
    himx=him.shape[1]-siftsize+1
    sift=numpy.zeros((himy,himx,siftsize,siftsize,hog.shape[2]))
    for sy in range(siftsize):
        for sx in range(siftsize):
            sift[:,:,sy,sx]=him[sy:sy+himy,sx:sx+himx]		
    sim=sift.reshape((himy,himx,siftsize**2*hog.shape[2]))	
    #sim=sift.reshape((himx,himy,siftsize**2*hog.shape[2]))
    #showImage(sim)
    feat=sim.T.reshape((sim.shape[2],himy*himx)).T
    if 0:
        pylab.figure()
        showHOGflat(hog,siftsize)
        pylab.figure()
        showBook(feat[:100],siftsize)
        pylab.draw()
        pylab.show()
        raw_input()
    if geom:
        return feat,sim
    return feat

def hog2bowrec(feat,bin=6,siftsize=2,pr=False,code=False):
    sel=numpy.array([0,2,4,5,7])#orientations
    nrm=numpy.array([3,1,2,0],dtype=numpy.int)#normalizations
    #nrm=numpy.array([3,2,1,0],dtype=numpy.int)#normalizations
    hist=numpy.zeros(bin**(siftsize**2))
    hog=feat[:,:,18:].copy()
    #import PySegment
    bow,bowg=hogtosift(hog,siftsize,geom=True)
    #print len(bow)
    i=0
    import drawHOG
    pylab.gray()
    pylab.figure()
    im1=drawHOG.drawHOG(feat)
    pylab.axis("off")
    pylab.imshow(im1)     
    #bcode=numpy.zeros((numpy.array(feat.shape[:2])-siftsize+1),dtype=numpy.int)
    rec=numpy.zeros(feat.shape,dtype=numpy.float32)
    for ly in range(bowg.shape[0]):
        for lx in range(bowg.shape[1]):
            bow1=bowg[ly,lx].reshape((4,13))
            val=2*bow1[:,sel]+bow1[:,sel+1]+bow1[:,(sel+8)%9]
            mmax=val.argmax(1)
            mmax[val.max(1)==0.0]=bin-1
            mmax[bow1[numpy.arange(4),9+nrm]<0.15]=bin-1
            print bow1[numpy.arange(4),9+nrm]
            pp=numpy.sum(mmax*numpy.array([1,bin,bin**2,bin**3]))
            hist[pp]=1.0#new value
            #bcode[i/bcode.shape[1],i%bcode.shape[1]]=pp
            for y in range(2):
                for x in range(2):
                    pp=mmax[x+y*2]
                    if pp!=5:
                        rec[ly+y,lx+x,18+sel[pp]]+=1.0
            #rec[ly:ly+sizesift,lx:lx+sizesift]=
            if pr:
                print i,pp,mmax,val.max(1)
            i+=1
    pylab.figure()
    im2=drawHOG.drawHOG(rec)
    pylab.axis("off")
    pylab.imshow(im2)    
    hist=hist/numpy.sqrt(numpy.sum(hist**2))
    if code:
        return bcode.astype(numpy.int32)
    return hist.astype(numpy.float32)

def showbow(hist,num=100,order=1,onlypos=True):
    sel=numpy.array([0,2,4,5,7])
    if onlypos:
        srt=numpy.argsort(-order*(hist))
    else:
        srt=numpy.argsort(-order*numpy.abs(hist))
    pylab.figure()
    ny=numpy.sqrt(num)
    nx=num/ny+1
    from util import baseconvert as baseconvert
    import drawHOG
    pylab.axis("off")
    pylab.gray()
    for i in range(num):
        nn=baseconvert(srt[i],tondigits=6, mindigits=4)
        print i,nn,hist[srt[i]]
        hog=numpy.zeros((2,2,31))    
        for y in range(2):
            for x in range(2):
                pp=int(nn[x+2*y])
                if pp!=5:
                    hog[y,x,18+sel[pp]]=1.0
        im=drawHOG.drawHOG(hog)
        pylab.subplot(nx,ny,i+1)        
        pylab.axis("off")
        pylab.text(0,0,"%.3f"%(hist[srt[i]]*1000),fontsize=8)
        pylab.imshow(im)
    pylab.show()

def histflip_5(bin=5,siftsize=2):
    flip=[0,4,3,2,1]
    ftab=numpy.zeros(5**4,dtype=numpy.int)
    for l0 in range(5):
        for l1 in range(5):
            for l2 in range(5):
                for l3 in range(5):
                    val=l3+l2*5+l1*25+l0*125
                    ftab[val]=flip[l3]*5+flip[l2]+flip[l1]*125+flip[l0]*25      
    return ftab

def histflip(bin=6,siftsize=2):
    flip=[0,4,3,2,1,5]
    ftab=numpy.zeros(bin**(siftsize**2),dtype=numpy.int)
    for l0 in range(bin):
        for l1 in range(bin):
            for l2 in range(bin):
                for l3 in range(bin):
                    val=l3+l2*bin+l1*(bin**2)+l0*(bin**3)
                    ftab[val]=flip[l3]*bin+flip[l2]+flip[l1]*(bin**3)+flip[l0]*(bin**2)      
                    #ftab[val]=flip[l3]*(bin**2)+flip[l2]*(bin**3)+flip[l1]+flip[l0]*(bin**1)      
    return ftab


#def hog2bow2(feat,bin=5,siftsize=2):
#    selbin=numpy.array([0,2,4,5,7])
#    hist=numpy.zeros(bin**(siftsize**2))
#    hog=feat[:,:,18:27].copy()
#    import PySegment
#    bow=PySegment.hogtosift(hog,siftsize,geom=False)
#    print len(bow)
#    i=0
#    for ll in bow:
#        bow1=ll.reshape((4,9)).astype(numpy.float32)
#        val=2*bow1[:,selbin]+bow1[:,selbin+1]+bow1[:,(selbin-1)%9]
#        pp=numpy.sum(val.argmax(1)*numpy.array([1,bin,bin**2,bin**3]))
#        hist[pp]=1
#        print i,val.argmax(1),val.max(1),pp
#        i+=1
#    hist=hist/numpy.sqrt(numpy.sum(hist**2))
#    return bow,hist.astype(numpy.float32)

#wrapper for the HOG computation
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

def hogd(img,sbin=8):
    """
    Compute the HOG descriptor of an image
    """
    if type(img)!=numpy.ndarray:
        raise "img must be a ndarray"
    if img.ndim!=3:
        raise "img must have 3 dimensions"
    hy=int(round(img.shape[0]/float(sbin)))-2
    hx=int(round(img.shape[1]/float(sbin)))-2
    mtype=c_double
    hog=numpy.zeros((hy,hx,31),dtype=mtype,order="f")
    lhog.process(numpy.asfortranarray(img,dtype=mtype),img.shape[0],img.shape[1],sbin,hog,hy,hx,31)
    return hog;#mfeatures.mfeatures(img , sbin);


def hogflip(feat,obin=9):
    """    
    returns the horizontally flipped version of the HOG features
    """
    #feature shape
    #[18 oriented][9 not oriented][4 normalization]
    if feat.shape[2]==31:
        p=numpy.array([10,9,8,7,6,5,4,3,2,1,18,17,16,15,14,13,12,11,19,27,26,25,24,23,22,21,20,30,31,28,29])-1
        #p=numpy.array([10,9,8,7,6,5,4,3,2,1,18,17,16,15,14,13,12,11,19,27,26,25,24,23,22,21,20,29,28,31,30])-1
    else:
        p=numpy.array([10,9,8,7,6,5,4,3,2,1,18,17,16,15,14,13,12,11,19,27,26,25,24,23,22,21,20,30,31,28,29,32])-1
    aux=feat[:,::-1,p]
    return numpy.ascontiguousarray(aux)


def crfflip(edge):
    aux=edge.copy()#numpy.zeros(edge)
    aux[0]=edge[0,:,::-1]#aux[0,:,::-1]
    aux[1,:,:-1]=edge[1,:,-2::-1]#aux[1,:,-2::-1]
    #aux[1,:,-1]=0
    return aux


def defflip(feat):
    """    
    returns the horizontally flipped version of the deformation features
    """
    sx=feat.shape[1]/2-1
    fflip=numpy.zeros(feat.shape,dtype=feat.dtype)
    for ly in range(feat.shape[0]/2):
        for lx in range(feat.shape[1]/2):
            fflip[ly*2:(ly+1)*2,lx*2:(lx+1)*2]=feat[ly*2:(ly+1)*2,(sx-lx)*2:(sx-lx+1)*2].T
    return fflip

#auxiliary class
class container(object):
    def __init__(self,objarray,ptrarray):
        self.obj=objarray
        self.ptr=ptrarray

#hogbuffer={}

def HOGi2f(pyr,outtype,maxf=2.0):
    if len(pyr)==0:
        return []
    intype=pyr[0].dtype
    maxin=numpy.iinfo(intype).max
    npyr=[]
    for l in pyr:
        npyr.append((l.astype(outtype)/maxin)*maxf)
    return npyr
        
def HOGf2i(pyr,outtype,maxf=2.0):
    #intype=pyr[0].dtype
    maxout=numpy.iinfo(outtype).max
    npyr=[]
    for l in pyr:
        npyr.append(numpy.round((l/maxf)*maxout).astype(outtype))
    return npyr

class pyrHOG:
    def __init__(self,im,interv=10,sbin=8,savedir="./",compress=False,notload=False,notsave=False,hallucinate=0,cformat=False,flip=False,mybuffer=None):
        """
        Compute the HOG pyramid of an image
        if im is a string call precomputed
        if im is an narray call compute
        """
        import time
        t=time.time()       
        self.hog=[]#hog pyramid as a list of hog features
        self.interv=interv
        self.oct=0
        self.sbin=sbin#number of pixels per spatial bin
        self.flip=flip
        if isinstance(im,pyrHOG):#build a copy
            self.__copy(im)
            #return
        if isinstance(im,str):
            rim=im.split("/")[-1]
            if mybuffer!=None:
                if mybuffer.has_key(rim):
                    print "Using Buffer"
                    aux=mybuffer[rim]
                    self.hog=HOGi2f(aux.hog,numpy.float32,maxf=2.0)
                    self.interv=aux.interv
                    self.oct=aux.oct
                    self.sbin=aux.sbin
                    #self.hog=aux.hog
                    self.scale=aux.scale
                    self.hallucinate=aux.hallucinate
                else:
                    print "Computing Buffer"
                    self._precompute(im,interv,sbin,savedir,compress,notload,notsave,hallucinate,cformat=cformat)
                    hogf=self.hog
                    hogi=HOGf2i(hogf,numpy.uint16,maxf=2.0)
                    self.hog=hogi
                    mybuffer[rim]=self
                    self.hog=hogf
            else:
                self._precompute(im,interv,sbin,savedir,compress,notload,notsave,hallucinate,cformat=cformat)
            #return
        if type(im)==numpy.ndarray:
            self._compute(im,interv,sbin,hallucinate,cformat=cformat)
            #return
        print "Features: %.3f s"%(time.time()-t)
        #raise "Error: im must be either a string or an image"
        
    def _compute(self,img,interv=10,sbin=8,hallucinate=0,cformat=False):
        """
        Compute the HOG pyramid of an image
        """
        l=[]
        scl=[]
        octimg=img.astype(numpy.float)#copy()
        maxoct=int(numpy.log2(int(numpy.min(img.shape[:-1])/sbin)))-1#-2
        intimg=octimg
        if hallucinate>1:
            #hallucinate features
            for i in range(interv):
                if cformat:
                    l.append(numpy.ascontiguousarray(hog(intimg,sbin/4),numpy.float32))
                else:
                    l.append(hog(intimg,sbin/4).astype(numpy.float32))                    
                intimg=resize.resize(octimg,math.pow(2,-float(i+1)/interv))
                scl.append(4.0*2.0**(-float(i)/interv))
        if hallucinate>0:
            #hallucinate features
            for i in range(interv):
                if cformat:
                    l.append(numpy.ascontiguousarray(hog(intimg,sbin/2),numpy.float32))                    
                else:
                    l.append(hog(intimg,sbin/2).astype(numpy.float32))
                intimg=resize.resize(octimg,math.pow(2,-float(i+1)/interv))
                scl.append(2.0*2.0**(-float(i)/interv))
        #normal features
        for o in range(maxoct):
            intimg=octimg
            for i in range(interv):
                t1=time.time()
                if cformat:
                    l.append(numpy.ascontiguousarray(hog(intimg,sbin),numpy.float32))                    
                else:
                    l.append(hog(intimg,sbin).astype(numpy.float32))
                scl.append(2.0**(-o-float(i)/interv))
                t2=time.time()
                intimg=resize.resize(octimg,math.pow(2,-float(i+1)/interv))
            octimg=intimg
        self.hog=l
        self.interv=interv
        self.oct=maxoct
        self.sbin=sbin
        self.scale=scl
        self.hallucinate=hallucinate
        
    def _precompute(self,imname,interv=10,sbin=8,savedir="./",compress=False,notload=False,notsave=True,hallucinate=0,cformat=False):
        """
        Check if the HOG if imname is already computed, otherwise 
        compute it and save in savedir
        """
        try:
            if notload:
                #generate an error to pass to computing hog
                error
            else:
                "Warning: image flip do not work with preload!!!"
            f=[]
            if compress:
                f=gzip.open(savedir+imname.split("/")[-1]+".zhog%d_%d_%d"%(interv,sbin,hallucinate),"rb")
            else:
                f=open(savedir+imname.split("/")[-1]+".hog%d_%d_%d"%(interv,sbin,hallucinate),"r")
            print "Loading precalculated Hog"
            aux=cPickle.load(f)
            self.interv=aux.interv
            self.oct=aux.oct
            self.sbin=aux.sbin
            self.hog=aux.hog
            self.scale=aux.scale
            self.hallucinate=aux.hallucinate
        except:
            print "Computing Hog"
            img=None
            img=util.myimread(imname,self.flip)
            if imname.split("_")[-1]=="_flip":  
                print "Flipping Image!"  
                img=img[:,::-1].copy()
            if img.ndim<3:
                aux=numpy.zeros((img.shape[0],img.shape[1],3))
                aux[:,:,0]=img
                aux[:,:,1]=img
                aux[:,:,2]=img
                img=aux
            self._compute(img,interv=interv,sbin=sbin,hallucinate=hallucinate,cformat=cformat)
            if notsave:
                return
            f=[]
            if compress:
                f=gzip.open(savedir+imname.split("/")[-1]+".zhog%d_%d_%d"%(self.interv,self.sbin,hallucinate),"wb")
            else:
                f=open(savedir+imname.split("/")[-1]+".hog%d_%d_%d"%(self.interv,self.sbin,hallucinate),"w")
            cPickle.dump(self,f,2)   

    def resetHOG(self):
        "reset the HOG computation counter"
        ff.resetHOG()

    def getHOG(self):
        "get the HOG computation counter"
        return ff.getHOG()

    def scanRCFL_old(self,model,initr=1,ratio=1,small=True,trunc=0):
        """
        scan the HOG pyramid using the CtF algorithm
        """        
        ww=model["ww"]
        rho=model["rho"]
        if model.has_key("occl"):
            print "Occlusions:",model["occl"]
            occl=numpy.array(model["occl"])*SMALL
        else:
            #print "No Occlusions"
            occl=numpy.zeros(len(model["ww"]))
        res=[]#score
        pparts=[]#parts position
        tot=0
        if not(small):
            self.starti=self.interv*(len(ww)-1)
        else:
            if type(small)==bool:
                self.starti=0
            else:
                self.starti=self.interv*(len(ww)-1-small)
        from time import time
        for i in range(self.starti,len(self.hog)):
            samples=numpy.mgrid[-ww[0].shape[0]+initr:self.hog[i].shape[0]+1:1+2*initr,-ww[0].shape[1]+initr:self.hog[i].shape[1]+1:1+2*initr].astype(ctypes.c_int)
            sshape=samples.shape[1:3]
            res.append(numpy.zeros(sshape,dtype=ctypes.c_float))
            pparts.append(numpy.zeros((2,len(ww),sshape[0],sshape[1]),dtype=c_int))
            for lev in range(len(ww)):
                if i-self.interv*lev>=0:
                    if lev==0:
                        r=initr
                    else:
                        r=ratio
                    auxres=res[-1].copy()
                    ff.scaneigh(self.hog[i-self.interv*lev],
                        self.hog[i-self.interv*lev].shape[0],
                        self.hog[i-self.interv*lev].shape[1],
                        ww[lev],
                        ww[lev].shape[0],ww[lev].shape[1],ww[lev].shape[2],
                        samples[0,:,:],
                        samples[1,:,:],
                        auxres,
                        pparts[-1][0,lev,:,:],
                        pparts[-1][1,lev,:,:],
                        r,r,
                        sshape[0]*sshape[1],trunc)
                    res[i-self.starti]+=auxres
                    samples[:,:,:]=(samples[:,:,:]+pparts[-1][:,lev,:,:])*2+1
                else:#resolution occlusion
                    if len(model["ww"])-1>lev:
                        res[i-self.starti]+=occl[lev-1]
            res[i-self.starti]-=rho
        return res,pparts


    def scanRCFL(self,model,initr=1,ratio=1,small=True,trunc=0,partScr=False):
        """
        scan the HOG pyramid using the CtF algorithm
        """        
        ww=model["ww"]
        rho=model["rho"]
        if model.has_key("occl"):
            print "Occlusions:",model["occl"]
            occl=numpy.array(model["occl"])*SMALL
        else:
            #print "No Occlusions"
            occl=numpy.zeros(len(model["ww"]))
        res=[]#score
        res2=[]#partial score
        pparts=[]#parts position
        tot=0
        if not(small):
            self.starti=self.interv*(len(ww)-1)
        else:
            if type(small)==bool:
                self.starti=0
            else:
                self.starti=self.interv*(len(ww)-1-small)
        from time import time
        for i in range(self.starti,len(self.hog)):
            samples=numpy.mgrid[-ww[0].shape[0]+initr:self.hog[i].shape[0]+1:1+2*initr,-ww[0].shape[1]+initr:self.hog[i].shape[1]+1:1+2*initr].astype(ctypes.c_int)
            sshape=samples.shape[1:3]
            res.append(numpy.zeros(sshape,dtype=ctypes.c_float))
            if partScr:
                res2.append(numpy.zeros((sshape[0],sshape[1],len(model["ww"])),dtype=ctypes.c_float))
            pparts.append(numpy.zeros((2,len(ww),sshape[0],sshape[1]),dtype=c_int))
            for lev in range(len(ww)):
                if i-self.interv*lev>=0:
                    if lev==0:
                        r=initr
                    else:
                        r=ratio
                    auxres=res[-1].copy()
                    ff.scaneigh(self.hog[i-self.interv*lev],
                        self.hog[i-self.interv*lev].shape[0],
                        self.hog[i-self.interv*lev].shape[1],
                        ww[lev],
                        ww[lev].shape[0],ww[lev].shape[1],ww[lev].shape[2],
                        samples[0,:,:],
                        samples[1,:,:],
                        auxres,
                        pparts[-1][0,lev,:,:],
                        pparts[-1][1,lev,:,:],
                        r,r,
                        sshape[0]*sshape[1],trunc)
                    res[i-self.starti]+=auxres
                    if partScr:
                        res2[i-self.starti][:,:,lev]=auxres
                    samples[:,:,:]=(samples[:,:,:]+pparts[-1][:,lev,:,:])*2+1
                else:#resolution occlusion
                    if len(model["ww"])-1>lev:
                        res[i-self.starti]+=occl[lev-1]
            res[i-self.starti]-=rho
        if partScr:
            return res,res2,pparts
        return res,pparts


    def scanRCFLbow(self,model,initr=1,ratio=1,small=True,trunc=0):
        """
        scan the HOG pyramid using the CtF algorithm
        """ 
        #print "Hist",len(model["hist"][0])       
        ww=model["ww"]
        rho=model["rho"]
        siftsize=2#int(numpy.sqrt(model["hist"][0].shape[0]/9))
        bin=6
        numvoc=bin**(siftsize**2)#model["voc"][0].shape[2]       
        if model.has_key("occl"):
            print "Occlusions:",model["occl"]
            occl=numpy.array(model["occl"])*SMALL
        else:
            #print "No Occlusions"
            occl=numpy.zeros(len(model["ww"]))
        res=[]#score
        pparts=[]#parts position
        tot=0
        #print "Using BOW"
        if not(small):
            self.starti=self.interv*(len(ww)-1)
        else:
            if type(small)==bool:
                self.starti=0
            else:
                self.starti=self.interv*(len(ww)-1-small)
        from time import time
        for i in range(self.starti,len(self.hog)):
            samples=numpy.mgrid[-ww[0].shape[0]+initr:self.hog[i].shape[0]+1:1+2*initr,-ww[0].shape[1]+initr:self.hog[i].shape[1]+1:1+2*initr].astype(ctypes.c_int)
            sshape=samples.shape[1:3]
            res.append(numpy.zeros(sshape,dtype=ctypes.c_float))
            pparts.append(numpy.zeros((2,len(ww),sshape[0],sshape[1]),dtype=c_int))
            for lev in range(len(ww)):
                if i-self.interv*lev>=0:
                    if lev==0:
                        r=initr
                    else:
                        r=ratio
                    auxres=res[-1].copy()
                    ff.scaneighbow(self.hog[i-self.interv*lev],
                        self.hog[i-self.interv*lev].shape[0],
                        self.hog[i-self.interv*lev].shape[1],
                        ww[lev],
                        ww[lev].shape[0],ww[lev].shape[1],ww[lev].shape[2],
                        samples[0,:,:],
                        samples[1,:,:],
                        auxres,
                        pparts[-1][0,lev,:,:],
                        pparts[-1][1,lev,:,:],
                        r,r,
                        sshape[0]*sshape[1],trunc,
                        siftsize,numvoc,model["hist"][lev],model["hist"][lev])
                    #print "Check",numpy.any(model["hist"][lev]>1000.0),"size",len(model["hist"][lev])
                    #raw_input()
                    res[i-self.starti]+=auxres
                    samples[:,:,:]=(samples[:,:,:]+pparts[-1][:,lev,:,:])*2+1
                else:#resolution occlusion
                    if len(model["ww"])-1>lev:
                        res[i-self.starti]+=occl[lev-1]
            res[i-self.starti]-=rho
        return res,pparts


    def scanRCFLDef_old(self,model,initr=1,ratio=1,small=True,usemrf=True,mysamples=None,trunc=0):
        """
        scan the HOG pyramid using the CtF algorithm but using deformations
        """     
        ww=model["ww"]
        rho=model["rho"]
        df=model["df"]
        if model.has_key("occl"):
            print "Occlusions:",model["occl"]
            occl=numpy.array(model["occl"])*SMALL
        else:
            #print "No Occlusions"
            occl=numpy.zeros(len(model["ww"]))
        res=[]#score
        pparts=[]#parts position
        tot=0
        if not(small):
            self.starti=self.interv*(len(ww)-1)
        else:
            if type(small)==bool:
                self.starti=0
            else:
                self.starti=self.interv*(len(ww)-1-small)
        from time import time
        for i in range(self.starti,len(self.hog)):
            if mysamples==None:
                samples=numpy.mgrid[-ww[0].shape[0]+initr:self.hog[i].shape[0]+1:1+2*initr,-ww[0].shape[1]+initr:self.hog[i].shape[1]+1:1+2*initr].astype(c_int)
            else:
                samples=mysamples[i]
            sshape=samples.shape[1:3]
            res.append(numpy.zeros(sshape,dtype=ctypes.c_float))
            pparts.append([])
            nelem=(sshape[0]*sshape[1])
            for l in range(len(ww)):
                pparts[-1].append(numpy.zeros((2**l,2**l,4,sshape[0],sshape[1]),dtype=c_int))
            ff.scaneigh(self.hog[i],
                self.hog[i].shape[0],
                self.hog[i].shape[1],
                ww[0],
                ww[0].shape[0],ww[0].shape[1],ww[0].shape[2],
                samples[0,:,:],
                samples[1,:,:],
                res[-1],
                pparts[-1][0][0,0,0,:,:],
                pparts[-1][0][0,0,1,:,:],
                initr,initr,
                nelem,trunc)
            samples[:,:,:]=(samples[:,:,:]+pparts[-1][0][0,0,:2,:,:])*2+1
            if i-self.interv>=0 and len(model["ww"])-1>0:
                self.scanRCFLPart(model,samples,pparts[-1],res[i-self.starti],i-self.interv,1,0,0,ratio,usemrf,occl,trunc) 
            else:
                res[i-self.starti]+=numpy.sum(occl[1:])
            res[i-self.starti]-=rho
        return res,pparts

    def scanRCFLDef(self,model,initr=1,ratio=1,small=True,usemrf=True,mysamples=None,trunc=0):
        """
        scan the HOG pyramid using the CtF algorithm but using deformations
        """     
        ww=model["ww"]
        rho=model["rho"]
        df=model["df"]
        if model.has_key("occl"):
            print "Occlusions:",model["occl"]
            occl=numpy.array(model["occl"])*SMALL
        else:
            #print "No Occlusions"
            occl=numpy.zeros(len(model["ww"]))
        model2=decompose(model)
        res=[]#score
        pparts=[]#parts position
        tot=0
        if not(small):
            self.starti=self.interv*(len(ww)-1)
        else:
            if type(small)==bool:
                self.starti=0
            else:
                self.starti=self.interv*(len(ww)-1-small)
        from time import time
        for i in range(self.starti,len(self.hog)):
            if mysamples==None:
                samples=numpy.mgrid[-ww[0].shape[0]+initr:self.hog[i].shape[0]+1:1+2*initr,-ww[0].shape[1]+initr:self.hog[i].shape[1]+1:1+2*initr].astype(c_int)
            else:
                samples=mysamples[i]
            sshape=samples.shape[1:3]
            res.append(numpy.zeros(sshape,dtype=ctypes.c_float))
            pparts.append([])
            nelem=(sshape[0]*sshape[1])
            for l in range(len(ww)):
                pparts[-1].append(numpy.zeros((2**l,2**l,4,sshape[0],sshape[1]),dtype=c_int))
            ff.scaneigh(self.hog[i],
                self.hog[i].shape[0],
                self.hog[i].shape[1],
                ww[0],
                ww[0].shape[0],ww[0].shape[1],ww[0].shape[2],
                samples[0,:,:],
                samples[1,:,:],
                res[-1],
                pparts[-1][0][0,0,0,:,:],
                pparts[-1][0][0,0,1,:,:],
                initr,initr,
                nelem,trunc)
            samples[:,:,:]=(samples[:,:,:]+pparts[-1][0][0,0,:2,:,:])*2+1
            if i-self.interv>=0 and len(model["ww"])-1>0:
                self.scanRCFLPart(model2,samples,pparts[-1],res[i-self.starti],i-self.interv,1,0,0,ratio,usemrf,occl,trunc) 
            else:
                res[i-self.starti]+=numpy.sum(occl[1:])
            res[i-self.starti]-=rho
        return res,pparts


    def scanRCFLPart_old(self,model,samples,pparts,res,i,lev,locy,locx,ratio,usemrf,occl,trunc):
        """
        auxiliary function for the recursive search of the parts
        """     
        locy=locy*2
        locx=locx*2
        fy=model["ww"][0].shape[0]
        fx=model["ww"][0].shape[1]
        ww1=model["ww"][lev][(locy+0)*fy:(locy+1)*fy,(locx+0)*fx:(locx+1)*fx,:].copy()
        ww2=model["ww"][lev][(locy+0)*fy:(locy+1)*fy,(locx+1)*fx:(locx+2)*fx,:].copy()
        ww3=model["ww"][lev][(locy+1)*fy:(locy+2)*fy,(locx+0)*fx:(locx+1)*fx,:].copy()
        ww4=model["ww"][lev][(locy+1)*fy:(locy+2)*fy,(locx+1)*fx:(locx+2)*fx,:].copy()
        df1=model["df"][lev][(locy+0):(locy+1),(locx+0):(locx+1),:].copy()
        df2=model["df"][lev][(locy+0):(locy+1),(locx+1):(locx+2),:].copy()
        df3=model["df"][lev][(locy+1):(locy+2),(locx+0):(locx+1),:].copy()
        df4=model["df"][lev][(locy+1):(locy+2),(locx+1):(locx+2),:].copy()
        parts=numpy.zeros((2,2,4,res.shape[0],res.shape[1]),dtype=c_int)
        auxres=numpy.zeros(res.shape,numpy.float32)
        ff.scanDef2(ww1,ww2,ww3,ww4,fy,fx,ww1.shape[2],df1,df2,df3,df4,self.hog[i],self.hog[i].shape[0],self.hog[i].shape[1],samples[0,:,:],samples[1,:,:],parts,auxres,ratio,samples.shape[1]*samples.shape[2],usemrf,numpy.array([],dtype=numpy.float32),0,ctypes.POINTER(c_float)(),0,0,trunc)
        res+=auxres
        pparts[lev][(locy+0):(locy+2),(locx+0):(locx+2),:,:,:]=parts
        if i-self.interv>=0 and len(model["ww"])-1>lev:
            samples1=(samples+parts[0,0,:2,:,:])*2+1
            self.scanRCFLPart(model,samples1.copy(),pparts,res,i-self.interv,lev+1,(locy+0),(locx+0),ratio,usemrf,occl,trunc)
            samples2=((samples.T+parts[0,1,:2,:,:].T+numpy.array([0,fx],dtype=c_int).T)*2+1).T
            self.scanRCFLPart(model,samples2.copy(),pparts,res,i-self.interv,lev+1,(locy+0),(locx+1),ratio,usemrf,occl,trunc)
            samples3=((samples.T+parts[1,0,:2,:,:].T+numpy.array([fy,0],dtype=c_int).T)*2+1).T
            self.scanRCFLPart(model,samples3.copy(),pparts,res,i-self.interv,lev+1,(locy+1),(locx+0),ratio,usemrf,occl,trunc)
            samples4=((samples.T+parts[1,1,:2,:,:].T+numpy.array([fy,fx],dtype=c_int).T)*2+1).T
            self.scanRCFLPart(model,samples4.copy(),pparts,res,i-self.interv,lev+1,(locy+1),(locx+1),ratio,usemrf,occl,trunc)
        else:
            if len(model["ww"])-1>lev:
                res+=numpy.sum(occl[lev+1:])

    def scanRCFLPart(self,model2,samples,pparts,res,i,lev,locy,locx,ratio,usemrf,occl,trunc):
        """
        auxiliary function for the recursive search of the parts
        """     
        locy=locy*2
        locx=locx*2
        #model2=decompose(model)
        fy=model2["base"][0]
        fx=model2["base"][1]
        ww1=model2["parts"][0]["ww"];ww2=model2["parts"][1]["ww"]
        ww3=model2["parts"][2]["ww"];ww4=model2["parts"][3]["ww"]
        df1=model2["parts"][0]["df"];df2=model2["parts"][1]["df"]
        df3=model2["parts"][2]["df"];df4=model2["parts"][3]["df"]
#        ww1=model["ww"][lev][(locy+0)*fy:(locy+1)*fy,(locx+0)*fx:(locx+1)*fx,:].copy()
#        ww2=model["ww"][lev][(locy+0)*fy:(locy+1)*fy,(locx+1)*fx:(locx+2)*fx,:].copy()
#        ww3=model["ww"][lev][(locy+1)*fy:(locy+2)*fy,(locx+0)*fx:(locx+1)*fx,:].copy()
#        ww4=model["ww"][lev][(locy+1)*fy:(locy+2)*fy,(locx+1)*fx:(locx+2)*fx,:].copy()
#        df1=model["df"][lev][(locy+0):(locy+1),(locx+0):(locx+1),:].copy()
#        df2=model["df"][lev][(locy+0):(locy+1),(locx+1):(locx+2),:].copy()
#        df3=model["df"][lev][(locy+1):(locy+2),(locx+0):(locx+1),:].copy()
#        df4=model["df"][lev][(locy+1):(locy+2),(locx+1):(locx+2),:].copy()
        parts=numpy.zeros((2,2,4,res.shape[0],res.shape[1]),dtype=c_int)
        auxres=numpy.zeros(res.shape,numpy.float32)
        ff.scanDef2(ww1,ww2,ww3,ww4,fy,fx,ww1.shape[2],df1,df2,df3,df4,self.hog[i],self.hog[i].shape[0],self.hog[i].shape[1],samples[0,:,:],samples[1,:,:],parts,auxres,ratio,samples.shape[1]*samples.shape[2],usemrf,numpy.array([],dtype=numpy.float32),0,ctypes.POINTER(c_float)(),0,0,trunc)
        res+=auxres
        pparts[lev][(locy+0):(locy+2),(locx+0):(locx+2),:,:,:]=parts
        if i-self.interv>=0 and model2["len"]-1>lev:
            samples1=(samples+parts[0,0,:2,:,:])*2+1
            self.scanRCFLPart(model2["parts"][0],samples1.copy(),pparts,res,i-self.interv,lev+1,(locy+0),(locx+0),ratio,usemrf,occl,trunc)
            samples2=((samples.T+parts[0,1,:2,:,:].T+numpy.array([0,fx],dtype=c_int).T)*2+1).T
            self.scanRCFLPart(model2["parts"][1],samples2.copy(),pparts,res,i-self.interv,lev+1,(locy+0),(locx+1),ratio,usemrf,occl,trunc)
            samples3=((samples.T+parts[1,0,:2,:,:].T+numpy.array([fy,0],dtype=c_int).T)*2+1).T
            self.scanRCFLPart(model2["parts"][2],samples3.copy(),pparts,res,i-self.interv,lev+1,(locy+1),(locx+0),ratio,usemrf,occl,trunc)
            samples4=((samples.T+parts[1,1,:2,:,:].T+numpy.array([fy,fx],dtype=c_int).T)*2+1).T
            self.scanRCFLPart(model2["parts"][3],samples4.copy(),pparts,res,i-self.interv,lev+1,(locy+1),(locx+1),ratio,usemrf,occl,trunc)
        else:
            if model2["len"]-1>lev:
                res+=numpy.sum(occl[lev+1:])


    def scanRCFLDefbow(self,model,initr=1,ratio=1,small=True,usemrf=True,mysamples=None,trunc=0):
        """
        scan the HOG pyramid using the CtF algorithm but using deformations
        """     
        siftsize=2
        numvoc=6**(siftsize**2)
        ww=model["ww"]
        rho=model["rho"]
        df=model["df"]
        if model.has_key("occl"):
            print "Occlusions:",model["occl"]
            occl=numpy.array(model["occl"])*SMALL
        else:
            #print "No Occlusions"
            occl=numpy.zeros(len(model["ww"]))
        res=[]#score
        pparts=[]#parts position
        tot=0
        if not(small):
            self.starti=self.interv*(len(ww)-1)
        else:
            if type(small)==bool:
                self.starti=0
            else:
                self.starti=self.interv*(len(ww)-1-small)
        from time import time
        for i in range(self.starti,len(self.hog)):
            if mysamples==None:
                samples=numpy.mgrid[-ww[0].shape[0]+initr:self.hog[i].shape[0]+1:1+2*initr,-ww[0].shape[1]+initr:self.hog[i].shape[1]+1:1+2*initr].astype(c_int)
            else:
                samples=mysamples[i]
            sshape=samples.shape[1:3]
            res.append(numpy.zeros(sshape,dtype=ctypes.c_float))
            pparts.append([])
            nelem=(sshape[0]*sshape[1])
            for l in range(len(ww)):
                pparts[-1].append(numpy.zeros((2**l,2**l,4,sshape[0],sshape[1]),dtype=c_int))
            ff.scaneighbow(self.hog[i],
                self.hog[i].shape[0],
                self.hog[i].shape[1],
                ww[0],
                ww[0].shape[0],ww[0].shape[1],ww[0].shape[2],
                samples[0,:,:],
                samples[1,:,:],
                res[-1],
                pparts[-1][0][0,0,0,:,:],
                pparts[-1][0][0,0,1,:,:],
                initr,initr,
                nelem,trunc,
                siftsize,numvoc,model["hist"][0],model["hist"][0])
            samples[:,:,:]=(samples[:,:,:]+pparts[-1][0][0,0,:2,:,:])*2+1
            if i-self.interv>=0 and len(model["ww"])-1>0:
                self.scanRCFLPartbow(model,samples,pparts[-1],res[i-self.starti],i-self.interv,1,0,0,ratio,usemrf,occl,trunc) 
            else:
                res[i-self.starti]+=numpy.sum(occl[1:])
            res[i-self.starti]-=rho
        return res,pparts


    def scanRCFLPartbow(self,model,samples,pparts,res,i,lev,locy,locx,ratio,usemrf,occl,trunc):
        """
        auxiliary function for the recursive search of the parts
        """     
        siftsize=2
        numvoc=6**(siftsize**2)
        locy=locy*2
        locx=locx*2
        fy=model["ww"][0].shape[0]
        fx=model["ww"][0].shape[1]
        ww1=model["ww"][lev][(locy+0)*fy:(locy+1)*fy,(locx+0)*fx:(locx+1)*fx,:].copy()
        ww2=model["ww"][lev][(locy+0)*fy:(locy+1)*fy,(locx+1)*fx:(locx+2)*fx,:].copy()
        ww3=model["ww"][lev][(locy+1)*fy:(locy+2)*fy,(locx+0)*fx:(locx+1)*fx,:].copy()
        ww4=model["ww"][lev][(locy+1)*fy:(locy+2)*fy,(locx+1)*fx:(locx+2)*fx,:].copy()
        df1=model["df"][lev][(locy+0):(locy+1),(locx+0):(locx+1),:].copy()
        df2=model["df"][lev][(locy+0):(locy+1),(locx+1):(locx+2),:].copy()
        df3=model["df"][lev][(locy+1):(locy+2),(locx+0):(locx+1),:].copy()
        df4=model["df"][lev][(locy+1):(locy+2),(locx+1):(locx+2),:].copy()
        hst=numpy.zeros((4,len(model["hist"][0][0,0])),dtype=numpy.float32)
        hst[0]=model["hist"][lev][(locy+0):(locy+1),(locx+0):(locx+1),:].copy()
        hst[1]=model["hist"][lev][(locy+0):(locy+1),(locx+1):(locx+2),:].copy()
        hst[2]=model["hist"][lev][(locy+1):(locy+2),(locx+0):(locx+1),:].copy()
        hst[3]=model["hist"][lev][(locy+1):(locy+2),(locx+1):(locx+2),:].copy()
        parts=numpy.zeros((2,2,4,res.shape[0],res.shape[1]),dtype=c_int)
        auxres=numpy.zeros(res.shape,numpy.float32)
        ff.scanDefbow(ww1,ww2,ww3,ww4,fy,fx,ww1.shape[2],df1,df2,df3,df4,self.hog[i],self.hog[i].shape[0],self.hog[i].shape[1],samples[0,:,:],samples[1,:,:],parts,auxres,ratio,samples.shape[1]*samples.shape[2],usemrf,numpy.array([],dtype=numpy.float32),0,ctypes.POINTER(c_float)(),0,0,trunc,siftsize,numvoc,hst,hst)
        res+=auxres
        pparts[lev][(locy+0):(locy+2),(locx+0):(locx+2),:,:,:]=parts
        if i-self.interv>=0 and len(model["ww"])-1>lev:
            samples1=(samples+parts[0,0,:2,:,:])*2+1
            self.scanRCFLPartbow(model,samples1.copy(),pparts,res,i-self.interv,lev+1,(locy+0),(locx+0),ratio,usemrf,occl,trunc)
            samples2=((samples.T+parts[0,1,:2,:,:].T+numpy.array([0,fx],dtype=c_int).T)*2+1).T
            self.scanRCFLPartbow(model,samples2.copy(),pparts,res,i-self.interv,lev+1,(locy+0),(locx+1),ratio,usemrf,occl,trunc)
            samples3=((samples.T+parts[1,0,:2,:,:].T+numpy.array([fy,0],dtype=c_int).T)*2+1).T
            self.scanRCFLPartbow(model,samples3.copy(),pparts,res,i-self.interv,lev+1,(locy+1),(locx+0),ratio,usemrf,occl,trunc)
            samples4=((samples.T+parts[1,1,:2,:,:].T+numpy.array([fy,fx],dtype=c_int).T)*2+1).T
            self.scanRCFLPartbow(model,samples4.copy(),pparts,res,i-self.interv,lev+1,(locy+1),(locx+1),ratio,usemrf,occl,trunc)
        else:
            if len(model["ww"])-1>lev:
                res+=numpy.sum(occl[lev+1:])

    def scanRCFLDefThr(self,model,initr=1,ratio=1,small=True,usemrf=True,mythr=0):
        """
        scan the HOG pyramid using the CtF algorithm but using deformations and a pruning threshold
        """    
        ww=model["ww"]
        rho=model["rho"]
        df=model["df"]
        res=[]
        pparts=[]
        tot=0
        #print "MyTHR",mythr
        if not(small):
            self.starti=self.interv*(len(ww)-1)
        else:
            if type(small)==bool:
                self.starti=0
            else:
                self.starti=self.interv*(len(ww)-1-small)
        from time import time
        for i in range(self.starti,len(self.hog)):
            samples=numpy.mgrid[-ww[0].shape[0]+initr:self.hog[i].shape[0]+1:1+2*initr,-ww[0].shape[1]+initr:self.hog[i].shape[1]+1:1+2*initr].astype(c_int)
            sshape=samples.shape[1:3]
            res.append(numpy.zeros(sshape,dtype=ctypes.c_float))
            pparts.append([])
            nelem=(sshape[0]*sshape[1])
            for l in range(len(ww)):
                pparts[-1].append(numpy.zeros((2**l,2**l,4,sshape[0],sshape[1]),dtype=c_int))
            ff.scaneigh(self.hog[i],
                self.hog[i].shape[0],
                self.hog[i].shape[1],
                ww[0],
                ww[0].shape[0],ww[0].shape[1],ww[0].shape[2],
                samples[0,:,:],
                samples[1,:,:],
                res[-1],
                pparts[-1][0][0,0,0,:,:],
                pparts[-1][0][0,0,1,:,:],
                initr,initr,
                nelem,0)
            samples[:,:,:]=(samples[:,:,:]+pparts[-1][0][0,0,:2,:,:])*2+1
            self.scanRCFLPartThr(model,samples,pparts[-1],res[i-self.starti],i-self.interv,1,0,0,ratio,usemrf,mythr) 
            res[i-self.starti]-=rho
        return res,pparts

    def scanRCFLPartThr(self,model,samples,pparts,res,i,lev,locy,locx,ratio,usemrf,mythr):
        """
        auxiliary function for the recursive search of the parts with pruning threshold
        """   
        locy=locy*2
        locx=locx*2
        fy=model["ww"][0].shape[0]
        fx=model["ww"][0].shape[1]
        ww1=model["ww"][lev][(locy+0)*fy:(locy+1)*fy,(locx+0)*fx:(locx+1)*fx,:].copy()
        ww2=model["ww"][lev][(locy+0)*fy:(locy+1)*fy,(locx+1)*fx:(locx+2)*fx,:].copy()
        ww3=model["ww"][lev][(locy+1)*fy:(locy+2)*fy,(locx+0)*fx:(locx+1)*fx,:].copy()
        ww4=model["ww"][lev][(locy+1)*fy:(locy+2)*fy,(locx+1)*fx:(locx+2)*fx,:].copy()
        df1=model["df"][lev][(locy+0):(locy+1),(locx+0):(locx+1),:].copy()
        df2=model["df"][lev][(locy+0):(locy+1),(locx+1):(locx+2),:].copy()
        df3=model["df"][lev][(locy+1):(locy+2),(locx+0):(locx+1),:].copy()
        df4=model["df"][lev][(locy+1):(locy+2),(locx+1):(locx+2),:].copy()
        parts=numpy.zeros((2,2,4,res.shape[0],res.shape[1]),dtype=c_int)
        auxres=numpy.zeros(res.shape,numpy.float32)
        res[res<mythr]=-1000
        samples[:,res==-1000]=-1000
        ff.scanDef2(ww1,ww2,ww3,ww4,fy,fx,ww1.shape[2],df1,df2,df3,df4,self.hog[i],self.hog[i].shape[0],self.hog[i].shape[1],samples[0,:,:],samples[1,:,:],parts,auxres,ratio,samples.shape[1]*samples.shape[2],usemrf,numpy.array([],dtype=numpy.float32),0,ctypes.POINTER(c_float)(),0,0,0)
        res+=auxres
        pparts[lev][(locy+0):(locy+2),(locx+0):(locx+2),:,:,:]=parts
        if i-self.interv>=0 and len(model["ww"])-1>lev:
            samples1=(samples+parts[0,0,:2,:,:])*2+1
            self.scanRCFLPartThr(model,samples1.copy(),pparts,res,i-self.interv,lev+1,(locy+0),(locx+0),ratio,usemrf,mythr)
            samples2=((samples.T+parts[0,1,:2,:,:].T+numpy.array([0,fx],dtype=c_int).T)*2+1).T
            self.scanRCFLPartThr(model,samples2.copy(),pparts,res,i-self.interv,lev+1,(locy+0),(locx+1),ratio,usemrf,mythr)
            samples3=((samples.T+parts[1,0,:2,:,:].T+numpy.array([fy,0],dtype=c_int).T)*2+1).T
            self.scanRCFLPartThr(model,samples3.copy(),pparts,res,i-self.interv,lev+1,(locy+1),(locx+0),ratio,usemrf,mythr)
            samples4=((samples.T+parts[1,1,:2,:,:].T+numpy.array([fy,fx],dtype=c_int).T)*2+1).T
            self.scanRCFLPartThr(model,samples4.copy(),pparts,res,i-self.interv,lev+1,(locy+1),(locx+1),ratio,usemrf,mythr)


    def scanRCFLDefBU_old(self,model,initr=1,ratio=1,small=True,usemrf=True,mysamples=None):
        """
        scan the HOG pyramid using the full search and using deformations
        """   
        ww=model["ww"]
        rho=model["rho"]
        df=model["df"]
        res=[]#score
        prec=[]#precomputed scores
        pres=[]
        pparts=[]#parts position
        tot=0
        #model2=decompose(model)
        pady=model["ww"][-1].shape[0]
        padx=model["ww"][-1].shape[1]
        if not(small):
            self.starti=self.interv*(len(ww)-1)
        else:
            if type(small)==bool:
                self.starti=0
            else:
                self.starti=self.interv*(len(ww)-1-small)
        from time import time
        ttt=0
        for i in range(self.starti,len(self.hog)):
            #print "Level",i,"--------------------------------------------------"
            #print "Time:",time()-ttt
            ttt=time()
            if mysamples==None:
                samples=numpy.mgrid[-ww[0].shape[0]+initr:self.hog[i].shape[0]+1:1+2*initr,-ww[0].shape[1]+initr:self.hog[i].shape[1]+1:1+2*initr].astype(c_int)
            else:
                samples=mysamples[i]
            csamples=samples.copy()
            sshape=samples.shape[1:3]
            pres.append(numpy.zeros(((2*initr+1)*(2*initr+1),sshape[0],sshape[1]),dtype=ctypes.c_float))
            res.append(numpy.zeros(sshape,dtype=ctypes.c_float))
            pparts.append([])
            prec=[]#.append([])
            nelem=(sshape[0]*sshape[1])
            #auxpparts=[]
            for l in range(len(ww)):
                prec.append(-100000*numpy.ones((4**l,2**l*(self.hog[i].shape[0]+2)+pady*2,2**l*(self.hog[i].shape[1]+2)+padx*2),dtype=ctypes.c_float))
                pparts[-1].append(numpy.zeros((2**l,2**l,4,sshape[0],sshape[1]),dtype=c_int))                
            auxpparts=(numpy.zeros(((2*initr+1)*(2*initr+1),2,2,4,sshape[0],sshape[1]),dtype=c_int))
            auxptr=numpy.zeros((2*initr+1)*(2*initr+1),dtype=object)
            ct=container(auxpparts,auxptr)
            for l in range((2*initr+1)**2):
                maux=(numpy.zeros((4,(2*ratio+1)*(2*ratio+1),2,2,4,sshape[0],sshape[1]),dtype=c_int))
                auxptr=numpy.zeros((4,(2*ratio+1)*(2*ratio+1)),dtype=object)
                ct.ptr[l]=container(maux,auxptr)
            for dy in range(-initr,initr+1):
                for dx in range(-initr,initr+1):
                    csamples[0,:,:]=samples[0,:,:]+dy
                    csamples[1,:,:]=samples[1,:,:]+dx
                    ff.scaneigh(self.hog[i],
                        self.hog[i].shape[0],
                        self.hog[i].shape[1],
                        ww[0],
                        ww[0].shape[0],ww[0].shape[1],ww[0].shape[2],
                        csamples[0,:,:],
                        csamples[1,:,:],
                        pres[-1][(dy+initr)*(2*initr+1)+(dx+initr),:,:],
                        pparts[-1][0][0,0,0,:,:],
                        pparts[-1][0][0,0,1,:,:],
                        0,0,
                        nelem,0)
                    csamples=csamples[:,:,:]*2+1
                    self.scanRCFLPartBU(model,csamples,pparts[-1],ct.ptr[(dy+initr)*(2*initr+1)+(dx+initr)],pres[i-self.starti][(dy+initr)*(2*initr+1)+(dx+initr),:,:],i-self.interv,1,0,0,ratio,usemrf,prec,pady,padx) 
                    #self.scanRCFLPartBU2(model2["parts"],csamples,pparts[-1],ct.ptr[(dy+initr)*(2*initr+1)+(dx+initr)],pres[i-self.starti][(dy+initr)*(2*initr+1)+(dx+initr),:,:],i-self.interv,1,0,0,ratio,usemrf,prec,pady,padx) 
            res[i-self.starti]=pres[i-self.starti].max(0)
            el=pres[i-self.starti].argmax(0)
            pparts[-1][0][0,0,0,:,:]=el/(initr*2+1)-1
            pparts[-1][0][0,0,1,:,:]=el%(initr*2+1)-1
            for l in range(1,len(ww)):
                elx=numpy.tile(el,(2**l,2**l,4,1,1))
                for pt in range((2*initr+1)*(2*initr+1)):
                    if len(ct.ptr[pt].best)>=l:
                        pparts[-1][l][elx==pt]=ct.ptr[pt].best[l-1][elx==pt]
            res[i-self.starti]-=rho
        return res,pparts


    def scanRCFLPartBU(self,model,samples,pparts,ct,res,i,lev,locy,locx,ratio,usemrf,prec,pady,padx):
        """
        auxiliary function for the recursive search of the parts for the complete search
        """   
        locy=locy*2
        locx=locx*2
        fy=model["ww"][0].shape[0]
        fx=model["ww"][0].shape[1]
        ww1=model["ww"][lev][(locy+0)*fy:(locy+1)*fy,(locx+0)*fx:(locx+1)*fx,:].copy()
        ww2=model["ww"][lev][(locy+0)*fy:(locy+1)*fy,(locx+1)*fx:(locx+2)*fx,:].copy()
        ww3=model["ww"][lev][(locy+1)*fy:(locy+2)*fy,(locx+0)*fx:(locx+1)*fx,:].copy()
        ww4=model["ww"][lev][(locy+1)*fy:(locy+2)*fy,(locx+1)*fx:(locx+2)*fx,:].copy()
        df1=model["df"][lev][(locy+0):(locy+1),(locx+0):(locx+1),:].copy()
        df2=model["df"][lev][(locy+0):(locy+1),(locx+1):(locx+2),:].copy()
        df3=model["df"][lev][(locy+1):(locy+2),(locx+0):(locx+1),:].copy()
        df4=model["df"][lev][(locy+1):(locy+2),(locx+1):(locx+2),:].copy()
        parts=numpy.zeros((2,2,4,res.shape[0],res.shape[1]),dtype=c_int)
        auxparts=numpy.zeros((4,(2*ratio+1)*(2*ratio+1),2,2,4,res.shape[0],res.shape[1]),dtype=c_int)
        if i-self.interv>=0 and len(model["ww"])-1>lev:
            for l in range(len(ct.ptr)):
                maux=(numpy.zeros((4,(2*ratio+1)*(2*ratio+1),2,2,4,res.shape[0],res.shape[1]),dtype=c_int))
                auxptr=numpy.zeros((4,(2*ratio+1)*(2*ratio+1)),dtype=object)
                ct.ptr[l]=container(maux,auxptr)
        auxres=numpy.zeros(res.shape,numpy.float32)
        pres=numpy.zeros((4,(2*ratio+1),(2*ratio+1),res.shape[0],res.shape[1]),numpy.float32)
        csamples=samples.copy()
        if i-self.interv>=0 and len(model["ww"])-1>lev:
            for dy in range(-ratio,ratio+1):
                for dx in range(-ratio,ratio+1):
                    csamples[0,:,:]=(samples[0,:,:]+dy)
                    csamples[1,:,:]=(samples[1,:,:]+dx)
                    samples1=(csamples)*2+1
                    auxparts[0,(dy+ratio)*(2*ratio+1)+(dx+ratio)]=self.scanRCFLPartBU(model,samples1,pparts,ct.ptr[0,(dy+ratio)*(2*ratio+1)+(dx+ratio)],pres[0,(dy+ratio),(dx+ratio),:,:],i-self.interv,lev+1,(locy+0),(locx+0),ratio,usemrf,prec,pady,padx)
                    samples2=((csamples.T+numpy.array([0,fx],dtype=c_int).T)*2+1).T
                    auxparts[1,(dy+ratio)*(2*ratio+1)+(dx+ratio)]=self.scanRCFLPartBU(model,samples2.copy(),pparts,ct.ptr[1,(dy+ratio)*(2*ratio+1)+(dx+ratio)],pres[1,(dy+ratio),(dx+ratio),:,:],i-self.interv,lev+1,(locy+0),(locx+1),ratio,usemrf,prec,pady,padx)
                    samples3=((csamples.T+numpy.array([fy,0],dtype=c_int).T)*2+1).T
                    auxparts[2,(dy+ratio)*(2*ratio+1)+(dx+ratio)]=self.scanRCFLPartBU(model,samples3.copy(),pparts,ct.ptr[2,(dy+ratio)*(2*ratio+1)+(dx+ratio)],pres[2,(dy+ratio),(dx+ratio),:,:],i-self.interv,lev+1,(locy+1),(locx+0),ratio,usemrf,prec,pady,padx)
                    samples4=((csamples.T+numpy.array([fy,fx],dtype=c_int).T)*2+1).T
                    auxparts[3,(dy+ratio)*(2*ratio+1)+(dx+ratio)]=self.scanRCFLPartBU(model,samples4.copy(),pparts,ct.ptr[3,(dy+ratio)*(2*ratio+1)+(dx+ratio)],pres[3,(dy+ratio),(dx+ratio),:,:],i-self.interv,lev+1,(locy+1),(locx+1),ratio,usemrf,prec,pady,padx)
        
        auxprec=prec[lev][((locy/2)*2+(locx/2))*4:((locy/2)*2+(locx/2)+1)*4]
        tt=time.time()
        ff.scanDef2(ww1,ww2,ww3,ww4,fy,fx,ww1.shape[2],df1,df2,df3,df4,self.hog[i],self.hog[i].shape[0],self.hog[i].shape[1],samples[0,:,:],samples[1,:,:],parts,auxres,ratio,samples.shape[1]*samples.shape[2],usemrf,pres,1,auxprec.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),pady,padx,0)
        #print time.time()-tt
        res+=auxres
        ct.best=[parts]

        if i-self.interv>=0 and len(model["ww"])-1>lev:            
            for l in range(len(ct.ptr[0,0].best)):
                aux=numpy.zeros((ct.ptr[0,0].best[l].shape[0]*2,ct.ptr[0,0].best[l].shape[1]*2,4,res.shape[0],res.shape[1]))
                ct.best.append(aux)
                for py in range(res.shape[0]):
                    for px in range(res.shape[1]):
                        ps=(parts[0,0,0,py,px]+ratio)*(ratio*2+1)+parts[0,0,1,py,px]+ratio
                        ct.best[-1][locy+0:locy+0+2,locx+0:locx+0+2,:,py,px]=auxparts[0,ps][:,:,:,py,px]
                        ps=(parts[0,1,0,py,px]+ratio)*(ratio*2+1)+parts[0,1,1,py,px]+ratio
                        ct.best[-1][locy+0:locy+0+2,locx+2:locx+2+2,:,py,px]=auxparts[1,ps][:,:,:,py,px]
                        ps=(parts[1,0,0,py,px]+ratio)*(ratio*2+1)+parts[1,0,1,py,px]+ratio
                        ct.best[-1][locy+2:locy+2+2,locx+0:locx+0+2,:,py,px]=auxparts[2,ps][:,:,:,py,px]
                        ps=(parts[1,1,0,py,px]+ratio)*(ratio*2+1)+parts[1,1,1,py,px]+ratio
                        ct.best[-1][locy+2:locy+2+2,locx+2:locx+2+2,:,py,px]=auxparts[3,ps][:,:,:,py,px]
        return parts

    def scanRCFLDefBU(self,model,initr=1,ratio=1,small=True,usemrf=True,mysamples=None):
        """
        scan the HOG pyramid using the full search and using deformations
        """   
        ww=model["ww"]
        rho=model["rho"]
        df=model["df"]
        res=[]#score
        prec=[]#precomputed scores
        pres=[]
        pparts=[]#parts position
        tot=0
        model2=decompose(model)
        pady=model["ww"][-1].shape[0]
        padx=model["ww"][-1].shape[1]
        if not(small):
            self.starti=self.interv*(len(ww)-1)
        else:
            if type(small)==bool:
                self.starti=0
            else:
                self.starti=self.interv*(len(ww)-1-small)
        from time import time
        ttt=0
        for i in range(self.starti,len(self.hog)):
            #print "Level",i,"--------------------------------------------------"
            #print "Time:",time()-ttt
            ttt=time()
            if mysamples==None:
                samples=numpy.mgrid[-ww[0].shape[0]+initr:self.hog[i].shape[0]+1:1+2*initr,-ww[0].shape[1]+initr:self.hog[i].shape[1]+1:1+2*initr].astype(c_int)
            else:
                samples=mysamples[i]
            csamples=samples.copy()
            sshape=samples.shape[1:3]
            pres.append(numpy.zeros(((2*initr+1)*(2*initr+1),sshape[0],sshape[1]),dtype=ctypes.c_float))
            res.append(numpy.zeros(sshape,dtype=ctypes.c_float))
            pparts.append([])
            prec=[]#.append([])
            nelem=(sshape[0]*sshape[1])
            #auxpparts=[]
            for l in range(len(ww)):
                prec.append(-100000*numpy.ones((4**l,2**l*(self.hog[i].shape[0]+2)+pady*2,2**l*(self.hog[i].shape[1]+2)+padx*2),dtype=ctypes.c_float))
                pparts[-1].append(numpy.zeros((2**l,2**l,4,sshape[0],sshape[1]),dtype=c_int))                
            auxpparts=(numpy.zeros(((2*initr+1)*(2*initr+1),2,2,4,sshape[0],sshape[1]),dtype=c_int))
            auxptr=numpy.zeros((2*initr+1)*(2*initr+1),dtype=object)
            ct=container(auxpparts,auxptr)
            for l in range((2*initr+1)**2):
                maux=(numpy.zeros((4,(2*ratio+1)*(2*ratio+1),2,2,4,sshape[0],sshape[1]),dtype=c_int))
                auxptr=numpy.zeros((4,(2*ratio+1)*(2*ratio+1)),dtype=object)
                ct.ptr[l]=container(maux,auxptr)
            for dy in range(-initr,initr+1):
                for dx in range(-initr,initr+1):
                    csamples[0,:,:]=samples[0,:,:]+dy
                    csamples[1,:,:]=samples[1,:,:]+dx
                    ff.scaneigh(self.hog[i],
                        self.hog[i].shape[0],
                        self.hog[i].shape[1],
                        ww[0],
                        ww[0].shape[0],ww[0].shape[1],ww[0].shape[2],
                        csamples[0,:,:],
                        csamples[1,:,:],
                        pres[-1][(dy+initr)*(2*initr+1)+(dx+initr),:,:],
                        pparts[-1][0][0,0,0,:,:],
                        pparts[-1][0][0,0,1,:,:],
                        0,0,
                        nelem,0)
                    csamples=csamples[:,:,:]*2+1
                    self.scanRCFLPartBU2(model2,csamples,pparts[-1],ct.ptr[(dy+initr)*(2*initr+1)+(dx+initr)],pres[i-self.starti][(dy+initr)*(2*initr+1)+(dx+initr),:,:],i-self.interv,1,0,0,ratio,usemrf,prec,pady,padx) 
                    #self.scanRCFLPartBU2(model2["parts"],csamples,pparts[-1],ct.ptr[(dy+initr)*(2*initr+1)+(dx+initr)],pres[i-self.starti][(dy+initr)*(2*initr+1)+(dx+initr),:,:],i-self.interv,1,0,0,ratio,usemrf,prec,pady,padx) 
            res[i-self.starti]=pres[i-self.starti].max(0)
            el=pres[i-self.starti].argmax(0)
            pparts[-1][0][0,0,0,:,:]=el/(initr*2+1)-1
            pparts[-1][0][0,0,1,:,:]=el%(initr*2+1)-1
            for l in range(1,len(ww)):
                elx=numpy.tile(el,(2**l,2**l,4,1,1))
                for pt in range((2*initr+1)*(2*initr+1)):
                    if len(ct.ptr[pt].best)>=l:
                        pparts[-1][l][elx==pt]=ct.ptr[pt].best[l-1][elx==pt]
            res[i-self.starti]-=rho
        return res,pparts


    def scanRCFLPartBU2(self,model2,samples,pparts,ct,res,i,lev,locy,locx,ratio,usemrf,prec,pady,padx):
        """
        auxiliary function for the recursive search of the parts for the complete search
        """   
        locy=locy*2
        locx=locx*2
        fy=model2["base"][0]#model["ww"][0].shape[0]
        fx=model2["base"][1]#model["ww"][0].shape[1]
        ww1=model2["parts"][0]["ww"];ww2=model2["parts"][1]["ww"]
        ww3=model2["parts"][2]["ww"];ww4=model2["parts"][3]["ww"]
        df1=model2["parts"][0]["df"];df2=model2["parts"][1]["df"]
        df3=model2["parts"][2]["df"];df4=model2["parts"][3]["df"]
#        ww1=model["ww"][lev][(locy+0)*fy:(locy+1)*fy,(locx+0)*fx:(locx+1)*fx,:].copy()
#        ww2=model["ww"][lev][(locy+0)*fy:(locy+1)*fy,(locx+1)*fx:(locx+2)*fx,:].copy()
#        ww3=model["ww"][lev][(locy+1)*fy:(locy+2)*fy,(locx+0)*fx:(locx+1)*fx,:].copy()
#        ww4=model["ww"][lev][(locy+1)*fy:(locy+2)*fy,(locx+1)*fx:(locx+2)*fx,:].copy()
#        df1=model["df"][lev][(locy+0):(locy+1),(locx+0):(locx+1),:].copy()
#        df2=model["df"][lev][(locy+0):(locy+1),(locx+1):(locx+2),:].copy()
#        df3=model["df"][lev][(locy+1):(locy+2),(locx+0):(locx+1),:].copy()
#        df4=model["df"][lev][(locy+1):(locy+2),(locx+1):(locx+2),:].copy()
        parts=numpy.zeros((2,2,4,res.shape[0],res.shape[1]),dtype=c_int)
        auxparts=numpy.zeros((4,(2*ratio+1)*(2*ratio+1),2,2,4,res.shape[0],res.shape[1]),dtype=c_int)
        if i-self.interv>=0 and model2["len"]-1>lev:
            for l in range(len(ct.ptr)):
                maux=(numpy.zeros((4,(2*ratio+1)*(2*ratio+1),2,2,4,res.shape[0],res.shape[1]),dtype=c_int))
                auxptr=numpy.zeros((4,(2*ratio+1)*(2*ratio+1)),dtype=object)
                ct.ptr[l]=container(maux,auxptr)
        auxres=numpy.zeros(res.shape,numpy.float32)
        pres=numpy.zeros((4,(2*ratio+1),(2*ratio+1),res.shape[0],res.shape[1]),numpy.float32)
        csamples=samples.copy()
        if i-self.interv>=0 and model2["len"]-1>lev:
            for dy in range(-ratio,ratio+1):
                for dx in range(-ratio,ratio+1):
                    csamples[0,:,:]=(samples[0,:,:]+dy)
                    csamples[1,:,:]=(samples[1,:,:]+dx)
                    samples1=(csamples)*2+1
                    auxparts[0,(dy+ratio)*(2*ratio+1)+(dx+ratio)]=self.scanRCFLPartBU2(model2["parts"][0],samples1,pparts,ct.ptr[0,(dy+ratio)*(2*ratio+1)+(dx+ratio)],pres[0,(dy+ratio),(dx+ratio),:,:],i-self.interv,lev+1,(locy+0),(locx+0),ratio,usemrf,prec,pady,padx)
                    samples2=((csamples.T+numpy.array([0,fx],dtype=c_int).T)*2+1).T
                    auxparts[1,(dy+ratio)*(2*ratio+1)+(dx+ratio)]=self.scanRCFLPartBU2(model2["parts"][1],samples2.copy(),pparts,ct.ptr[1,(dy+ratio)*(2*ratio+1)+(dx+ratio)],pres[1,(dy+ratio),(dx+ratio),:,:],i-self.interv,lev+1,(locy+0),(locx+1),ratio,usemrf,prec,pady,padx)
                    samples3=((csamples.T+numpy.array([fy,0],dtype=c_int).T)*2+1).T
                    auxparts[2,(dy+ratio)*(2*ratio+1)+(dx+ratio)]=self.scanRCFLPartBU2(model2["parts"][2],samples3.copy(),pparts,ct.ptr[2,(dy+ratio)*(2*ratio+1)+(dx+ratio)],pres[2,(dy+ratio),(dx+ratio),:,:],i-self.interv,lev+1,(locy+1),(locx+0),ratio,usemrf,prec,pady,padx)
                    samples4=((csamples.T+numpy.array([fy,fx],dtype=c_int).T)*2+1).T
                    auxparts[3,(dy+ratio)*(2*ratio+1)+(dx+ratio)]=self.scanRCFLPartBU2(model2["parts"][3],samples4.copy(),pparts,ct.ptr[3,(dy+ratio)*(2*ratio+1)+(dx+ratio)],pres[3,(dy+ratio),(dx+ratio),:,:],i-self.interv,lev+1,(locy+1),(locx+1),ratio,usemrf,prec,pady,padx)
        
        auxprec=prec[lev][((locy/2)*2+(locx/2))*4:((locy/2)*2+(locx/2)+1)*4]
        tt=time.time()
        ff.scanDef2(ww1,ww2,ww3,ww4,fy,fx,ww1.shape[2],df1,df2,df3,df4,self.hog[i],self.hog[i].shape[0],self.hog[i].shape[1],samples[0,:,:],samples[1,:,:],parts,auxres,ratio,samples.shape[1]*samples.shape[2],usemrf,pres,1,auxprec.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),pady,padx,0)
        #print time.time()-tt
        res+=auxres
        ct.best=[parts]

        if i-self.interv>=0 and model2["len"]-1>lev:            
            ps=(parts[:,:,0]+ratio)*(ratio*2+1)+parts[:,:,1]+ratio
            for l in range(len(ct.ptr[0,0].best)):
                aux=numpy.zeros((ct.ptr[0,0].best[l].shape[0]*2,ct.ptr[0,0].best[l].shape[1]*2,4,res.shape[0],res.shape[1]))
                ct.best.append(aux)
                for py in range(res.shape[0]):
                    for px in range(res.shape[1]):
                        #ps=(parts[0,0,0,py,px]+ratio)*(ratio*2+1)+parts[0,0,1,py,px]+ratio
                        ct.best[-1][locy+0:locy+0+2,locx+0:locx+0+2,:,py,px]=auxparts[0,ps[0,0,py,px]][:,:,:,py,px]
                        #ps=(parts[0,1,0,py,px]+ratio)*(ratio*2+1)+parts[0,1,1,py,px]+ratio
                        ct.best[-1][locy+0:locy+0+2,locx+2:locx+2+2,:,py,px]=auxparts[1,ps[0,1,py,px]][:,:,:,py,px]
                        #ps=(parts[1,0,0,py,px]+ratio)*(ratio*2+1)+parts[1,0,1,py,px]+ratio
                        ct.best[-1][locy+2:locy+2+2,locx+0:locx+0+2,:,py,px]=auxparts[2,ps[1,0,py,px]][:,:,:,py,px]
                        #ps=(parts[1,1,0,py,px]+ratio)*(ratio*2+1)+parts[1,1,1,py,px]+ratio
                        ct.best[-1][locy+2:locy+2+2,locx+2:locx+2+2,:,py,px]=auxparts[3,ps[1,1,py,px]][:,:,:,py,px]
        return parts


class Treat:
    def __init__(self,f,scr,pos,sample,fy,fx,occl=False,trunc=0,small2=False):
        self.pos=pos
        self.scr=scr
        self.f=f
        self.interv=f.interv
        self.sbin=f.sbin
        self.fy=fy
        self.fx=fx
        self.scale=f.scale
        self.sample=sample
        self.occl=occl
        self.trunc=trunc
        self.small2=small2

    def showBBox(self,allgtbbox,colors=["w","g"],new_alpha=0.15):
        for item in allgtbbox:
            bbox=item["bbox"]
            pylab.fill([bbox[1],bbox[1],bbox[3],bbox[3],bbox[1]],[bbox[0],bbox[2],bbox[2],bbox[0],bbox[0]],colors[0], alpha=new_alpha, edgecolor=colors[0],lw=2)

    def doall(self,thr=-1,rank=1000,refine=True,cluster=0.5,rawdet=False,show=False,inclusion=False,cl=0):
        """
        executes all the detection steps for test
        """        
        #import time
        #t=time.time()
        self.det=self.select_fast(thr,cl=cl)        
        #print len(self.det)
        #print "Select",time.time()-t
        #t=time.time()
        self.det=self.rank_fast(self.det,rank,cl=cl) 
        #print "Rank",time.time()-t
        if refine:
        #    t=time.time()
            self.det=self.refine(self.det)
        #    print "Refine",time.time()-t
        #t=time.time()
        #if occl:
        #    self.det=self.occl(self.det)
        self.det=self.bbox(self.det)
        #print "Bbox",time.time()-t
        #t=time.time()
        if cluster>0:
        #    t=time.time()   
            self.det=self.cluster(self.det,ovr=cluster,maxcl=50,inclusion=inclusion)
        #    print "Cluster",time.time()-t
        if rawdet:
        #    t=time.time()
            self.det=self.rawdet(self.det)
        #    print "Rawdet",time.time()-t
        if show:
        #    t=time.time()
            self.show(self.det)
        #    print "Show",time.time()-t
        if show=="Parts":
        #    t=time.time()
            self.show(self.det,parts=True)
        #    print "Show Parts",time.time()-t
        return self.det
   
    def doalltrain(self,gtbbox,thr=-numpy.inf,rank=numpy.inf,refine=True,rawdet=True,show=False,mpos=0,posovr=0.5,numpos=1,numneg=10,minnegovr=0,minnegincl=0,cl=0,emptybb=False,useMaxOvr=False):
        """
        executes all the detection steps for training
        """        
        t=time.time()
        #print "THR:",thr
        self.det=self.select_fast(thr,cl=cl)
        #print "Select time:",time.time()-t
        t=time.time()
        self.det=self.rank_fast(self.det,rank,cl=cl) 
        #print "Rank time:",time.time()-t
        if refine:
            t=time.time()
            self.det=self.refine(self.det)
        #    print "Refine time:",time.time()-t
        t=time.time()
        self.det=self.bbox(self.det)
        #print "BBox time:",time.time()-t
        t=time.time()
        self.best,self.worste=self.getbestworste(self.det,gtbbox,numpos=numpos,numneg=numneg,mpos=mpos,posovr=posovr,minnegovr=minnegovr,minnegincl=minnegincl,emptybb=emptybb,useMaxOvr=useMaxOvr)
        #print "Bestworse time:",time.time()-t
        if rawdet:
            t=time.time()
            self.best=self.rawdet(self.best)
            self.worste=self.rawdet(self.worste)
        #    print "Raw Det time:",time.time()-t
        if show:
            self.show(self.best,colors=["b"])
            self.show(self.worste,colors=["r"])
            self.showBBox(gtbbox)
        if show=="Parts":
            self.show(self.best,parts=True)
            self.show(self.worste,parts=True)
        return self.best,self.worste

    def select(self,thr=0,cl=0,dense=DENSE):
        """
        select the best detections
        """
        det=[]
        initr=self.sample
        for i in range(len(self.scr)):
            if len(self.scr)-i<dense:
                initr=0
            cy,cx=numpy.where(self.scr[i]>thr)
            for l in range(len(cy)):
                mcy=(cy[l])*(2*initr+1)-self.fy+1+initr
                mcx=(cx[l])*(2*initr+1)-self.fx+1+initr
                det.append({"i":i,"py":cy[l],"px":cx[l],"scr":self.scr[i][cy[l],cx[l]],"ry":mcy,"rx":mcx,"scl":self.scale[i+self.f.starti],"fy":self.fy,"fx":self.fx,"cl":cl})
        return det

    def select_fast(self,thr=0,cl=0,dense=DENSE):
        """
        select the best detections in a faster way
        """
        det=[];mcy=[];mcx=[];ii=[];ir=[]
        for i in range(len(self.scr)):
            if len(self.scr)-i<dense:
                initr=0
            else:
                initr=self.sample
            cy,cx=numpy.where(self.scr[i]>thr)
            mcy+=(cy).tolist()
            mcx+=(cx).tolist()
            ii+=(i*numpy.ones(len(cy),dtype=numpy.int)).tolist()
            det+=self.scr[i][cy,cx].tolist()
            ir+=((initr*numpy.ones(len(cy)) ).tolist())
        return det,mcy,mcx,ii,ir

    def compare(self,a, b):
        return cmp(b["scr"], a["scr"]) # compare as integers

    def rank(self,det,maxnum=1000):
        """
           rank detections based on score
        """
        rdet=det[:]
        rdet.sort(self.compare)
        if maxnum==numpy.inf:
            maxnum=len(rdet)
        return rdet[:maxnum]

    def rank_fast(self,detx,maxnum=1000,cl=0,dense=DENSE):
        """
           rank detections based on score fast
        """
        rdet=[]
        det=detx[0]
        cy=detx[1]
        cx=detx[2]
        i=detx[3]
        initr=numpy.array(detx[4])
        pos=numpy.argsort(-numpy.array(det))      
        if maxnum==numpy.inf:
            maxnum=len(rdet)
        mcy=numpy.array(cy)*(2*initr+1)-self.fy+1+initr
        mcx=numpy.array(cx)*(2*initr+1)-self.fx+1+initr
        for l in pos[:maxnum]:
            rdet.append({"i":i[l],"py":cy[l],"px":cx[l],"scr":det[l],"ry":mcy[l],"rx":mcx[l],"scl":self.scale[i[l]+self.f.starti],"fy":self.fy,"fx":self.fx,"cl":cl})
        return rdet

    def refine(self,ldet):
        """
            refine the localization of the object based on higher resolutions
        """
        rdet=[]
        for item in ldet:
            i=item["i"];cy=item["py"];cx=item["px"];
            el=item.copy()
            el["ny"]=el["ry"]
            el["nx"]=el["rx"]
            mov=numpy.zeros(2)
            el["def"]={"dy":numpy.zeros(self.pos[i].shape[1]),"dx":numpy.zeros(self.pos[i].shape[1])}
            for l in range(self.pos[i].shape[1]):
                aux=self.pos[i][:,l,cy,cx]#[cy,cx,:,l]
                el["def"]["dy"][l]=aux[0]
                el["def"]["dx"][l]=aux[1]
                mov=mov+aux*2**(-l)
            el["ry"]+=mov[0]
            el["rx"]+=mov[1]
            rdet.append(el)
        return rdet

    def bbox(self,det,redy=0,redx=0):
        """
        convert a list of detections in (id,y1,x1,y2,x2,scr)
        """
        bb=[]
        for el in det:
            l=el.copy()
            y1=l["ry"]/l["scl"]*self.sbin
            x1=l["rx"]/l["scl"]*self.sbin
            y2=(l["ry"]+self.fy)/l["scl"]*self.sbin
            x2=(l["rx"]+self.fx)/l["scl"]*self.sbin
            if l.has_key("endy"):
                y2=(l["endy"])/l["scl"]*self.sbin
                x2=(l["endx"])/l["scl"]*self.sbin
            l["bbox"]=[y1,x1,y2,x2]
            bb.append(l)
        return bb

    def cluster(self,det,ovr=0.5,maxcl=20,inclusion=False):
        """
        cluster detection with a minimum area k of overlapping
        """
        cllist=[]
        for ls in det:
            found=False
            for cl in cllist:
                for cle in cl:
                    if not(inclusion):
                        myovr=util.overlap(ls["bbox"],cle["bbox"])
                    else:   
                        myovr=util.inclusion(ls["bbox"],cle["bbox"])
                    if myovr>ovr:
                        cl.append(ls)
                        found=True
                        break
            if not(found):
                if len(cllist)<maxcl:
                    cllist.append([ls])
                else:
                    break
        return [el[0] for el in cllist]

    def rawdet(self,det):
        """
        extract features from detections and store in "feat"
        """        
        rdet=det[:]
        hogdim=31
        if self.trunc:
            hogdim=32
        for item in det:
            i=item["i"];cy=item["ny"];cx=item["nx"];
            fy=self.fy
            fx=self.fx
            item["feat"]=[]
            item["feat2"]=[]
            my=0;mx=0;
            for l in range(len(item["def"]["dy"])):
                if i+self.f.starti-(l)*self.interv>=0:
                    my=2*my+item["def"]["dy"][l]
                    mx=2*mx+item["def"]["dx"][l]
                    aux=getfeat(self.f.hog[i+self.f.starti-(l)*self.interv],cy+my-1,cy+my+fy*2**l-1,cx+mx-1,cx+mx+fx*2**l-1,self.trunc)
                    item["feat"].append(aux)
                    #item["feat2"].append(hog2bow(aux))
                    cy=(cy)*2
                    cx=(cx)*2
                else:
                    item["feat"].append(numpy.zeros((fy*2**l,fx*2**l,hogdim)))
        return rdet


    def show(self,ldet,parts=False,colors=["w","r","g","b"],thr=-numpy.inf,maxnum=numpy.inf,scr=True,cls=None):
        """
        show the detections in an image
        """        
        ar=[5,4,2]
        count=0
        for item in ldet:
            nlev=0
            if item.has_key("def"):
                nlev=len(item["def"]["dy"])
            if item["scr"]>thr:
                bbox=item["bbox"]
                if parts:
                    d=item["def"]
                    scl=item["scl"]
                    mx=0
                    my=0
                    for l,val in enumerate(d["dy"]):
                        my+=d["dy"][l]*2**-l
                        mx+=d["dx"][l]*2**-l
                        y1=(item["ny"]+my)*self.f.sbin/scl
                        x1=(item["nx"]+mx)*self.f.sbin/scl
                        y2=(item["ny"]+my+item["fy"])*self.f.sbin/scl
                        x2=(item["nx"]+mx+item["fx"])*self.f.sbin/scl
                        pylab.fill([x1,x1,x2,x2,x1],[y1,y2,y2,y1,y1],colors[1+l], alpha=0.1, edgecolor=colors[1+l],lw=ar[l],fill=False)        
                util.box(bbox[0],bbox[1],bbox[2],bbox[3],colors[0],lw=2)
                if item["i"]-(nlev-1)*self.interv>=-self.f.starti:#no occlusion
                    strsmall=""
                else:
                    strsmall="S%d"%(-((item["i"]+self.f.starti-(nlev-1)*self.interv)/self.interv))
                if scr:
                    if cls!=None:
                        pylab.text(bbox[1],bbox[0],"%d %.3f %s"%(item["cl"],item["scr"],cls),bbox=dict(facecolor='w', alpha=0.5),fontsize=10)
                    else:
                        if item["cl"]==0:
                            pylab.text(bbox[1],bbox[0],"%.3f %s"%(item["scr"],strsmall),bbox=dict(facecolor='w',alpha=0.5),fontsize=10)
                        else:
                            pylab.text(bbox[1],bbox[0],"%d %.3f %s"%(item["cl"],item["scr"],strsmall),bbox=dict(facecolor='w', alpha=0.5),fontsize=10)                            
            count+=1
            if count>maxnum:
                break
            
    def descr(self,det,flip=False,usemrf=False,usefather=False,k=0,usebow=False):
        """
        convert each detection in a feature descriptor for the SVM
        """           
        ld=[]
        for item in det:
            d=numpy.array([],dtype=numpy.float32)
            for l in range(len(item["feat"])):
                if not(flip):
                    aux=item["feat"][l]
                else:
                    aux=hogflip(item["feat"][l])
                d=numpy.concatenate((d,aux.flatten()))
                if self.occl:
                    if item["i"]-l*self.interv>=-self.f.starti:
                        d=numpy.concatenate((d,[0.0]))
                    else:
                        d=numpy.concatenate((d,[1.0*SMALL]))
            #ld.append(d.astype(numpy.float32))
            if self.small2:
                aux=numpy.array([0.0,0.0,0.0])
                if item["i"]<10:
                    aux[0]=1.0*SMALL
                elif item["i"]<20:
                    aux[1]=1.0*SMALL
                d=numpy.concatenate((d,aux))
            if usebow:#usebow:#625*3-->later 625*21
                for l in range(len(item["feat"])):
                    #hist=numpy.zeros(625,dtype=numpy.float32)
                    if not(flip):
                        hist=hog2bow(item["feat"][l])
                    else:
                        #hist=hog2bow((item["feat"][l]))
                        #hist=hog2bow(hogflip(item["feat"][l].astype(numpy.float32)))
                        hist=hog2bow((item["feat"][l]))[histflip()]#why this works and the other no?
                    d=numpy.concatenate((d,hist))
            ld.append(d.astype(numpy.float32))
        return ld

    def mixture(self,det): 
        """
        returns the mixture number if the detector is a mixture of models
        """    
        ld=[]
        for item in det:
            ld.append(item["cl"])
        return ld

#    def model(self,descr,rho,lev,fsz,fy=[],fx=[],usemrf=False,usefather=False,usebow=False,numbow=6**4):
#        """
#        build a new model from the weights of the SVM
#        """     
#        ww=[]  
#        p=0
#        occl=[0]*lev
#        if fy==[]:
#            fy=self.fy
#        if fx==[]:
#            fx=self.fx
#        d=descr
#        for l in range(lev):
#            dp=(fy*fx)*4**l*fsz
#            ww.append((d[p:p+dp].reshape((fy*2**l,fx*2**l,fsz))).astype(numpy.float32))
#            p+=dp
#            if self.occl:
#                occl[l]=d[p]
#                p+=1
#        hist=[]
#        if usebow:#usebow:
#            for l in range(lev):
#                hist.append(d[p:p+numbow].astype(numpy.float32)) 
#                #hist.append(numpy.zeros(625,dtype=numpy.float32))#remind to remove this line
#                p=p+numbow
#        m={"ww":ww,"rho":rho,"fy":fy,"fx":fx,"occl":occl,"hist":hist,"voc":[]}
#        return m

    def getbestworste(self,det,gtbbox,numpos=1,numneg=10,posovr=0.5,negovr=0.2,mpos=0,minnegovr=0,minnegincl=0,emptybb=True,useMaxOvr=False):
        """
        returns the detection that best overlap with the ground truth and also the best not overlapping
        """    
        lpos=[]
        lneg=[]
        lnegfull=False
        for gt in gtbbox:
            lpos.append(gt.copy())
            lpos[-1]["scr"]=-numpy.inf
            lpos[-1]["ovr"]=1.0
        for d in det:
            goodneg=True
            for gt in lpos: 
                ovr=util.overlap(d["bbox"],gt["bbox"])
                incl=util.inclusion(d["bbox"],gt["bbox"])
                if ovr>posovr:
                    if d["scr"]-mpos*(1-ovr)>gt["scr"]-mpos*(1-gt["ovr"]):
                        gt["scr"]=d["scr"]
                        gt["ovr"]=ovr
                        gt["data"]=d.copy()
                if ovr>negovr or ovr<minnegovr or incl<minnegincl:
                    goodneg=False
            if goodneg and not(lnegfull):
                noovr=True
                for n in lneg:
                    ovr=util.overlap(d["bbox"],n["bbox"])
                    if ovr>0.5:
                        noovr=False
                if noovr:
                    if len(lneg)>=numneg:   
                        lnegfull=True
                    else:
                        lneg.append(d)
        lpos2=[]
        for idbbox,gt in enumerate(lpos):
            if gt["scr"]>-numpy.inf:
                lpos2.append(gt["data"])
                lpos2[-1]["ovr"]=gt["ovr"]
                lpos2[-1]["gtbb"]=gt["bbox"]
                lpos2[-1]["bbid"]=idbbox
                if gt.has_key("img"):
                    lpos2[-1]["img"]=gt["img"]
            else:
                if emptybb:
                    lpos2.append({"scr":-10,"bbox":gt["bbox"],"notfound":True})#not detected bbox
        return lpos2,lneg

    def getbestworste_wrong(self,det,gtbbox,numpos=1,numneg=10,posovr=0.5,negovr=0.2,mpos=0,minnegovr=0,minnegincl=0,emptybb=True,useMaxOvr=False):
        """
        returns the detection that best overlap with the ground truth and also the best not overlapping
        """    
        lpos=[]
        lneg=[]
        #lnegfull=False
        for gt in gtbbox:
            lpos.append(gt.copy())
            lpos[-1]["scr"]=-numpy.inf
            lpos[-1]["ovr"]=1.0
        fullcount=numpy.ones(len(lpos))*numneg
        for d in det:
            #goodneg=True
            #lnegfull=False
            for idgt,gt in enumerate(lpos): 
                goodneg=True
                ovr=util.overlap(d["bbox"],gt["bbox"])
                incl=util.inclusion(d["bbox"],gt["bbox"])
                if useMaxOvr:
                    ovr=util.myinclusion(gt["bbox"],d["bbox"])
                    #ovr=util.myinclusion(gt["bbox"],d["bbox"])
                    #print "OVR:",ovr
                    #raw_input()
                    #print posovr
                if ovr>posovr:#good positive
                    if d["scr"]-mpos*(1-ovr)>gt["scr"]-mpos*(1-gt["ovr"]):
                        gt["scr"]=d["scr"]
                        gt["ovr"]=ovr
                        gt["data"]=d.copy()
                        #print "GT:",gt["bbox"]
                        #print "D:",d["bbox"]
                        #print "OVR",ovr
                        #raw_input()
                if ovr>negovr or ovr<minnegovr or incl<minnegincl:
                    goodneg=False
                #negative for each gt bbox
                if goodneg and fullcount[idgt]>0:#not(lnegfull):
                    noovr=True
                    for n in lneg:
                        ovr=util.overlap(d["bbox"],n["bbox"])
                        if ovr>0.5:
                            noovr=False
                    if noovr:
                        #print "XXX",idgt,fullcount[idgt]
                        if fullcount[idgt]>0:#len(lneg)>=numneg:   
                            #print "Added 1 Negative in GT ",idgt
                            fullcount[idgt]=fullcount[idgt]-1
                            lneg.append(d)
        #print "END:",fullcount

        lpos2=[]
        for idbbox,gt in enumerate(lpos):
            if gt["scr"]>-numpy.inf:
                lpos2.append(gt["data"])
                lpos2[-1]["ovr"]=gt["ovr"]
                lpos2[-1]["gtbb"]=gt["bbox"]
                lpos2[-1]["bbid"]=idbbox
                if gt.has_key("img"):
                    lpos2[-1]["img"]=gt["img"]
            else:
                if emptybb:
                    lpos2.append({"scr":-10,"bbox":gt["bbox"],"notfound":True})#not detected bbox
        return lpos2,lneg

    def goodsamples(self,det,initr,ratio):
        f=self.f
        samples=[]
        for i in range(0,len(f.hog)):
            samples.append(numpy.mgrid[-self.fy+initr:f.hog[i].shape[0]+1:1+2*initr,-self.fx+initr:f.hog[i].shape[1]+1:1+2*initr].astype(c_int))
            csamples=samples[-1][0,:,:].copy()
            samples[-1][0,:,:]=-1000
            for d in det:
                if d["i"]==i-f.starti:
                    samples[-1][0,d["py"],d["px"]]=csamples[d["py"],d["px"]]      
        return samples

#class TreatSMALL(Treat):
#    def __init__(self,tr):
#        self.tr=tr

#    def doall(self,*arg,**args):
#        self.tr.doall(arg,args)
    
        


import crf3

class TreatCRF(Treat):

    def __init__(self,f,scr,pos,sample,fy,fx,model,pscr,ranktr,occl=False,trunc=0):
        self.pos=pos
        self.scr=scr
        self.f=f
        self.interv=f.interv
        self.sbin=f.sbin
        self.fy=fy
        self.fx=fx
        self.scale=f.scale
        self.sample=sample
        self.occl=occl
        self.trunc=trunc
        self.model=model
        self.pscr=pscr
        self.ranktr=ranktr
        self.pad=2

    def select_fast(self,thr=0,cl=0,dense=DENSE):
        thr_margin=1.0
        res=Treat.select_fast(self,thr-thr_margin,cl,dense)
        self.thr=thr
        return res

    def refine(self,ldet):
        #normal refine
        pad=self.pad
        print "Refine CRF",len(ldet)
        rdet=[]
        fy=self.fy
        fx=self.fx
        for idi,item in enumerate(ldet):
            i=item["i"];cy=item["py"];cx=item["px"];
            el=item.copy()
            el["ny"]=el["ry"]
            el["nx"]=el["rx"]
            mov=numpy.zeros(2)
            el["def"]={"dy":numpy.zeros(self.pos[i].shape[1]),"dx":numpy.zeros(self.pos[i].shape[1])}
            my=0;mx=0
            for l in range(self.pos[i].shape[1]):
                aux=self.pos[i][:,l,cy,cx]#[cy,cx,:,l]
                el["def"]["dy"][l]=aux[0]
                el["def"]["dx"][l]=aux[1]
                my=2*my+el["def"]["dy"][l]
                mx=2*mx+el["def"]["dx"][l]
                mov=mov+aux*2**(-l)
            el["ry"]+=mov[0]
            el["rx"]+=mov[1]
            el["pscr"]=self.pscr[i][cy,cx]
            rdet.append(el)
            l=len(self.model["ww"])-1
            if i+self.f.starti-(l)*self.interv>=0:
                #print idi
                #getfeat(self.f.hog[i+self.f.starti-(l)*self.interv],cy+my-1,cy+my+fy*2**l-1,cx+mx-1,cx+mx+fx*2**l-1,self.trunc)
                feat=getfeat(self.f.hog[i+self.f.starti-(l)*self.interv],el["ny"]*2**l+my-1-pad,el["ny"]*2**l+my+fy*2**l-1+pad,el["nx"]*2**l+mx-1-pad,el["nx"]*2**l+mx+fx*2**l-1+pad,self.trunc).astype(c_float)
                m=self.model["ww"][-1]
                cost=self.model["cost"]
                #nscr,ndef,dfeat,edge=crf3.match(m,feat,cost,pad=pad,show=False)
                check=0
                if check:
                    nscr,ndef,dfeat,edge=crf3.match(m,feat,cost,pad=pad,show=False)
                    el["efeat"]=feat
                    el["dfeat"]=dfeat                
                    el["edge"]=edge
                else:
                    nscr,ndef=crf3.match(m,feat,cost,pad=pad,show=False,feat=False)
                el["CRF"]=ndef
                el["oldscr"]=item["scr"]
                el["scr"]=nscr+sum(el["pscr"][:-1])-self.model["rho"]
            #else:
                #m=self.model["ww"][-1]
                #el["dfeat"]=numpy.zeros(m.shape,dtype=numpy.float32)
                #item["CRF"]=numpy.ones(m.shape,dtype=numpy.float32)
                #item["oldscr"]=item["scr"]
                #item["scr"]=nscr
        #print self.model["ww"][-1].shape,el["edge"].shape
        #rerank
        rdet=self.rank(rdet,maxnum=self.ranktr)
        #thresholding
        idel=0 
        for idel,el in enumerate(rdet):
            if el["scr"]<self.thr:
                break    
        rdet=rdet[:idel]
        return rdet

    def rawdet(self,det):
        """
        extract features from detections and store in "feat"
        """        
        rdet=det[:]
        hogdim=31
        if self.trunc:
            hogdim=32
        for item in det:
            i=item["i"];cy=item["ny"];cx=item["nx"];
            fy=self.fy
            fx=self.fx
            item["feat"]=[]
            #item["feat2"]=[]
            my=0;mx=0;
            for l in range(len(item["def"]["dy"])):
                if i+self.f.starti-(l)*self.interv>=0:
                    #if l==len(item["def"]["dy"])-1:#CRF
                        #item["feat"].append(item["dfeat"])
                    #else:
                    my=2*my+item["def"]["dy"][l]
                    mx=2*mx+item["def"]["dx"][l]
                    if l==len(item["def"]["dy"])-1:#CRF
                        aux2=getfeat(self.f.hog[i+self.f.starti-(l)*self.interv],cy+my-1-self.pad,cy+my+fy*2**l-1+self.pad,cx+mx-1-self.pad,cx+mx+fx*2**l-1+self.pad,self.trunc)               
                        aux,edge=crf3.getfeat(aux2,self.pad,item["CRF"])
#                        if numpy.sum(aux2-item["efeat"])>0.0001:
#                            print "Error rigid:",numpy.sum(aux2-item["efeat"])
#                            raw_input()
#                        if numpy.sum(aux-item["dfeat"]):
#                            print "Error deform:",numpy.sum(aux-item["dfeat"])
#                            raw_input()                            
                        item["edge"]=edge
                    else:
                        aux=getfeat(self.f.hog[i+self.f.starti-(l)*self.interv],cy+my-1,cy+my+fy*2**l-1,cx+mx-1,cx+mx+fx*2**l-1,self.trunc)
                    item["feat"].append(aux)
                    #item["feat2"].append(hog2bow(aux))
                    cy=(cy)*2
                    cx=(cx)*2
                else:
                    item["feat"].append(numpy.zeros((fy*2**l,fx*2**l,hogdim)))
        return rdet

    def descr(self,det,flip=False,usemrf=False,usefather=False,k=0.3,usebow=False):
        """
        convert each detection in a feature descriptor for the SVM
        """  
        ld=Treat.descr(self,det,flip,usemrf,usefather,k,usebow)         
        for idl,item in enumerate(det):
            #ld[idl]=numpy.concatenate((ld[idl],-abs((item["edge"]*self.model["cost"]).flatten())))
            if not(flip):
                ld[idl]=numpy.concatenate((ld[idl],(item["edge"]/float(k)).flatten()))
            else:
                aux=crfflip(item["edge"])
                #aux[0]=item["edge"][0,:,::-1]#aux[0,:,::-1]
                #aux[1,:,:-1]=item["edge"][1,:,-2::-1]#aux[1,:,-2::-1]
                ld[idl]=numpy.concatenate((ld[idl],(aux/float(k)).flatten()))
        return ld

#    def model(self,descr,rho,lev,fsz,fy=[],fx=[],usemrf=False,usefather=False,usebow=False,numbow=6**4):
#        """
#        build a new model from the weights of the SVM
#        """     
#        m=Treat.model(self,descr,rho,lev,fsz,fy,fx,usemrf,usefather,usebow,numbow)
#        p=2*2*fy*2*fx
#        m["cost"]=(d[-p:].reshape((2,2*fy,2*fx))).clip(0.001,10)
#        #import crf3
#        #cache_cost=crf3.cost(fy*2,fx*2,(fy*2*2-1)/2,(fx*2*2-1)/2,c=0.001,ch=m["cost"][0],cv=m["cost"][1])
#        #m["cache_cost"]=cache_cost
#        return m

    def show(self,ldet,parts=False,colors=["w","r","g","b"],thr=-numpy.inf,maxnum=numpy.inf,scr=True,cls=None):
        """
        show the detections in an image
        """        
        ar=[5,4,2]
        count=0
        for item in ldet:
            cpy=item["fy"]*2#self.model["cost"][0].shape[0]
            cpx=item["fx"]*2#self.model["cost"][0].shape[1]        
            nlev=0
            if item.has_key("def"):
                nlev=len(item["def"]["dy"])
            if item["scr"]>thr:
                bbox=item["bbox"]
                if parts:
                    d=item["def"]
                    scl=item["scl"]
                    mx=0
                    my=0
                    for l,val in enumerate(d["dy"]):
                        my+=d["dy"][l]*2**-l
                        mx+=d["dx"][l]*2**-l
                        y1=(item["ny"]+my)*self.f.sbin/scl
                        x1=(item["nx"]+mx)*self.f.sbin/scl
                        y2=(item["ny"]+my+item["fy"])*self.f.sbin/scl
                        x2=(item["nx"]+mx+item["fx"])*self.f.sbin/scl
                        pylab.fill([x1,x1,x2,x2,x1],[y1,y2,y2,y1,y1],colors[1+l], alpha=0.1, edgecolor=colors[1+l],lw=ar[l],fill=False)        
                        if l==len(d["dy"])-1 and parts:
                            sy=float(y2-y1)/cpy
                            sx=float(x2-x1)/cpx
                            defy=item["CRF"][0]
                            defx=item["CRF"][1]
                            for py in range(cpy):
                                for px in range(cpx):
                                    py1=y1+sy*py+defy[py,px]*sy/2
                                    px1=x1+sx*px+defx[py,px]*sx/2
                                    pylab.fill([px1,px1,px1+sx,px1+sx,px1],[py1,py1+sy,py1+sy,py1,py1],colors[0], alpha=0.1, edgecolor=colors[0],lw=2,fill=False)        
                util.box(bbox[0],bbox[1],bbox[2],bbox[3],colors[0],lw=2)
                if item["i"]-(nlev-1)*self.interv>=-self.f.starti:#no occlusion
                    strsmall=""
                else:
                    strsmall="S%d"%(-((item["i"]+self.f.starti-(nlev-1)*self.interv)/self.interv))
                if scr:
                    if cls!=None:
                        pylab.text(bbox[1],bbox[0],"%d %.3f %s"%(item["cl"],item["scr"],cls),bbox=dict(facecolor='w', alpha=0.5),fontsize=10)
                    else:
                        if item["cl"]==0:
                            pylab.text(bbox[1],bbox[0],"%.3f %s"%(item["scr"],strsmall),bbox=dict(facecolor='w',alpha=0.5),fontsize=10)
                        else:
                            pylab.text(bbox[1],bbox[0],"%d %.3f %s"%(item["cl"],item["scr"],strsmall),bbox=dict(facecolor='w', alpha=0.5),fontsize=10)                            
            count+=1
            if count>maxnum:
                break

class TreatDef(Treat):

    def refine(self,ldet):
        """
            refine the localization of the object based on higher resolutions
        """
        rdet=[]
        for item in ldet:
            i=item["i"];cy=item["py"];cx=item["px"];
            el=item.copy()
            el["ny"]=el["ry"]
            el["nx"]=el["rx"]
            mov=numpy.zeros((1,1,2))
            el["def"]={"dy":[],"dx":[],"ddy":[],"ddx":[],"party":[],"partx":[]}
            for l in range(len(self.pos[i])):
                aux=self.pos[i][l][:,:,:,cy,cx]
                el["def"]["dy"].append(aux[:,:,0])
                el["def"]["dx"].append(aux[:,:,1])
                el["def"]["ddy"].append(aux[:,:,2])
                el["def"]["ddx"].append(aux[:,:,3])
                mov=mov+aux[:,:,:2]*2**(-l)
                el["def"]["party"].append(el["ny"]+mov[:,:,0])
                el["def"]["partx"].append(el["nx"]+mov[:,:,1])
                aux1=numpy.kron(mov.T,[[1,1],[1,1]]).T
                aux2=numpy.zeros((2,2,2))
                aux2[:,:,0]=numpy.array([[0,0],[self.fy*2**-(l+1),self.fy*2**-(l+1)]])
                aux2[:,:,1]=numpy.array([[0,self.fx*2**-(l+1)],[0,self.fx*2**-(l+1)]])
                aux3=numpy.kron(numpy.ones((2**l,2**l)),aux2.T).T
                mov=aux1+aux3
            el["ry"]=numpy.min(el["def"]["party"][-1])
            el["rx"]=numpy.min(el["def"]["partx"][-1])
            el["endy"]=numpy.max(el["def"]["party"][-1])+self.fy*(2**-(l))
            el["endx"]=numpy.max(el["def"]["partx"][-1])+self.fx*(2**-(l))
            rdet.append(el)
        return rdet

    def rawdet(self,det):
        """
        extract features from detections and store in "feat"
        """        
        rdet=det[:]
        hogdim=31
        if self.trunc:
            hogdim=32
        for item in det:
            i=item["i"];cy=item["ny"];cx=item["nx"];
            fy=self.fy
            fx=self.fx
            item["feat"]=[]
            mov=numpy.zeros((1,1,2))
            for l in range(len(item["def"]["party"])):
                sz=2**l
                aux=numpy.zeros((fy*sz,fx*sz,hogdim))
                if i+self.f.starti-(l)*self.interv>=0:
                    for py in range(sz):
                        for px in range(sz):
                            mov[py,px,0]=2*mov[py,px,0]+item["def"]["dy"][l][py,px]
                            mov[py,px,1]=2*mov[py,px,1]+item["def"]["dx"][l][py,px]
                            aux[py*fy:(py+1)*fy,px*fx:(px+1)*fx,:]=getfeat(self.f.hog[i+self.f.starti-(l)*self.interv],cy+mov[py,px,0]-1,cy+mov[py,px,0]+fy-1,cx+mov[py,px,1]-1,cx+mov[py,px,1]+fx-1,self.trunc)
                    cy=(cy)*2
                    cx=(cx)*2
                    aux1=numpy.kron(mov.T,[[1,1],[1,1]]).T
                    aux2=numpy.zeros((2,2,2))
                    aux2[:,:,0]=numpy.array([[0,0],[self.fy/2.0,self.fy/2.0]])
                    aux2[:,:,1]=numpy.array([[0,self.fx/2.0],[0,self.fx/2.0]])
                    aux3=numpy.kron(numpy.ones((2**l,2**l)),aux2.T).T
                    mov=aux1+aux3
                item["feat"].append(aux)
        return rdet


    def show(self,ldet,parts=False,colors=["w","r","g","b"],thr=-numpy.inf,maxnum=numpy.inf,scr=True,cls=None):
        """
        show the detections in an image
        """  
        ar=[5,4,2]
        count=0
        if parts:
            for item in ldet:
                if item["scr"]>thr:
                    scl=item["scl"]
                    for l in range(len(item["def"]["dy"])):
                        py=item["def"]["party"][l]
                        px=item["def"]["partx"][l]
                        for lpy in range(py.shape[0]):
                            for lpx in range(px.shape[1]):
                                y1=py[lpy,lpx]*self.f.sbin/scl
                                y2=(py[lpy,lpx]+item["fy"]*2**-l)*self.f.sbin/scl
                                x1=px[lpy,lpx]*self.f.sbin/scl
                                x2=(px[lpy,lpx]+item["fx"]*2**-l)*self.f.sbin/scl
                                pylab.fill([x1,x1,x2,x2,x1],[y1,y2,y2,y1,y1],colors[1+l], alpha=0.1, edgecolor=colors[1+l],lw=ar[l],fill=False)                               
                count+=1
                if count>maxnum:
                    break
        Treat.show(self,ldet,colors=colors,thr=thr,maxnum=maxnum,scr=scr,cls=cls)        

    def descr(self,det,flip=False,usemrf=True,usefather=True,k=K,usebow=False,onlybow=False):
        """
        convert each detection in a feature descriptor for the SVM
        """      
        ld=[]
        for item in det:
            d=numpy.array([])
            for l in range(len(item["feat"])):
                if not(flip):
                    d=numpy.concatenate((d,item["feat"][l].flatten()))       
                    if l>0: #skip deformations level 0
                        if usefather:
                            d=numpy.concatenate((d, k*k*(item["def"]["dy"][l].flatten()**2)  ))
                            d=numpy.concatenate((d, k*k*(item["def"]["dx"][l].flatten()**2)  ))
                        if usemrf:
                            d=numpy.concatenate((d,k*k*item["def"]["ddy"][l].flatten()))
                            d=numpy.concatenate((d,k*k*item["def"]["ddx"][l].flatten()))
                else:
                    d=numpy.concatenate((d,hogflip(item["feat"][l]).flatten()))        
                    if l>0: #skip deformations level 0
                        if usefather:
                            aux=(k*k*(item["def"]["dy"][l][:,::-1]**2))#.copy()
                            d=numpy.concatenate((d,aux.flatten()))
                            aux=(k*k*(item["def"]["dx"][l][:,::-1]**2))#.copy()
                            d=numpy.concatenate((d,aux.flatten()))
                        if usemrf:
                            aux=defflip(k*k*item["def"]["ddy"][l])
                            d=numpy.concatenate((d,aux.flatten()))
                            aux=defflip(k*k*item["def"]["ddx"][l])
                            d=numpy.concatenate((d,aux.flatten()))
                if self.occl:
                    if item["i"]-l*self.interv>=-self.f.starti:
                        d=numpy.concatenate((d,[0.0]))
                    else:
                        d=numpy.concatenate((d,[1.0*SMALL]))
            if onlybow:#set everything to 0
                d[:]=0.0
            if usebow:
                fy=item["feat"][0].shape[0]
                fx=item["feat"][0].shape[1]
                for l in range(len(item["feat"])):
                    if not(flip):
                        #d=numpy.concatenate((d,pyrHOG2.hog2bow(item["feat"][l].flatten())))
                        auxd=numpy.zeros((2**l,2**l,6**4),dtype=numpy.float32)
                        for py in range(2**l):
                            for px in range(2**l):
                                auxd[py,px,:]=hog2bow((item["feat"][l][py*fy:(py+1)*fy,px*fx:(px+1)*fx]).copy())
                                
                    else:
                        auxd=numpy.zeros((2**l,2**l,6**4),dtype=numpy.float32)
                        for py in range(2**l):
                            for px in range(2**l):
                                auxd[py,2**l-px-1,:]=hog2bow((item["feat"][l][py*fy:(py+1)*fy,px*fx:(px+1)*fx]))[histflip()]
                    d=numpy.concatenate((d,auxd.flatten()))
            ld.append(d.astype(numpy.float32))
        return ld

    def model(self,descr,rho,lev,fsz,fy=[],fx=[],mindef=0.001,usemrf=True,usefather=True,usebow=False): 
        """
        build a new model from the weights of the SVM
        """     
        ww=[]  
        df=[]
        occl=[0]*lev
        if fy==[]:
            fy=self.fy
        if fx==[]:
            fx=self.fx
        p=0
        d=descr
        for l in range(lev):
            dp=(fy*fx)*4**l*fsz
            ww.append((d[p:p+dp].reshape((fy*2**l,fx*2**l,fsz))).astype(numpy.float32))
            p+=dp
            if l>0: #skip level 0
                ddp=4**l
                aux=numpy.zeros((2**l,2**l,4))
                if usefather:
                    aux[:,:,0]=d[p:p+ddp].reshape((2**l,2**l))
                    p+=ddp
                    aux[:,:,1]=d[p:p+ddp].reshape((2**l,2**l))
                    p+=ddp
                if usemrf:
                    aux[:,:,2]=d[p:p+ddp].reshape((2**l,2**l))
                    p+=ddp
                    aux[:,:,3]=d[p:p+ddp].reshape((2**l,2**l))
                    p+=ddp
                df.append(aux.astype(numpy.float32))
            else:
                df.append(numpy.zeros((2**l,2**l,4),dtype=numpy.float32))
            if self.occl:
                occl[l]=d[p]
                p+=1
            if usebow:
                hist=[]
                for l in range(lev):
                    hist.append(d[p:p+(4**l)*bin**(siftsize**2)].astype(numpy.float32).reshape((2**l,2**l,bin**(siftsize**2))))
                    #hist.append(numpy.zeros(625,dtype=numpy.float32))
                    p=p+(4**l)*bin**(siftsize**2)
        m={"ww":ww,"rho":rho,"fy":fy,"fx":fx,"df":df,"occl":occl,"hist":hist}
        return m

import time

def detect(f,m,gtbbox=None,auxdir=".",hallucinate=1,initr=1,ratio=1,deform=False,bottomup=False,usemrf=False,numneg=0,thr=-2,posovr=0.7,minnegincl=0.5,small=True,show=False,cl=0,mythr=-10,nms=0.5,inclusion=False,usefather=True,mpos=1,emptybb=False,useprior=False,K=1.0,occl=False,trunc=0,useMaxOvr=False,ranktr=1000,fastBU=False,usebow=False,CRF=False,small2=False):
    """Detect objects in an image
        used for both test --> gtbbox=None
        and trainig --> gtbbox = list of bounding boxes
    """
    ff.setK(K)#set the degree of deformation
    if useprior:
        numrank=200
    else:
        numrank=1000
    if gtbbox!=None and gtbbox!=[] and useprior:
        t1=time.time()
        pr=f.buildPrior(gtbbox,m["fy"],m["fx"])
        print "Prior Time:",time.time()-t1
    else:
        pr=None
    t=time.time()        
    f.resetHOG()
    #CRF=True
    if deform:
        if bottomup:
            scr,pos=f.scanRCFLDefBU(m,initr=initr,ratio=ratio,small=small,usemrf=usemrf)
        else:
            #scr,pos=f.scanRCFLDefThr(m,initr=initr,ratio=ratio,small=small,usemrf=usemrf,mythr=mythr)
            if usebow:
                scr,pos=f.scanRCFLDefbow(m,initr=initr,ratio=ratio,small=small,usemrf=usemrf,trunc=trunc)
            else:
                scr,pos=f.scanRCFLDef(m,initr=initr,ratio=ratio,small=small,usemrf=usemrf,trunc=trunc)
        tr=TreatDef(f,scr,pos,initr,m["fy"],m["fx"],occl=occl,trunc=trunc)
    else:
        if usebow:
            scr,pos=f.scanRCFLbow(m,initr=initr,ratio=ratio,small=small,trunc=trunc)
            #scr,pos=f.scanRCFL(m,initr=initr,ratio=ratio,small=small,trunc=trunc)
        else:
            if CRF:
                scr,pscr,pos=f.scanRCFL(m,initr=initr,ratio=ratio,small=small,trunc=trunc,partScr=True)
            else:                
                scr,pos=f.scanRCFL(m,initr=initr,ratio=ratio,small=small,trunc=trunc)
        if small2:
            for l in range(f.interv):
                scr[l]=scr[l]+m["small2"][0]*SMALL    
                scr[l+f.interv]=scr[l+f.interv]+m["small2"][1]*SMALL    
        if CRF:
            tr=TreatCRF(f,scr,pos,initr,m["fy"],m["fx"],m,pscr,ranktr,occl=occl,trunc=trunc)        
        else:
            tr=Treat(f,scr,pos,initr,m["fy"],m["fx"],occl=occl,trunc=trunc,small2=small2)
    print "Scan: %.3f s"%(time.time()-t)    
    if gtbbox==None:#test
        if show==True:
            showlabel="Parts"
        else:
            showlabel=False
        if fastBU:#enable TD+BU
            t1=time.time()
            det=tr.doall(thr=thr,rank=10,refine=True,rawdet=False,cluster=False,show=False,inclusion=inclusion,cl=cl)
            samples=tr.goodsamples(det,initr=initr,ratio=ratio)
            scr,pos=f.scanRCFLDefBU(m,initr=initr,ratio=ratio,small=small,usemrf=usemrf,mysamples=samples)
            print "Refine Time:",time.time()-t1
            tr=TreatDef(f,scr,pos,initr,m["fy"],m["fx"])
            print len(det)
            det=tr.doall(thr=thr,rank=100,refine=True,rawdet=False,cluster=nms,show=False,inclusion=inclusion,cl=cl)
        else:
            det=tr.doall(thr=thr,rank=100,refine=True,rawdet=False,cluster=nms,show=False,inclusion=inclusion,cl=cl)
        numhog=f.getHOG()
        dettime=time.time()-t

        if show==True:
            tr.show(det,parts=showlabel,thr=-1.0,maxnum=5)           
        return tr,det,dettime,numhog
    else:#training
        t2=time.time()
        best1,worste1=tr.doalltrain(gtbbox,thr=thr,rank=ranktr,show="Parts",mpos=mpos,numpos=1,posovr=posovr,numneg=numneg,minnegovr=0,minnegincl=minnegincl,cl=cl,emptybb=emptybb,useMaxOvr=useMaxOvr)        
        ipos=[];ineg=[]
        print "Treat Time:",time.time()-t2
        print "Detect: %.3f s"%(time.time()-t)
        return tr,best1,worste1,ipos,ineg

