#class used to manage the pyramid HOG features
#import mfeatures
import resize
import util
import numpy
import math
import pylab
import scipy.misc.pilutil
import cPickle
#import scipy.ndimage.filters as flt
import time

#global ccc
#ccc=0

DENSE=0

from numpy import ctypeslib
from ctypes import c_int,c_double,c_float
#libmrf=ctypeslib.load_library("libmyrmf.so.1.0.1","")

import ctypes
ctypes.cdll.LoadLibrary("./libexcorr.so")
ff= ctypes.CDLL("libexcorr.so")

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

ff.scaneigh.argtypes=[numpy.ctypeslib.ndpointer(dtype=c_float,ndim=3,flags="C_CONTIGUOUS"),c_int,c_int,numpy.ctypeslib.ndpointer(dtype=c_float,ndim=3,flags="C_CONTIGUOUS"),c_int,c_int,c_int,numpy.ctypeslib.ndpointer(dtype=c_int,flags="C_CONTIGUOUS"),numpy.ctypeslib.ndpointer(dtype=c_int,flags="C_CONTIGUOUS"),numpy.ctypeslib.ndpointer(dtype=c_float,flags="C_CONTIGUOUS"),numpy.ctypeslib.ndpointer(dtype=c_int,flags="C_CONTIGUOUS"),numpy.ctypeslib.ndpointer
(dtype=c_int,flags="C_CONTIGUOUS"),c_int,c_int,c_int]

ff.scaneighpr.argtypes=[numpy.ctypeslib.ndpointer(dtype=c_float,ndim=3,flags="C_CONTIGUOUS"),c_int,c_int,numpy.ctypeslib.ndpointer(dtype=c_float,ndim=3,flags="C_CONTIGUOUS"),c_int,c_int,c_int,numpy.ctypeslib.ndpointer(dtype=c_int,flags="C_CONTIGUOUS"),numpy.ctypeslib.ndpointer(dtype=c_int,flags="C_CONTIGUOUS"),numpy.ctypeslib.ndpointer(dtype=c_float,flags="C_CONTIGUOUS"),numpy.ctypeslib.ndpointer(dtype=c_int,flags="C_CONTIGUOUS"),numpy.ctypeslib.ndpointer
(dtype=c_int,flags="C_CONTIGUOUS"),c_int,c_int,c_int,ctypes.POINTER(c_float)]


#ff.scanDef.argtypes = [
#    ctypeslib.ndpointer(c_float),ctypeslib.ndpointer(c_float),ctypeslib.ndpointer(c_float),ctypeslib.ndpointer(c_float),
#    c_int,c_int,c_int,
#    ctypeslib.ndpointer(c_float),ctypeslib.ndpointer(c_float),ctypeslib.ndpointer(c_float),ctypeslib.ndpointer(c_float),
#    ctypeslib.ndpointer(c_float),
#    c_int,c_int,ctypeslib.ndpointer(c_int),ctypeslib.ndpointer(c_int),
#    ctypeslib.ndpointer(c_int),
#    ctypeslib.ndpointer(c_float),
#    c_int,c_int,c_int,ctypeslib.ndpointer(c_float),c_int,
#    ctypes.POINTER(c_float),c_int,c_int]#ctypeslib.ndpointer(c_float,ndim=3,flags="C_CONTIGUOUS")]
#ff.scanDef.restype=ctypes.c_float

ff.scanDef2.argtypes = [
    ctypeslib.ndpointer(c_float),ctypeslib.ndpointer(c_float),ctypeslib.ndpointer(c_float),ctypeslib.ndpointer(c_float),
    c_int,c_int,c_int,
    ctypeslib.ndpointer(c_float),ctypeslib.ndpointer(c_float),ctypeslib.ndpointer(c_float),ctypeslib.ndpointer(c_float),
    ctypeslib.ndpointer(c_float),
    c_int,c_int,ctypeslib.ndpointer(c_int),ctypeslib.ndpointer(c_int),
    ctypeslib.ndpointer(c_int),
    ctypeslib.ndpointer(c_float),
    c_int,c_int,c_int,ctypeslib.ndpointer(c_float),c_int,
    ctypes.POINTER(c_float),c_int,c_int]#ctypeslib.ndpointer(c_float,ndim=3,flags="C_CONTIGUOUS")]
ff.scanDef2.restype=ctypes.c_float



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

def hogflip_old(feat,obin=18):
    """    
    returns the orizontally flipped version of the HOG features
    """
    aux=feat[:,::-1,:]
    last=obin+obin/2
    aux=numpy.concatenate((aux[:,:,obin/2::-1],aux[:,:,obin-1:obin/2:-1],aux[:,:,obin].reshape(aux.shape[0],aux.shape[1],1,aux.shape[3]),aux[:,:,last-1:obin:-1],aux[:,:,last:last+2],aux[:,:,last+3:last+1:-1]),2)
    return aux

def hogflip_last(feat,obin=9):
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
    #aux=numpy.concatenate((oriented,noriented,norm1,norm2,norm3,norm4),2) #remember you have to change this!!!!!!!!
    aux=numpy.concatenate((noriented,oriented,norm1,norm2,norm3,norm4),2)
    return aux

def hogflip(feat,obin=9):#pedro
    """    
    returns the orizontally flipped version of the HOG features
    """
    #feature shape
    #[9 not oriented][18 oriented][4 normalization]
    p=numpy.array([10,9,8,7,6,5,4,3,2,1,18,17,16,15,14,13,12,11,19,27,26,25,24,23,22,21,20,30,31,28,29])-1
    aux=feat[:,::-1,p]
    return aux

def defflip(feat):
    #print feat
    sx=feat.shape[1]/2-1
    fflip=numpy.zeros(feat.shape,dtype=feat.dtype)
    for ly in range(feat.shape[0]/2):
        for lx in range(feat.shape[1]/2):
            fflip[ly*2:(ly+1)*2,lx*2:(lx+1)*2]=feat[ly*2:(ly+1)*2,(sx-lx)*2:(sx-lx+1)*2].T
    #print fflip
    #raw_input()
    return fflip

class container(object):
    def __init__(self,objarray,ptrarray):
        self.obj=objarray
        self.ptr=ptrarray

class pyrHOG:
    def __init__(self,im,interv=10,sbin=8,savedir="./",compress=False,notload=False,notsave=False,hallucinate=0,cformat=False):
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
        if isinstance(im,pyrHOG):#build a copy
            self.__copy(im)
            #return
        if isinstance(im,str):
            self._precompute(im,interv,sbin,savedir,compress,notload,notsave,hallucinate,cformat=cformat)
            #return
        if type(im)==numpy.ndarray:
            self._compute(im,interv,sbin,hallucinate,cformat=cformat)
            #return
        print "Features: ",time.time()-t
        #raise "Error: im must be either a string or an image"
        
    def _compute(self,img,interv=10,sbin=8,hallucinate=0,cformat=False):
        """
        Compute the HOG pyramid of an image in a list
        """
        l=[]
        scl=[]
        octimg=img.astype(numpy.float)#copy()
        maxoct=int(numpy.log2(int(numpy.min(img.shape[:-1])/sbin)))-1#-2
        intimg=octimg
        if hallucinate>1:
            #halucinate features
            for i in range(interv):
                if cformat:
                    l.append(numpy.ascontiguousarray(hog(intimg,sbin/4),numpy.float32))
                else:
                    l.append(hog(intimg,sbin/4).astype(numpy.float32))                    
                intimg=resize.resize(octimg,math.pow(2,-float(i+1)/interv))
                scl.append(4.0*2.0**(-float(i)/interv))
        if hallucinate>0:
            #halucinate features
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
                #print l[-1].flags
                scl.append(2.0**(-o-float(i)/interv))
                #print "hog:",time.time()-t1
                t2=time.time()
                #intimg=resize.resize(octimg.astype(numpy.float),math.pow(2,-float(i+1)/interv))
                intimg=resize.resize(octimg,math.pow(2,-float(i+1)/interv))
                #pylab.imshow(intimg)
                #pylab.show()
                #pylab.draw()
                #raw_input()
                #print "resize:",time.time()-t2
            octimg=intimg
        self.hog=l
        self.interv=interv
        self.oct=maxoct
        self.sbin=sbin
        self.scale=scl
        self.hallucinate=hallucinate

    def _computeold(self,img,interv=10,sbin=8):
        """
        Compute the HOG pyramid of an image in a list
        """
        import time
        t=time.time()
        l=[]
        octimg=img.astype(numpy.float)#copy()
        maxoct=int(numpy.log2(int(numpy.min(img.shape[:-1])/sbin)))-1#-2
        #intimg=octimg
        for o in range(maxoct):
            intimg=octimg
            for i in range(interv):
                t1=time.time()
                l.append(hog(intimg,sbin))
                #print "hog:",time.time()-t1
                t2=time.time()
                #intimg=resize.resize(octimg.astype(numpy.float),math.pow(2,-float(i+1)/interv))
                intimg=resize.resize(octimg,math.pow(2,-float(i+1)/interv))
                #pylab.imshow(intimg)
                #pylab.show()
                #pylab.draw()
                #raw_input()
                #print "resize:",time.time()-t2
            octimg=intimg
        self.hog=l
        self.interv=interv
        self.oct=maxoct
        self.sbin=sbin
        print "Features: ",time.time()-t
        
    def _precompute(self,imname,interv=10,sbin=8,savedir="./",compress=False,notload=False,notsave=True,hallucinate=0,cformat=False):
        """
        Check if the HOG if imname is already computed, otherwise 
        compute it and save in savedir
        """
        try:
            if notload:
                #generate an error to pass to computing hog
                error
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
            #if imname.split(".")[-1]=="png":
            #    img=pylab.imread(imname)
            #else:
            #    img=scipy.misc.pilutil.imread(imname)
            img=util.myimread(imname)
            if img.ndim<3:
                aux=numpy.zeros((img.shape[0],img.shape[1],3))
                aux[:,:,0]=img
                aux[:,:,1]=img
                aux[:,:,2]=img
                img=aux
            self._compute(img,interv=interv,sbin=sbin,hallucinate=hallucinate,cformat=cformat)
            if notsave:#savedir=="/home/databases/hog/":#"/home/databases/hog/":
                return
            f=[]
            if compress:
                f=gzip.open(savedir+imname.split("/")[-1]+".zhog%d_%d_%d"%(self.interv,self.sbin,hallucinate),"wb")
            else:
                f=open(savedir+imname.split("/")[-1]+".hog%d_%d_%d"%(self.interv,self.sbin,hallucinate),"w")
            cPickle.dump(self,f,2)
        
    def show(self,oct=[],interv=[]):
        """
        Draw the pyramid of HOG features in a figure
        """
        if oct==[]:
            oct=range(self.oct)
        if interv==[]:
            interv=range(self.interv)
        cnt=0
        for o in oct:
            for i in interv:
                pos=o*self.interv+i
                pylab.subplot(len(oct),len(interv),cnt+1)
                pylab.title("Oct: %d Int: %d"%(o,i))
                drawHOG2(self.hog[pos])
                cnt+=1
    
    def ifsupp(self,p1y,p1x,p2y,p2x,fy,fx,retfeat=False):
        """
            transform from the image space coordinates to the feature space
            and return the corresponding features
        """
        sbin=self.sbin
        dy=abs(p1y-p2y)/(fy)#/2+1)
        dx=abs(p1x-p2x)/(fx)#/2+1)
        d=max(dy,dx)#or mean
        i=numpy.arange(len(self.hog)).astype(numpy.float)
        s=sbin*2**(i/self.interv)
        ibest=numpy.argmin(abs(s-d))
        sbest=s[ibest]
        print d,sbest
       
        cfy=(float((p1y+p2y)/2.0-sbest)/float(sbest))
        cfx=(float((p1x+p2x)/2.0-sbest)/float(sbest))
        
        #print cfy,cfx,fy,fx
        rrect=(cfy-float(fy)/2,cfx-float(fx)/2,cfy+float(fy)/2,cfx+float(fx)/2)
        rect=numpy.floor(numpy.array(rrect)+0.5)
        print rrect,rect
        if retfeat:
            feat=util.getfeat(self.hog[ibest],rect[0],rect[2],rect[1],rect[3])
            return ibest,sbest,rect,ovr,feat
        ovr=util.overlap(rrect,rect)
        return ibest,sbest,rect,ovr
    
    def fisupp(self,i,hy,hx,h2y,h2x):
        """
            transform from feature sapce to image space coordinates
        """
        sbin=self.sbin
        s=sbin*2**(float(i)/self.interv)
        nhy= s*(hy)+s
        nhx= s*(hx)+s
        nh2y= s*(h2y)+s
        nh2x= s*(h2x)+s
        return (nhy,nhx,nh2y,nh2x)
    
    def fmov(self,i,hy,hx,h2y,h2x,d,retfeat=False):
        """
        returns the closer features in (interval=i+d) closest to the original bb
        """
        sbin=self.sbin
        fy=abs(hy-h2y)
        fx=abs(hx-h2x)
        s=sbin*2**(float(i)/self.interv)
        cpy=s*(hy+h2y)/2-s
        cpx=s*(hx+h2x)/2-s
        s1=sbin*2**(float(i+d)/self.interv)
        cfy=(cpy/s1+1)
        cfx=(cpx/s1+1)
        
        rrect=(cfy-float(fy)/2,cfx-float(fx)/2,cfy+float(fy)/2,cfx+float(fx)/2)
        rect=numpy.round(rrect)
        if retfeat:
            feat=util.getfeat(self.hog[i+d],rect[0],rect[1],rect[2],rect[3])
            return i+d,s1,rect,ovr,feat
        ovr=util.overlap(rrect,rect)
        #sf
        return i+d,s1,rect,ovr

    def scan(self,ww,rho,filter=None,minlev=1):
        """
        scan the fature pyramid to detect where the SVM model gives higher response
        """
        print "scan"
        ttot=0
        tm=0
        scn=0
        list=[]
        winlist=[]
        win=filter
        for intrv in range(len(self.hog)):
            auxlist=[]
            for oct in range(len(self.hog[intrv])):
                if oct<minlev:
                    win=-10*numpy.ones(self.hog[intrv][oct].shape[:2])
                else:
                    if filter==None:
                        win=numpy.array([])
                    win=scanFull(self.hog,intrv,ww,rho,0,win)
                auxlist.append(win)
            winlist.append(auxlist)                        
        return winlist

    #useful to convert scanRCFL into C from Bishop
    class ListPOINTER(object):
        '''Just like a POINTER but accept a list of ctype as an argument'''
        def __init__(self, etype):
            self.etype = etype
    
        def from_param(self, param):
            if isinstance(param, (list,tuple)):
                return (self.etype * len(param))(*param)

    def resetHOG(self):
        "reset the HOG computation counter"
        ff.resetHOG()

    def getHOG(self):
        "get the HOG computation counter"
        return ff.getHOG()

    def buildPrior(self,gt,fy,fx,py=1,px=1,v=1):
        pr=[]
        cnt=0
        #print "Len HOG:", len(self.hog)
        for i in range(len(self.hog)):
            #samples=numpy.mgrid[-ww[0].shape[0]+initr:self.hog[i].shape[0]+1:1+2*initr,-ww[0].shape[1]+initr:self.hog[i].shape[1]+1:1+2*initr].astype(ctypes.c_int)
            prior=numpy.zeros((self.hog[i].shape[0],self.hog[i].shape[1]),numpy.float32)
            scl=self.scale[i]
            for l in gt:
                bbx=l["bbox"]
                #h=(bbx[2]-bbx[0])/2
                #w=(bbx[3]-bbx[1])/2
                #print fy,fx
                #hg=32;ddy=0.9;ddx=1.1;
                #bbx=(hg*ddy,hg*ddx,hg*(ddy+fy),hg*(ddx+fx))
                #print bbx
                cy1=(float(bbx[0])*scl/self.sbin)-1#-1
                cx1=(float(bbx[1])*scl/self.sbin)-1#-1
                cy2=(float(bbx[2])*scl/self.sbin)-1#-1
                cx2=(float(bbx[3])*scl/self.sbin)-1#-1
                #print float(8)/scl,cy,cx,cy1,cx1
                for dy in range(-int(py+1),int(py+1)+1):#[-1,0,+1]:
                    for dx in range(-int(px+1),int(px+1)+1):
                        cpy=round(cy1+dy)
                        cpx=round(cx1+dx)
                        cpy1=cy2+dy
                        cpx1=cx2+dx
                        cyr=cpy+fy
                        cxr=cpx+fx
                        #print "Scale",scl
                        #print "Cy1",cy1,"cyr",cyr
                        #print "Cx1",cx1,"cxr",cxr
                        dsy1=abs(cy1-cpy);dsx1=abs(cx1-cpx)
                        dsy2=abs(cy2-cyr);dsx2=abs(cx2-cxr)
                        if dsy1<=py and dsx1<=px and dsy2<=py and dsx2<=px:
                            #if int(cpy)>=0 and int(cpx)>=0 and int(cpy)<prior.shape[0] and int(cpx)<prior.shape[1]:
                            if round(cpy)>=0 and round(cpx)>=0 and round(cpy)<prior.shape[0] and round(cpx)<prior.shape[1]:
                                #print "Prior:",round(cpy),round(cpx)
                                #print "Distances",dsy1,dsx1,dsy2,dsx2,v*(py-dsy1)/py*(px-dsx1)/px*(py-dsy2)/py*(px-dsx2)/px
                                cnt+=1
                                prior[round(cpy),round(cpx)]=v*(py-dsy1)/py*(px-dsx1)/px*(py-dsy2)/py*(px-dsx2)/px
                                #prior[int(cpy),int(cpx)]=v*(py-dsy)/py*(px-dsx)/px
            pr.append(prior)
            #if numpy.any(prior!=0):        
            #    pylab.figure(89)
            #    pylab.imshow(prior,interpolation="nearest")
            #    pylab.show()
            #    raw_input()
        print "Prior Locations:",cnt
        return pr                     


    def scanRCFL(self,model,initr=1,ratio=1,small=True):
        ww=model["ww"]
        rho=model["rho"]
        res=[]#score
        pparts=[]#parts position
        tot=0
        if not(small):
            self.starti=self.interv*(len(ww)-1)
        else:
            self.starti=0
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
                        sshape[0]*sshape[1])
                        #res[-1].size)
                    res[i-self.starti]+=auxres
                    samples[:,:,:]=(samples[:,:,:]+pparts[-1][:,lev,:,:])*2+1
                    #samples[1,:,:]=(samples[1,:,:]+pparts[-1][1,lev,:,:])*2+1
                else:
                    pass
                    #set occlusion
            res[i-self.starti]-=rho
        return res,pparts

    def scanRCFLpr(self,model,initr=1,ratio=1,small=True,pr=None,dense=DENSE):
        ww=model["ww"]
        rho=model["rho"]
        res=[]#score
        pparts=[]#parts position
        tot=0
        if not(small):
            self.starti=self.interv*(len(ww)-1)
        else:
            self.starti=0
        from time import time
        for i in range(self.starti,len(self.hog)):
            if len(self.hog)-i<dense:
                initr=0
            samples=numpy.mgrid[-ww[0].shape[0]+initr:self.hog[i].shape[0]+1:1+2*initr,-ww[0].shape[1]+initr:self.hog[i].shape[1]+1:1+2*initr].astype(ctypes.c_int)
            sshape=samples.shape[1:3]
            res.append(numpy.zeros(sshape,dtype=ctypes.c_float))
            pparts.append(numpy.zeros((2,len(ww),sshape[0],sshape[1]),dtype=c_int))
            for lev in range(len(ww)):
                if i-self.interv*lev>=0:
                    #print "Lev:",lev
                    if lev==0:
                        r=initr
                        if pr!=None:
                            ipr=pr[i].ctypes.data_as(ctypes.POINTER(c_float))
                        else:
                            ipr=ctypes.POINTER(c_float)()
                    else:
                        ipr=ctypes.POINTER(c_float)()
                        r=ratio
                    auxres=res[-1].copy()
                    ff.scaneighpr(self.hog[i-self.interv*lev],
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
                        sshape[0]*sshape[1],
                        ipr)
                        #res[-1].size)
                    res[i-self.starti]+=auxres
                    samples[:,:,:]=(samples[:,:,:]+pparts[-1][:,lev,:,:])*2+1
                    #samples[1,:,:]=(samples[1,:,:]+pparts[-1][1,lev,:,:])*2+1
                else:
                    pass
                    #set occlusion
            res[i-self.starti]-=rho
        return res,pparts


    def scanRCFLDef(self,model,initr=1,ratio=1,small=True,usemrf=True):
        ww=model["ww"]
        rho=model["rho"]
        df=model["df"]
        res=[]#score
        pparts=[]#parts position
        tot=0
        if not(small):
            self.starti=self.interv*(len(ww)-1)
        else:
            self.starti=0
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
                nelem)
            samples[:,:,:]=(samples[:,:,:]+pparts[-1][0][0,0,:2,:,:])*2+1
            #self.scanRCFLPart(model,samples,pparts[-1],res[-1],i-self.interv,1,0,0,ratio,usemrf) 
            self.scanRCFLPart(model,samples,pparts[-1],res[i-self.starti],i-self.interv,1,0,0,ratio,usemrf) 
            res[i-self.starti]-=rho
            #pylab.figure(1)
            #pylab.imshow(res[-1],interpolation="nearest")
            #pylab.show()
            #print "Int",i,"Scr:",res[-1].max()
            #print "RES:",res[-1].max()
            #raw_input()
        return res,pparts

    def scanRCFLprDef(self,model,initr=1,ratio=1,small=True,usemrf=True,pr=None):
        ww=model["ww"]
        rho=model["rho"]
        df=model["df"]
        res=[]#score
        pparts=[]#parts position
        tot=0
        if not(small):
            self.starti=self.interv*(len(ww)-1)
        else:
            self.starti=0
        from time import time
        for i in range(self.starti,len(self.hog)):
            samples=numpy.mgrid[-ww[0].shape[0]+initr:self.hog[i].shape[0]+1:1+2*initr,-ww[0].shape[1]+initr:self.hog[i].shape[1]+1:1+2*initr].astype(c_int)
            sshape=samples.shape[1:3]
            res.append(numpy.zeros(sshape,dtype=ctypes.c_float))
            pparts.append([])
            nelem=(sshape[0]*sshape[1])
            for l in range(len(ww)):
                pparts[-1].append(numpy.zeros((2**l,2**l,4,sshape[0],sshape[1]),dtype=c_int))
            if pr!=None:
                ipr=pr[i].ctypes.data_as(ctypes.POINTER(c_float))
            else:
                ipr=ctypes.POINTER(c_float)()
            ff.scaneighpr(self.hog[i],
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
                nelem,ipr)
            samples[:,:,:]=(samples[:,:,:]+pparts[-1][0][0,0,:2,:,:])*2+1
            #self.scanRCFLPart(model,samples,pparts[-1],res[-1],i-self.interv,1,0,0,ratio,usemrf) 
            if len(ww)>1:
                self.scanRCFLPart(model,samples,pparts[-1],res[i-self.starti],i-self.interv,1,0,0,ratio,usemrf) 
            res[i-self.starti]-=rho
            #pylab.figure(1)
            #pylab.imshow(res[-1],interpolation="nearest")
            #pylab.show()
            #print "Int",i,"Scr:",res[-1].max()
            #print "RES:",res[-1].max()
            #raw_input()
        return res,pparts

    def scanRCFLPart(self,model,samples,pparts,res,i,lev,locy,locx,ratio,usemrf):
        #if len(model["ww"])<=lev:
        #    return
        #print "Level",lev,"Interval",i
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
        #print "Calling from here"
        ff.scanDef2(ww1,ww2,ww3,ww4,fy,fx,ww1.shape[2],df1,df2,df3,df4,self.hog[i],self.hog[i].shape[0],self.hog[i].shape[1],samples[0,:,:],samples[1,:,:],parts,auxres,ratio,samples.shape[1]*samples.shape[2],usemrf,numpy.array([],dtype=numpy.float32),0,ctypes.POINTER(c_float)(),0,0)
        res+=auxres
        pparts[lev][(locy+0):(locy+2),(locx+0):(locx+2),:,:,:]=parts
        if i-self.interv>=0 and len(model["ww"])-1>lev:
            samples1=(samples+parts[0,0,:2,:,:])*2+1
            #self.scanRCFLPart(model,samples1.copy(),pparts,res,i-self.interv,lev+1,(locx+0),(locx+0),ratio,usemrf)
            self.scanRCFLPart(model,samples1.copy(),pparts,res,i-self.interv,lev+1,(locy+0),(locx+0),ratio,usemrf)
            samples2=((samples.T+parts[0,1,:2,:,:].T+numpy.array([0,fx],dtype=c_int).T)*2+1).T
            #self.scanRCFLPart(model,samples2.copy(),pparts,res,i-self.interv,lev+1,(locx+0),(locx+1),ratio,usemrf)
            self.scanRCFLPart(model,samples2.copy(),pparts,res,i-self.interv,lev+1,(locy+0),(locx+1),ratio,usemrf)
            samples3=((samples.T+parts[1,0,:2,:,:].T+numpy.array([fy,0],dtype=c_int).T)*2+1).T
            self.scanRCFLPart(model,samples3.copy(),pparts,res,i-self.interv,lev+1,(locy+1),(locx+0),ratio,usemrf)
            samples4=((samples.T+parts[1,1,:2,:,:].T+numpy.array([fy,fx],dtype=c_int).T)*2+1).T
            self.scanRCFLPart(model,samples4.copy(),pparts,res,i-self.interv,lev+1,(locy+1),(locx+1),ratio,usemrf)

    def scanRCFLDefThr(self,model,initr=1,ratio=1,small=True,usemrf=True,mythr=0):
        ww=model["ww"]
        rho=model["rho"]
        df=model["df"]
        res=[]#score
        pparts=[]#parts position
        tot=0
        if not(small):
            self.starti=self.interv*(len(ww)-1)
        else:
            self.starti=0
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
                nelem)
            samples[:,:,:]=(samples[:,:,:]+pparts[-1][0][0,0,:2,:,:])*2+1
            #self.scanRCFLPart(model,samples,pparts[-1],res[-1],i-self.interv,1,0,0,ratio,usemrf) 
            self.scanRCFLPartThr(model,samples,pparts[-1],res[i-self.starti],i-self.interv,1,0,0,ratio,usemrf,mythr) 
            res[i-self.starti]-=rho
            #pylab.figure(1)
            #pylab.imshow(res[-1],interpolation="nearest")
            #pylab.show()
            #print "Int",i,"Scr:",res[-1].max()
            #print "RES:",res[-1].max()
            #raw_input()
        return res,pparts

    def scanRCFLPartThr(self,model,samples,pparts,res,i,lev,locy,locx,ratio,usemrf,mythr):
        #if len(model["ww"])<=lev:
        #    return
        #print "Level",lev,"Interval",i
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
        #print "Calling from here"
        #mythr=0#threshold value
        res[res<mythr]=-1000
        #print "Res:",res.max()
        #print res==-1000
        samples[:,res==-1000]=-1000
        #print numpy.any(samples==-1000)
        ff.scanDef2(ww1,ww2,ww3,ww4,fy,fx,ww1.shape[2],df1,df2,df3,df4,self.hog[i],self.hog[i].shape[0],self.hog[i].shape[1],samples[0,:,:],samples[1,:,:],parts,auxres,ratio,samples.shape[1]*samples.shape[2],usemrf,numpy.array([],dtype=numpy.float32),0,ctypes.POINTER(c_float)(),0,0)
        res+=auxres
        pparts[lev][(locy+0):(locy+2),(locx+0):(locx+2),:,:,:]=parts
        if i-self.interv>=0 and len(model["ww"])-1>lev:
            samples1=(samples+parts[0,0,:2,:,:])*2+1
            #self.scanRCFLPart(model,samples1.copy(),pparts,res,i-self.interv,lev+1,(locx+0),(locx+0),ratio,usemrf)
            self.scanRCFLPart(model,samples1.copy(),pparts,res,i-self.interv,lev+1,(locy+0),(locx+0),ratio,usemrf)
            samples2=((samples.T+parts[0,1,:2,:,:].T+numpy.array([0,fx],dtype=c_int).T)*2+1).T
            #self.scanRCFLPart(model,samples2.copy(),pparts,res,i-self.interv,lev+1,(locx+0),(locx+1),ratio,usemrf)
            self.scanRCFLPart(model,samples2.copy(),pparts,res,i-self.interv,lev+1,(locy+0),(locx+1),ratio,usemrf)
            samples3=((samples.T+parts[1,0,:2,:,:].T+numpy.array([fy,0],dtype=c_int).T)*2+1).T
            self.scanRCFLPart(model,samples3.copy(),pparts,res,i-self.interv,lev+1,(locy+1),(locx+0),ratio,usemrf)
            samples4=((samples.T+parts[1,1,:2,:,:].T+numpy.array([fy,fx],dtype=c_int).T)*2+1).T
            self.scanRCFLPart(model,samples4.copy(),pparts,res,i-self.interv,lev+1,(locy+1),(locx+1),ratio,usemrf)

    def scanRCFLDefBU(self,model,initr=1,ratio=1,small=True,usemrf=True):
        ww=model["ww"]
        rho=model["rho"]
        df=model["df"]
        res=[]#score
        prec=[]#precomputed scores
        pres=[]
        pparts=[]#parts position
        tot=0
        pady=model["ww"][-1].shape[0]
        padx=model["ww"][-1].shape[1]
        if not(small):
            self.starti=self.interv*(len(ww)-1)
        else:
            self.starti=0
        from time import time
        #self.starti=19 #just for debug!!!!!!!
        for i in range(self.starti,len(self.hog)):
            samples=numpy.mgrid[-ww[0].shape[0]+initr:self.hog[i].shape[0]+1:1+2*initr,-ww[0].shape[1]+initr:self.hog[i].shape[1]+1:1+2*initr].astype(c_int)
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
            #for p in range((2*initr+1)*(2*initr+1)):
            auxpparts=(numpy.zeros(((2*initr+1)*(2*initr+1),2,2,4,sshape[0],sshape[1]),dtype=c_int))
            auxptr=numpy.zeros((2*initr+1)*(2*initr+1),dtype=object)
            ct=container(auxpparts,auxptr)
            #level=1
            for l in range((2*initr+1)**2):
                maux=(numpy.zeros((4,(2*initr+1)*(2*initr+1),2,2,4,sshape[0],sshape[1]),dtype=c_int))
                auxptr=numpy.zeros((4,(2*initr+1)*(2*initr+1)),dtype=object)
                ct.ptr[l]=container(maux,auxptr)
                #for l in range(len(ww)):
                #auxpparts[-1].append([numpy.zeros((2**l,2**l,4,sshape[0],sshape[1]),dtype=c_int)])
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
                        pparts[-1][0][0,0,0,:,:],#auxpparts[0][(dy+initr)*(2*initr+1)+(dx+initr),0,0,0,:,:],
                        pparts[-1][0][0,0,1,:,:],#auxpparts[0][(dy+initr)*(2*initr+1)+(dx+initr),0,0,1,:,:],
                        0,0,
                        nelem)
                    csamples=csamples[:,:,:]*2+1
                    self.scanRCFLPartBU(model,csamples,pparts[-1],ct.ptr[(dy+initr)*(2*initr+1)+(dx+initr)],pres[i-self.starti][(dy+initr)*(2*initr+1)+(dx+initr),:,:],i-self.interv,1,0,0,ratio,usemrf,prec,pady,padx) 
            res[i-self.starti]=pres[i-self.starti].max(0)
            el=pres[i-self.starti].argmax(0)
            pparts[-1][0][0,0,0,:,:]=el/(initr*2+1)-1#auxpparts[0][elx,aa[0],aa[1],aa[2],aa[3],aa[4]]
            pparts[-1][0][0,0,1,:,:]=el%(initr*2+1)-1
            #for l in range(1,len(ww)):
            for l in range(1,len(ww)):
                elx=numpy.tile(el,(2**l,2**l,4,1,1))
                for pt in range((2*initr+1)*(2*initr+1)):
                    if len(ct.ptr[pt].best)>=l:
                        #l=1
                        pparts[-1][l][elx==pt]=ct.ptr[pt].best[l-1][elx==pt]#auxpparts[pt,elx==pt]
            res[i-self.starti]-=rho
        return res,pparts

    def scanRCFLPartBU(self,model,samples,pparts,ct,res,i,lev,locy,locx,ratio,usemrf,prec,pady,padx):
        #if len(model["ww"])<=lev:
        #    return
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
        #ct.best=numpy.zeros((2**lev,2**lev,4,res.shape[0],res.shape[1]),dtype=c_int)
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
        #print "Level %d"%lev,"i=",i
        #print "Position: ",((locy/2)*2+(locx/2))*4,((locy/2)*2+(locx/2)+1)*4
        #print "Size of prec",auxprec.shape
        #substituted scandef with scnadef2 but not tested!!!!
        ff.scanDef2(ww1,ww2,ww3,ww4,fy,fx,ww1.shape[2],df1,df2,df3,df4,self.hog[i],self.hog[i].shape[0],self.hog[i].shape[1],samples[0,:,:],samples[1,:,:],parts,auxres,ratio,samples.shape[1]*samples.shape[2],usemrf,pres,1,auxprec.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),pady,padx)#,ctypes.POINTER(c_float)())#auxprec.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        res+=auxres
        #ct.parts=parts
        ct.best=[parts]
        #if i==5 and lev==2:
        #    global ccc
        #    ccc=ccc+1
        #    print lev,ccc
        #    print parts[0,1,0,2:5,3:8]
        #    print parts[0,1,1,2:5,3:8]
            #raw_input()

        if i-self.interv>=0 and len(model["ww"])-1>lev:
            
            #d1=ct.obj.shape[3];d2=ct.obj.shape[4];d3=ct.obj.shape[5];
            #d4=ct.obj.shape[6];d5=ct.obj.shape[7];
            #aa=numpy.mgrid[:d1,:d2,:d3,:d4,:d5]
            for l in range(len(ct.ptr[0,0].best)):
                aux=numpy.zeros((ct.ptr[0,0].best[l].shape[0]*2,ct.ptr[0,0].best[l].shape[1]*2,4,res.shape[0],res.shape[1]))
                ct.best.append(aux)
                for py in range(res.shape[0]):
                    for px in range(res.shape[1]):
                        ps=(parts[0,0,0,py,px]+ratio)*(ratio*2+1)+parts[0,0,1,py,px]+ratio
                        ct.best[-1][locy+0:locy+0+2,locx+0:locx+0+2,:,py,px]=auxparts[0,ps][:,:,:,py,px]#ct.obj[0,ps][:,:,:,py,px]
                        ps=(parts[0,1,0,py,px]+ratio)*(ratio*2+1)+parts[0,1,1,py,px]+ratio
                        ct.best[-1][locy+0:locy+0+2,locx+2:locx+2+2,:,py,px]=auxparts[1,ps][:,:,:,py,px]
                        ps=(parts[1,0,0,py,px]+ratio)*(ratio*2+1)+parts[1,0,1,py,px]+ratio
                        ct.best[-1][locy+2:locy+2+2,locx+0:locx+0+2,:,py,px]=auxparts[2,ps][:,:,:,py,px]
                        ps=(parts[1,1,0,py,px]+ratio)*(ratio*2+1)+parts[1,1,1,py,px]+ratio
                        ct.best[-1][locy+2:locy+2+2,locx+2:locx+2+2,:,py,px]=auxparts[3,ps][:,:,:,py,px]
        return parts


class Treat:
    def __init__(self,f,scr,pos,sample,fy,fx,occl=False):
        self.pos=pos
        self.scr=scr
        self.f=f
        self.interv=f.interv
        self.sbin=f.sbin
        self.fy=fy
        self.fx=fx
        self.scale=f.scale#2**f.hallucinate*2**(-numpy.arange(len(f.hog))/float(f.interv))
        self.sample=sample
        self.occl=occl

    def showBBox(self,allgtbbox,colors=["w","g"],new_alpha=0.15):
        for item in allgtbbox:
            bbox=item["bbox"]
            pylab.fill([bbox[1],bbox[1],bbox[3],bbox[3],bbox[1]],[bbox[0],bbox[2],bbox[2],bbox[0],bbox[0]],colors[0], alpha=new_alpha, edgecolor=colors[0],lw=2)

    def doall(self,thr=-1,rank=1000,refine=True,occl=False,cluster=0.5,rawdet=False,show=False,inclusion=False,cl=0):
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
        if occl:
            self.det=self.occl(self.det)
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
   
    def doall_debug(self,thr=-1,rank=1000,refine=True,cluster=0.5,rawdet=False,show=False,cl=0):
        self.det=self.select(thr,cl=cl)        
        self.det1=self.rank(self.det,rank) 
        if refine:
            self.det2=self.refine(self.det1)
        else:  
            self.det2=self.det1[:]
        self.det3=self.bbox(self.det2)
        #import time
        #t=time.time()
        if cluster>0:
            self.det4=self.cluster(self.det3,ovr=cluster)
        else:
            self.det4=self.det3[:]
        #print time.time()-t
        if rawdet:
            self.det5=self.rawdet(self.det4)
        else:
            self.det5=self.det4[:]
        if show:
            self.show(self.det5)
        if show=="Parts":
            self.show(self.det5,parts=True)
        return self.det5

    def doalltrain(self,gtbbox,thr=-numpy.inf,rank=numpy.inf,refine=True,rawdet=True,show=False,mpos=0,posovr=0.5,numpos=1,numneg=10,minnegovr=0,minnegincl=0,cl=0,emptybb=False):
        t=time.time()
        self.det=self.select_fast(thr,cl=cl)
        #if gtbbox!=[]:# select only over the positive gtbbox
        #    self.det=self.filter_pos(self.det,gtbbox)
        #    pr=self.f.buildPrior(gtbbox,self.fy,self.fx)
        self.det=self.rank_fast(self.det,rank,cl=cl) 
        if refine:
            self.det=self.refine(self.det)
        self.det=self.bbox(self.det)
        #self.show(self.det,colors=["g"],thr=0.01)
        #print [x["scr"] for x in self.det]
        #pylab.show()
        #raw_input()
        self.best,self.worste=self.getbestworste(self.det,gtbbox,numpos=numpos,numneg=numneg,mpos=mpos,posovr=posovr,minnegovr=minnegovr,minnegincl=minnegincl,emptybb=emptybb)
        if rawdet:
            self.best=self.rawdet(self.best)
            self.worste=self.rawdet(self.worste)
        #show="Parts" maybe it produces the increase of memory
        if show:
            self.show(self.best,colors=["b"])
            self.show(self.worste,colors=["r"])
            self.showBBox(gtbbox)
        if show=="Parts":
            self.show(self.best,parts=True)
            self.show(self.worste,parts=True)
        #self.trpos=self.descr(self.best)
        #self.trneg=self.descr(self.worste)
        #print "DoAll Time:",time.time()-t
        #raw_input()
        return self.best,self.worste#,self.trpos,self.trneg

    def doalltrain_debug(self,gtbbox,thr=-numpy.inf,rank=numpy.inf,refine=True,rawdet=True,show=False,mpos=0,minnegovr=0,minnegincl=0,cl=0):
        self.det=self.select(thr,cl=cl)        
        #import time
        #t=time.time()
        self.det1=self.rank(self.det,rank) 
        #print time.time()-t
        #self.det1=self.det
        if refine:
            self.det2=self.refine(self.det1)
        else:  
            self.det2=self.det1[:]
        self.det3=self.bbox(self.det2)
        #import time
        #t=time.time()
        self.best1,self.worste1=self.getbestworste(self.det3,gtbbox,mpos=mpos)
        #print time.time()-t
        #self.worste1=self.getworste(self.det3,gtbbox)
        if rawdet:
            self.best2=self.rawdet(self.best1)
            self.worste2=self.rawdet(self.worste1)
        else:
            self.best2=self.best1[:]
            self.worste2=self.worste1[:]
        if show:
            self.show(self.best2,colors=["b"])
            self.show(self.worste2,colors=["r"])
        if show=="Parts":
            self.show(self.best2,parts=True)
            self.show(self.worste2,parts=True)
        self.trpos=self.descr(self.best2)
        self.trneg=self.descr(self.worste2)
        return self.best2,self.worste2,self.trpos,self.trneg


    def select(self,thr=0,cl=0,dense=DENSE):
        """
        for each response higher than the threshold execute doitobj.doit
        """
        det=[]
        pylab.ioff()
        initr=self.sample
        for i in range(len(self.scr)):
            if len(self.scr)-i<dense:
                initr=0
            cy,cx=numpy.where(self.scr[i]>thr)
            for l in range(len(cy)):
                mcy=(cy[l])*(2*initr+1)-self.fy+1+initr#self.sample
                mcx=(cx[l])*(2*initr+1)-self.fx+1+initr#self.sample
                det.append({"i":i,"py":cy[l],"px":cx[l],"scr":self.scr[i][cy[l],cx[l]],"ry":mcy,"rx":mcx,"scl":self.scale[i+self.f.starti],"fy":self.fy,"fx":self.fx,"cl":cl})
        return det

    def select_fast(self,thr=0,cl=0,dense=DENSE):
        """
        for each response higher than the threshold execute doitobj.doit
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

    def filter_pos(self,detx,gtbbox=[]):
        """
        for each response higher than the threshold execute doitobj.doit
        """
        #gt["bbox"]
        rdet=[]
        det=numpy.array(detx[0])
        cy=numpy.array(detx[1])
        cx=numpy.array(detx[2])
        i=numpy.array(detx[3])
        scale=numpy.array(self.scale)
        scl=numpy.array(scale[i+self.f.starti])
        mcy=(numpy.array(cy)*(2*self.sample+1)-self.fy+1+self.sample).astype(numpy.float)
        mcx=(numpy.array(cx)*(2*self.sample+1)-self.fx+1+self.sample).astype(numpy.float)
        y1=mcy/scl*self.sbin
        x1=mcx/scl*self.sbin
        y2=(mcy+self.fy)/scl*self.sbin
        x2=(mcx+self.fx)/scl*self.sbin
        sel=numpy.zeros(len(det))
        for l in gtbbox:
            bbx=l["bbox"]
            h=(bbx[2]-bbx[0])/2
            w=(bbx[3]-bbx[1])/2
            #bigbb=[bbx[0]-h,bbx[1]-w,bbx[2]+h,bbx[3]+w]
            #sely=numpy.logical_and(y1>bigbb[0],y2<bigbb[2])
            #selx=numpy.logical_and(x1>bigbb[1],x2<bigbb[3])
            #sel=numpy.logical_or(sel,numpy.logical_and(sely,selx))
            sely1=numpy.logical_and(y1>bbx[0]-h,y1<bbx[0]+h)
            sely2=numpy.logical_and(y2>bbx[2]-h,y2<bbx[2]+h)
            selx1=numpy.logical_and(x1>bbx[1]-h,x1<bbx[1]+h)
            selx2=numpy.logical_and(x2>bbx[3]-h,x2<bbx[3]+h)
            sely=numpy.logical_and(sely1,sely2)
            selx=numpy.logical_and(selx1,selx2)
            sel=numpy.logical_or(sel,numpy.logical_and(sely,selx))
        det=det[sel]#.tolist()
        cy=cy[sel]#.tolist()
        cx=cx[sel]#.tolist()
        i=i[sel]#.tolist()
        print "Number bbox before",len(detx[0])
        print "Number bbox after",len(det)
        #raw_input()
        return det,cy,cx,i

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
           rank detections based on score
        """
        rdet=[]
        det=detx[0]
        cy=detx[1]
        cx=detx[2]
        i=detx[3]
        initr=numpy.array(detx[4])
        #initr=self.sample*numpy.ones(len(i))
        #initr[len(self.scr)-i<dense]=0
        pos=numpy.argsort(-numpy.array(det))      
        if maxnum==numpy.inf:
            maxnum=len(rdet)
        mcy=numpy.array(cy)*(2*initr+1)-self.fy+1+initr#+max(self.hy)
        mcx=numpy.array(cx)*(2*initr+1)-self.fx+1+initr#+max(self.hx)
        for l in pos[:maxnum]:
            rdet.append({"i":i[l],"py":cy[l],"px":cx[l],"scr":det[l],"ry":mcy[l],"rx":mcx[l],"scl":self.scale[i[l]+self.f.starti],"fy":self.fy,"fx":self.fx,"cl":cl})
        return rdet

    def refine(self,ldet):
        """
            refine the localization of the object (py,px) based on higher resolutions
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

    def occl(self,det):
        """rescore based on occluded resolutions"""
        for el in det:
            for l in range(len(el["occl"])):
                if el["i"]-l*self.interv<0:
                    el["scr"]+=el["occl"][l]        
        return det

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
                        #myovr=util.overlap(ls["bbox"],cle["bbox"])
                        myovr=util.overlap(ls["bbox"],cle["bbox"])
                    else:   
                        myovr=util.inclusion(cle["bbox"],ls["bbox"])
                        #myovr=util.inclusion(ls["bbox"],cle["bbox"])
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
        rdet=det[:]
        for item in det:
            if item!=[] and not(item.has_key("notfound")):
                i=item["i"];cy=item["ny"];cx=item["nx"];
                fy=self.fy
                fx=self.fx
                item["feat"]=[]
                my=0;mx=0;
                for l in range(len(item["def"]["dy"])):
                    if i+self.f.starti-(l)*self.interv>=0:
                        my=2*my+item["def"]["dy"][l]
                        mx=2*mx+item["def"]["dx"][l]
                        #print my,mx
                        aux=util.getfeat(self.f.hog[i+self.f.starti-(l)*self.interv],cy+my-1,cy+my+fy*2**l-1,cx+mx-1,cx+mx+fx*2**l-1)
                        item["feat"].append(aux)
                        cy=(cy)*2
                        cx=(cx)*2
                    else:
                        item["feat"].append(numpy.zeros((fy*2**l,fx*2**l,31)))
        return rdet


    def show(self,ldet,parts=False,colors=["w","r","g","b"],thr=-numpy.inf,maxnum=numpy.inf):
        count=0
        for item in ldet:
            if item!=[] and not(item.has_key("notfound")):
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
                            pylab.fill([x1,x1,x2,x2,x1],[y1,y2,y2,y1,y1],colors[1+l], alpha=0.15, edgecolor=colors[1+l],lw=2)           
                    pylab.fill([bbox[1],bbox[1],bbox[3],bbox[3],bbox[1]],[bbox[0],bbox[2],bbox[2],bbox[0],bbox[0]],colors[0], alpha=0.15, edgecolor=colors[0],lw=2)
                    pylab.text(bbox[1],bbox[0],"%d %.3f"%(item["cl"],item["scr"]),bbox=dict(facecolor='w', alpha=0.5),fontsize=10)
                count+=1
                if count>maxnum:
                    break
            
    def descr(self,det,flip=False,usemrf=False,usefather=False):   
        ld=[]
        for item in det:
            if item!=[] and not(item.has_key("notfound")):
                d=numpy.array([])
                for l in range(len(item["feat"])):
                    if not(flip):
                        aux=item["feat"][l]
                        #print "No flip",aux.shape
                    else:
                        aux=hogflip(item["feat"][l])
                        #print "Flip",aux.shape
                    d=numpy.concatenate((d,aux.flatten()))
                    if self.occl:
                        if item["i"]-l*self.interv>=0:
                            d=nump.concatenate((d,[0.0]))
                        else:
                            d=nump.concatenate((d,[1.0]))
                ld.append(d.astype(numpy.float32))
            else:
                ld.append([])
        return ld

    def mixture(self,det):   
        ld=[]
        for item in det:
            if item!=[] and not(item.has_key("notfound")):
                ld.append(item["cl"])
            else:
                ld.append([])
        return ld

    def model(self,descr,rho,lev,fsz,fy=[],fx=[],usemrf=False,usefather=False): 
        ww=[]  
        p=0
        occl=[0]*lev
        if fy==[]:
            fy=self.fy
        if fx==[]:
            fx=self.fx
        d=descr
        for l in range(lev):
            dp=(fy*fx)*4**l*fsz
            ww.append((d[p:p+dp].reshape((fy*2**l,fx*2**l,fsz))).astype(numpy.float32))
            p+=dp
            if self.occl:
                occl[l]=d[p]
                p+=1
        m={"ww":ww,"rho":rho,"fy":fy,"fx":fx,"occl":occl}
        return m

    def getbestworste(self,det,gtbbox,numpos=1,numneg=10,posovr=0.5,negovr=0.2,mpos=0,minnegovr=0,minnegincl=0,emptybb=True):
        lpos=[]
        lneg=[]
        lnegfull=False
        for gt in gtbbox:
            lpos.append(gt.copy())
            lpos[-1]["scr"]=-numpy.inf
            lpos[-1]["ovr"]=1.0
        #c=0
        for d in det:
            #print c
            #c+=1
            goodneg=True
            for gt in lpos: 
                ovr=util.overlap(d["bbox"],gt["bbox"])
                incl=util.inclusion(d["bbox"],gt["bbox"])
                #print incl,d["bbox"],gt["bbox"]
                #raw_input()
                if ovr>posovr:
                    if d["scr"]-mpos*(1-ovr)>gt["scr"]-mpos*(1-gt["ovr"]):
                        #print d
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
                #lpos2[-1]["idbbox"]=idbbox
            else:
                if emptybb:
                    lpos2.append({"scr":-10,"bbox":gt["bbox"],"notfound":True})#not detected bbox
        return lpos2,lneg

                

class TreatDef(Treat):

    def refine(self,ldet):
        """
            refine the localization of the object (py,px) based on higher resolutions
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
                aux=self.pos[i][l][:,:,:,cy,cx]#[cy,cx,:,l]
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
        rdet=det[:]
        for item in det:
            if item!=[] and not(item.has_key("notfound")):
                i=item["i"];cy=item["ny"];cx=item["nx"];
                fy=self.fy
                fx=self.fx
                item["feat"]=[]
                mov=numpy.zeros((1,1,2))
                for l in range(len(item["def"]["party"])):
                    sz=2**l
                    aux=numpy.zeros((fy*sz,fx*sz,self.f.hog[i].shape[2]))
                    if i+self.f.starti-(l)*self.interv>=0:
                        for py in range(sz):
                            for px in range(sz):
                                mov[py,px,0]=2*mov[py,px,0]+item["def"]["dy"][l][py,px]
                                mov[py,px,1]=2*mov[py,px,1]+item["def"]["dx"][l][py,px]
                                aux[py*fy:(py+1)*fy,px*fx:(px+1)*fx,:]=util.getfeat(self.f.hog[i+self.f.starti-(l)*self.interv],cy+mov[py,px,0]-1,cy+mov[py,px,0]+fy-1,cx+mov[py,px,1]-1,cx+mov[py,px,1]+fx-1)
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


    def show(self,ldet,parts=False,colors=["w","r","g","b"],thr=-numpy.inf,maxnum=numpy.inf):
        
        count=0
        if parts:
            for item in ldet:
                if item!=[] and not(item.has_key("notfound")):
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
                                    pylab.fill([x1,x1,x2,x2,x1],[y1,y2,y2,y1,y1],colors[1+l], alpha=0.15, edgecolor=colors[1+l],lw=2)                               
                    count+=1
                    if count>maxnum:
                        break
        Treat.show(self,ldet,colors=colors,thr=thr,maxnum=maxnum)        

    def descr(self,det,flip=False,usemrf=True,usefather=True,k=0.3):   
        ld=[]
        for item in det:
            if item!=[] and not(item.has_key("notfound")):
                d=numpy.array([])
                for l in range(len(item["feat"])):
                    if not(flip):
                        d=numpy.concatenate((d,item["feat"][l].flatten()))       
                        #d=numpy.concatenate((d,numpy.zeros((2**l,2**l),dtype=numpy.float32).flatten()))
                        #d=numpy.concatenate((d,numpy.zeros((2**l,2**l),dtype=numpy.float32).flatten()))
                        #d=numpy.concatenate((d,numpy.zeros((2**l,2**l),dtype=numpy.float32).flatten()))
                        #d=numpy.concatenate((d,numpy.zeros((2**l,2**l),dtype=numpy.float32).flatten())) 
                        if l>0: #skip deformations level 0
                            if usefather:
                                d=numpy.concatenate((d, k*k*(item["def"]["dy"][l].flatten()**2)  ))
                                d=numpy.concatenate((d, k*k*(item["def"]["dx"][l].flatten()**2)  ))
                            if usemrf:
                                d=numpy.concatenate((d,k*k*item["def"]["ddy"][l].flatten()))
                                d=numpy.concatenate((d,k*k*item["def"]["ddx"][l].flatten()))
                    else:
                        d=numpy.concatenate((d,hogflip(item["feat"][l]).flatten()))        
                        #d=numpy.concatenate((d,numpy.zeros((2**l,2**l),dtype=numpy.float32).flatten()))
                        #d=numpy.concatenate((d,numpy.zeros((2**l,2**l),dtype=numpy.float32).flatten()))
                        #d=numpy.concatenate((d,numpy.zeros((2**l,2**l),dtype=numpy.float32).flatten()))
                        #d=numpy.concatenate((d,numpy.zeros((2**l,2**l),dtype=numpy.float32).flatten())) 
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
                ld.append(d.astype(numpy.float32))
            else:
                ld.append([])
        return ld

    def model(self,descr,rho,lev,fsz,fy=[],fx=[],mindef=0.001,usemrf=True,usefather=True): 
        ww=[]  
        df=[]
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
                #####df.append(-numpy.abs(-aux.astype(numpy.float32)))
    #            aux[aux>-mindef/4**l]=-mindef/4**l
    #            #if not(usemrf):
    #            #    aux[:,:,2:]=0
    #            #if not(usefather):
    #            #    aux[:,:,:2]=0
                df.append(aux.astype(numpy.float32))
            else:
                df.append(numpy.zeros((2**l,2**l,4),dtype=numpy.float32))
        m={"ww":ww,"rho":rho,"fy":fy,"fx":fx,"df":df}
        return m

import time

def detect(f,m,gtbbox=None,auxdir=".",hallucinate=1,initr=1,ratio=1,deform=False,bottomup=False,usemrf=False,numneg=0,thr=-2,posovr=0.7,minnegincl=0.5,small=True,show=False,cl=0,mythr=-10,nms=0.5,inclusion=False,usefather=True,mpos=1,emptybb=False,useprior=False):
    """Detect objec in images
        used for both test --> gtbbox=None
        and trainig --> gtbbox = list of bounding boxes
    """
    #f=pyrHOG2.pyrHOG(imname,interv=10,savedir=auxdir+"/hog/",notsave=False,hallucinate=hallucinate,cformat=True)
    #print "Scanning"
    if useprior:
        numrank=200
    else:
        numrank=2000
    if gtbbox!=None and gtbbox!=[] and useprior:
        t1=time.time()
        pr=f.buildPrior(gtbbox,m["fy"],m["fx"])
        print "Prior Time:",time.time()-t1
    else:
        pr=None
    t=time.time()        
    f.resetHOG()
    if deform:
        if bottomup:
            scr,pos=f.scanRCFLDefBU(m,initr=initr,ratio=ratio,small=small,usemrf=usemrf)
        else:
            #scr,pos=f.scanRCFLDefThr(m,initr=initr,ratio=ratio,small=small,usemrf=usemrf,mythr=mythr)
##            scr,pos=f.scanRCFLDef(m,initr=initr,ratio=ratio,small=small,usemrf=usemrf)
            scr,pos=f.scanRCFLprDef(m,initr=initr,ratio=ratio,small=small,usemrf=usemrf,pr=pr)
        tr=TreatDef(f,scr,pos,initr,m["fy"],m["fx"])
    else:
##        scr,pos=f.scanRCFL(m,initr=initr,ratio=ratio,small=small)
        scr,pos=f.scanRCFLpr(m,initr=initr,ratio=ratio,small=small,pr=pr)
        tr=Treat(f,scr,pos,initr,m["fy"],m["fx"])
    numhog=f.getHOG()
    #for h in range(18,len(scr)):
        #pylab.figure(200)
        #pylab.clf()
   #     print h
        #print "Pr Min",pr[h+f.starti].min(),"Max",pr[h+f.starti].max()
        #print "SCR Min",scr[h].min(),"Max",scr[h].max()
        #pylab.imshow(scr[h],interpolation="nearest")
        #pylab.figure(201)
        #pylab.clf()
        #pylab.imshow(pr[h+f.starti],interpolation="nearest")
        #pylab.show()
        #raw_input()
    #print "Scan:",time.time()-t    
    #dettime=time.time()-t
    #print "Elapsed Time:",dettime
    #print "Number HOG:",numhog
    #print "Getting Detections"
    #best1,worste1,ipos,ineg=tr.doalltrain(gtbbox,thr=-5,rank=10000,show=show,mpos=10,numpos=1,numneg=5,minnegovr=0.01)        
    if gtbbox==None:
        if show==True:
            showlabel="Parts"
        else:
            showlabel=False
        det=tr.doall(thr=thr,rank=100,refine=True,rawdet=False,cluster=nms,show=False,inclusion=inclusion,cl=cl)#remember to take away inclusion
        #pylab.gca().set_ylim(0,img.shape[0])
        #pylab.gca().set_xlim(0,img.shape[1])
        #pylab.gca().set_ylim(pylab.gca().get_ylim()[::-1])
        dettime=time.time()-t
        #print "Elapsed Time:",dettime
        #print "Number HOG:",numhog
        tr.show(det,parts=showlabel,thr=-1.0,maxnum=5)           
        return tr,det,dettime,numhog
    else:
        best1,worste1=tr.doalltrain(gtbbox,thr=thr,rank=numrank,show=show,mpos=mpos,numpos=1,posovr=posovr,numneg=numneg,minnegovr=0,minnegincl=minnegincl,cl=cl,emptybb=emptybb)        
        if True:#remember to use it in INRIA
            if deform:
                ipos=tr.descr(best1,flip=False,usemrf=usemrf,usefather=usefather)
                iposflip=tr.descr(best1,flip=True,usemrf=usemrf,usefather=usefather)
                ipos=ipos+iposflip
                ineg=tr.descr(worste1,flip=False,usemrf=usemrf,usefather=usefather)
                inegflip=tr.descr(worste1,flip=True,usemrf=usemrf,usefather=usefather)
                ineg=ineg+inegflip
            else:
                ipos=tr.descr(best1,flip=False)
                iposflip=tr.descr(best1,flip=True)
                ipos=ipos+iposflip
                ineg=tr.descr(worste1,flip=False)
                inegflip=tr.descr(worste1,flip=True)
                ineg=ineg+inegflip
        else:
            ipos=[];ineg=[]
        print "Detect:",time.time()-t    
        return tr,best1,worste1,ipos,ineg

