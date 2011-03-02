
import numpy
import pylab
import cPickle
import time
#from svm import *
import scipy.ndimage.filters as flt
from subprocess import call,PIPE
import scipy.misc.pilutil as pil

i32=numpy.int32
part=numpy.dtype([("itr", i32),("oct",i32),("y",i32),("x",i32),("sy",i32),("sx",i32)])

#colors=numpy.array([[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[0.5,0.5,0.5],
#                    [0,0,0.5],[0,0.5,0],[0,0.5,0.5],[0.5,0,0],[0.5,0,0.5],[0.5,0.5,0],[0.3,0.5,0.7],[0.7,0.5,0.3]])
colors=["r","g","c",'m','w']                    
     
   
def myimread(imgname):
    img=None
    if imgname.split(".")[-1]=="png":
        img=pylab.imread(imgname)
    else:
        img=pil.imread(imgname)        
    return img

def gaussian(sigma=0.5, shape=None):
  """
  Gaussian kernel numpy array with given sigma and shape.
  The shape argument defaults to ceil(6*sigma).
  """
  sigma = max(abs(sigma), 1e-10)
  if shape is None:
    shape = max(int(6*sigma+0.5), 1)
  if not isinstance(shape, tuple):
    shape = (shape, shape)
  x = numpy.arange(-(shape[0]-1)/2.0, (shape[0]-1)/2.0+1e-8)
  y = numpy.arange(-(shape[1]-1)/2.0, (shape[1]-1)/2.0+1e-8)
  Kx = numpy.exp(-x**2/(2*sigma**2))
  Ky = numpy.exp(-y**2/(2*sigma**2))
  ans = numpy.outer(Kx, Ky) / (2.0*numpy.pi*sigma**2)
  return ans/sum(sum(ans))
        
                    
def save(filename,obj,prt=2):
    fd=open(filename,"w")
    cPickle.dump(obj,fd,prt)
    fd.close()
    
def savemat(filename,dic):
    import scipy.io.matlab
    fd=open(filename,"w")
    scipy.io.matlab.savemat(filename,dic)
    #cPickle.dump(obj,fd,prt)
    fd.close()

def load(filename):
    fd=open(filename,"r")
    aux= cPickle.load(fd)
    fd.close()
    return aux                    

def callfunc(cfg,name,func,args=[],force=False,verbose=False,ascii=False):
    modname=cfg.tname+"."+name
    try:
        #tries to load the data  
        if force:
            raise IOError
        print "Loading %s"%modname.split(".")[-1]
        results=load(modname)
    except IOError:
        #otherwise compute them
        if force:
            print "Forced Computing %s"%modname.split(".")[-1]
        else:
            print "Load failed... \nComputing %s"%modname.split(".")[-1]
        results=func(*([cfg]+args+[verbose]))
##        pfeat,maxpos=getPosSamples(trPosImages,body[it],imname,interv=cfg.trint,show=False,obin=cfg.obin,add=cfg.add)
##        #posfeat,posfeatloc,imname=collectPosSamplesCrop(trPosImages,cfg.fy,cfg.fx,maxnum=cfg.trpos,show=False,obin=cfg.obin,add=cfg.add,ovr=cfg.trovr,interv=cfg.trint,maxdeep=0,icy=cfg.icy,icx=cfg.icx)
##        nfeat,maxneg=getNegSamples(trNegImages,genstruct[it],interv=cfg.trint,show=False,maxnum=cfg.trneg,obin=cfg.obin,add=cfg.add,maxdeep=1)
##        for id,p in enumerate(pfeat):
##            pfeat[id]=numpy.concatenate((pfeat[id][:,:,:,:maxpos],hog.hogflip(pfeat[sym[it][id]][:,:,:,:maxpos],obin=cfg.obin,add=cfg.add)),3)
##        for id,p in enumerate(nfeat):
##            nfeat[id]=numpy.concatenate((nfeat[id][:,:,:,:maxneg],hog.hogflip(nfeat[sym[it][id]][:,:,:,:maxneg],obin=cfg.obin,add=cfg.add)),3)
        print "Saving %s"%modname.split(".")[-1]
        if ascii:
            save(modname,results,0)
        else:       
            save(modname,results)
    return results

def callfuncmat(cfg,name,func,args=[],force=False,verbose=False):
    import scipy.io.matlab
    modname=cfg.tname+"."+name
    try:
        #tries to load the data  
        if force:
            raise IOError
        print "Loading %s"%modname.split(".")[-1]
        #results=load(modname)
        dic={}
        scipy.io.matlab.loadmat(modname,dic)
        results=[]
        id=0
        cont=True
        while cont:
            try:
                results.append(dic["res%d"%id])
            except:
                cont=False
            id+=1    
    except IOError:
        #otherwise compute them
        if force:
            print "Forced Computing %s"%modname.split(".")[-1]
        else:
            print "Load failed... \nComputing %s"%modname.split(".")[-1]
        results=func(*([cfg]+args+[verbose]))
##        pfeat,maxpos=getPosSamples(trPosImages,body[it],imname,interv=cfg.trint,show=False,obin=cfg.obin,add=cfg.add)
##        #posfeat,posfeatloc,imname=collectPosSamplesCrop(trPosImages,cfg.fy,cfg.fx,maxnum=cfg.trpos,show=False,obin=cfg.obin,add=cfg.add,ovr=cfg.trovr,interv=cfg.trint,maxdeep=0,icy=cfg.icy,icx=cfg.icx)
##        nfeat,maxneg=getNegSamples(trNegImages,genstruct[it],interv=cfg.trint,show=False,maxnum=cfg.trneg,obin=cfg.obin,add=cfg.add,maxdeep=1)
##        for id,p in enumerate(pfeat):
##            pfeat[id]=numpy.concatenate((pfeat[id][:,:,:,:maxpos],hog.hogflip(pfeat[sym[it][id]][:,:,:,:maxpos],obin=cfg.obin,add=cfg.add)),3)
##        for id,p in enumerate(nfeat):
##            nfeat[id]=numpy.concatenate((nfeat[id][:,:,:,:maxneg],hog.hogflip(nfeat[sym[it][id]][:,:,:,:maxneg],obin=cfg.obin,add=cfg.add)),3)
        print "Saving %s"%modname.split(".")[-1]
        #save(modname,results)
        dic={}
        for id,r in enumerate(results):
            dic["res%d"%id]=r
        scipy.io.matlab.savemat(modname,dic,appendmat=False)
    return results


def callproc(cfg,name,func,args=[],force=False,verbose=False):
    modname=cfg.tname+"."+name
    try:
        #tries to load the data  
        if force:
            raise IOError
        print "Loading %s"%modname.split(".")[-1]
        fl=open(modname,"r")
        fl.close()
        return True
        #results=load(modname)
    except IOError:
        #otherwise compute them
        if force:
            print "Forced Computing %s"%modname.split(".")[-1]
        else:
            print "Load failed... \nComputing %s"%modname.split(".")[-1]
        func(*([cfg,modname]+args+[verbose]))
##        pfeat,maxpos=getPosSamples(trPosImages,body[it],imname,interv=cfg.trint,show=False,obin=cfg.obin,add=cfg.add)
##        #posfeat,posfeatloc,imname=collectPosSamplesCrop(trPosImages,cfg.fy,cfg.fx,maxnum=cfg.trpos,show=False,obin=cfg.obin,add=cfg.add,ovr=cfg.trovr,interv=cfg.trint,maxdeep=0,icy=cfg.icy,icx=cfg.icx)
##        nfeat,maxneg=getNegSamples(trNegImages,genstruct[it],interv=cfg.trint,show=False,maxnum=cfg.trneg,obin=cfg.obin,add=cfg.add,maxdeep=1)
##        for id,p in enumerate(pfeat):
##            pfeat[id]=numpy.concatenate((pfeat[id][:,:,:,:maxpos],hog.hogflip(pfeat[sym[it][id]][:,:,:,:maxpos],obin=cfg.obin,add=cfg.add)),3)
##        for id,p in enumerate(nfeat):
##            nfeat[id]=numpy.concatenate((nfeat[id][:,:,:,:maxneg],hog.hogflip(nfeat[sym[it][id]][:,:,:,:maxneg],obin=cfg.obin,add=cfg.add)),3)
        #print "Saving %s"%modname.split(".")[-1]
        #save(modname,results)
    #return results
        return False


def showresult(cfg,name,func,args=[],force=False,verbose=False):
    modname=cfg.tname+"."+name
    try:
        #tries to load the data  
        namefig=modname+".png"
        if force:
            raise IOError
        print "Loading %s"%modname.split(".")[-1]
        #results=load(modname)
        graph=pil.imread(namefig)
        pylab.figure()
        pylab.imshow(graph)
        pylab.show()
        pylab.draw()
    except IOError:
        #otherwise compute them
        print "Load failed... \nComputing %s"%modname.split(".")[-1]
        func(*([cfg]+args+[verbose]))
##        pfeat,maxpos=getPosSamples(trPosImages,body[it],imname,interv=cfg.trint,show=False,obin=cfg.obin,add=cfg.add)
##        #posfeat,posfeatloc,imname=collectPosSamplesCrop(trPosImages,cfg.fy,cfg.fx,maxnum=cfg.trpos,show=False,obin=cfg.obin,add=cfg.add,ovr=cfg.trovr,interv=cfg.trint,maxdeep=0,icy=cfg.icy,icx=cfg.icx)
##        nfeat,maxneg=getNegSamples(trNegImages,genstruct[it],interv=cfg.trint,show=False,maxnum=cfg.trneg,obin=cfg.obin,add=cfg.add,maxdeep=1)
##        for id,p in enumerate(pfeat):
##            pfeat[id]=numpy.concatenate((pfeat[id][:,:,:,:maxpos],hog.hogflip(pfeat[sym[it][id]][:,:,:,:maxpos],obin=cfg.obin,add=cfg.add)),3)
##        for id,p in enumerate(nfeat):
##            nfeat[id]=numpy.concatenate((nfeat[id][:,:,:,:maxneg],hog.hogflip(nfeat[sym[it][id]][:,:,:,:maxneg],obin=cfg.obin,add=cfg.add)),3)
        #print "Saving %s"%modname.split(".")[-1]
        #save(modname,results)
        pylab.savefig(namefig)
#    return results

def maxdimy(struct):
    return struct["sy"].max()+struct["y"].max()-struct["y"].min()  

def maxdimx(struct):
    return struct["sx"].max()+struct["x"].max()-struct["x"].min()  

ref=0

def pdone(val,tot,t=0.5):
    global ref
    if time.time()-ref>t:
        print int(100*float(val)/tot),"%"
        ref=time.time()

#import scipy.signal

def select(a,y1,y2,x1,x2):
    ay=a.shape[0];ax=a.shape[1]
    #my1=y1;my2=y2;mx1=x1;mx2=x2
    my1=max(0,y1)
    my2=min(ay,y2)
    #my2=min(ay,max(y2,y1))
    mx1=max(0,x1)
    mx2=min(ax,x2)
    #mx2=min(ax,max(x2,x1))
    b=numpy.zeros((y2-y1,x2-x1))
    if (my2-my1)>=0 and (mx2-mx1)>=0:#
        b[my1-y1:my2-my1+my1-y1,mx1-x1:mx2-mx1+mx1-x1]=a[my1:my2,mx1:mx2]
    return b

def select3d(a,y1,y2,x1,x2):
    ay=a.shape[0];ax=a.shape[1]
    #my1=y1;my2=y2;mx1=x1;mx2=x2
    my1=max(0,y1)
    my2=min(ay,y2)
    mx1=max(0,x1)
    mx2=min(ax,x2)
    b=numpy.zeros((y2-y1,x2-x1,a.shape[2]))
    b[my1-y1:my2-my1+my1-y1,mx1-x1:mx2-mx1+mx1-x1]=a[my1:my2,mx1:mx2]
    return b
 
def build(a,posy,posx,dimy,dimx):
    b=numpy.zeros((dimy,dimx))
    b[posy:posy+a.shape[0],posx:posx+a.shape[1]]=a
    return b

def mycorr(a,b,posy,posx,dimy,dimx):
    movy=dimy-2*posy;movx=dimx-2*posx
    ay=a.shape[0];ax=a.shape[1]
    by=b.shape[0];bx=b.shape[1]
    #c=scipy.signal.correlate2d(a,b,mode="full")
    b1=numpy.zeros((by+ay/2*2,bx+ax/2*2))
    b1[ay/2:ay/2+by,ax/2:ax/2+bx]=b
    c=flt.correlate(b1,a,mode="constant")
    #d=select(c,(ay-movy)/2+1,(ay-movy)/2+by+1,(ax-movx)/2+1,(ax-movx)/2+bx+1)
    d=select(c,ay/2+(ay-movy)/2,ay/2+(ay-movy)/2+by,ax/2+(ax-movx)/2,ax/2+(ax-movx)/2+bx)
    return d

def overlap(rect1,rect2):
    """
        Calculate the overlapping percentage between two rectangles
    """
    dy1=abs(rect1[0]-rect1[2])+1
    dx1=abs(rect1[1]-rect1[3])+1
    dy2=abs(rect2[0]-rect2[2])+1
    dx2=abs(rect2[1]-rect2[3])+1
    a1=dx1*dy1
    a2=dx2*dy2
    ia=0
    if rect1[2]>rect2[0] and rect2[2]>rect1[0] and rect1[3]>rect2[1] and rect2[3]>rect1[1]:
    #    py=numpy.sort(numpy.array([rect1[0],rect1[2],rect2[0],rect2[2]]))[1:3]
    #    px=numpy.sort(numpy.array([rect1[1],rect1[3],rect2[1],rect2[3]]))[1:3]
    #    ia=abs(py[1]-py[0])*abs(px[1]-px[0])
        xx1 = max(rect1[1], rect2[1]);
        yy1 = max(rect1[0], rect2[0]);
        xx2 = min(rect1[3], rect2[3]);
        yy2 = min(rect1[2], rect2[2]);
        ia=(xx2-xx1+1)*(yy2-yy1+1)
    return ia/float(a1+a2-ia)

def inclusion(rect1,rect2):
    """
        Calculate the intersection percentage between two rectangles
        Note that it is not anymore symmetric
    """
    dy1=abs(rect1[0]-rect1[2])+1
    dx1=abs(rect1[1]-rect1[3])+1
    dy2=abs(rect2[0]-rect2[2])+1
    dx2=abs(rect2[1]-rect2[3])+1
    a1=dx1*dy1
    a2=dx2*dy2
    ia=0
    if rect1[2]>rect2[0] and rect2[2]>rect1[0] and rect1[3]>rect2[1] and rect2[3]>rect1[1]:
    #    py=numpy.sort(numpy.array([rect1[0],rect1[2],rect2[0],rect2[2]]))[1:3]
    #    px=numpy.sort(numpy.array([rect1[1],rect1[3],rect2[1],rect2[3]]))[1:3]
    #    ia=abs(py[1]-py[0])*abs(px[1]-px[0])
        xx1 = max(rect1[1], rect2[1]);
        yy1 = max(rect1[0], rect2[0]);
        xx2 = min(rect1[3], rect2[3]);
        yy2 = min(rect1[2], rect2[2]);
        ia=(xx2-xx1+1)*(yy2-yy1+1)
    #fsd
    return ia/float(a1)#ia/float(min(a1,a2))#float(a1)


def overlap_array(a,b):
    """
        Calculate the overlapping percentage between two rectangles
    """
    y1=max(a[:,0],b[0])
    x1=max(a[:,1],b[1])
    y2=min(a[:,2],b[2])
    x2=min(a[:,3],b[3])
    w=x2-x1+1
    h=y2-y1+1
    inter = w * h
    areaa = (a[:,2]-a[:,0]+1) * (a[:,3]-a[:,1]+1)
    areab = (b[2]-b[0]+1) * (b[3]-b[1]+1)
    o = inter / float(areaa+areab-inter)
    o[w<0]=0
    o[h<0]=0
    return abs(o)


def box(p1y,p1x,p2y,p2x,col='b',lw=1):
    """
        plot a bbox with the given coordinates
    """
    pylab.plot([p1x,p1x,p2x,p2x,p1x],[p1y,p2y,p2y,p1y,p1y],col,lw=lw)

def getfeat(a,y1,y2,x1,x2):
    """
        returns the hog features at the given position and 
        zeros in case the coordiantes are outside the borders
        """
    dimy=a.shape[0]
    dimx=a.shape[1]
    py1=y1;py2=y2;px1=x1;px2=x2
    dy1=0;dy2=0;dx1=0;dx2=0
    b=numpy.zeros((abs(y2-y1),abs(x2-x1),a.shape[2]))
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
    #print dy1,dy2,py1
    #print dy1,dy2,py2
    if numpy.array(a[py1:py2,px1:px2].shape).min()==0 or numpy.array(b[dy1:y2-y1-dy2,dx1:x2-x1-dx2].shape).min()==0:
        #print "Error in the requested shape: returning all zeros!!"
        #print "Warning: features totally out of the borders!"
        pass
    else:
        b[dy1:y2-y1-dy2,dx1:x2-x1-dx2]=a[py1:py2,px1:px2]
    return b

def getnextlev(i,y1,y2,x1,x2,lev=1,interv=10):
    return numpy.array((i-lev*interv,y1*2**lev+2**lev-1,y1*2**lev+2**lev-1+(y2-y1)*2**lev,x1*2**lev+2**lev-1,x1*2**lev+2**lev-1+(x2-x1)*2**lev))

def getfeatpyr(fp,i,y1,y2,x1,x2,lev=3):
    """
        returns a pyramid hog features at the given position and 
        zeros in case the coordiantes are outside the borders
        Error codes:
        0:no error -1:totally out of scale -2:totally out of borders
        1:partially out of scale 2:partially out of borders
    """
    error=0
    pyr=[]
    if i<0 or i>len(fp.hog):
        return pyr,-1
    if (y1<0 and y2<0) or (x1<0 and x2<0) or (y1>fp.hog[i].shape[0] and y2>fp.hog[i].shape[0]) or (x1>fp.hog[i].shape[1] and x2>fp.hog[i].shape[1]):
        return pyr,-2
    for l in range(lev):
        b=numpy.zeros((abs(y2-y1)*2**l,abs(x2-x1)*2**l,fp.hog[0].shape[2]))
        if (i-l*fp.interv)<0:
            pyr.append(b)
            error+=1
        else:
            a=fp.hog[i-l*fp.interv]
            dimy=a.shape[0]
            dimx=a.shape[1]
            my1=2**l*y1+2**l-1;my2=2**l*y2+2**l-1;mx1=2**l*x1+2**l-1;mx2=2**l*x2+2**l-1
            py1=my1;py2=my2;px1=mx1;px2=mx2
            dy1=0;dy2=0;dx1=0;dx2=0
            if py1<0:
                py1=0
                dy1=py1-my1
                error=2
            if py2>=dimy:
                py2=dimy
                dy2=my2-py2
                error=2
            if px1<0:
                px1=0
                dx1=px1-mx1
                error=2
            if px2>=dimx:
                px2=dimx
                dx2=mx2-px2
                error=2
        #print dy1,dy2,py1
        #print dy1,dy2,py2
        #if numpy.array(a[py1:py2,px1:px2].shape).min()==0 or numpy.array(b[dy1:y2-y1-dy2,dx1:x2-x1-dx2].shape).min()==0:
            #print "Error in the requested shape: returning all zeros!!"
        #    print "Warning: features totally out of the borders!"
        #else:
            b[dy1:my2-my1-dy2,dx1:mx2-mx1-dx2]=a[py1:py2,px1:px2]
            pyr.append(b)
    return pyr,error

def getfeatpyrPlus(fp,i,y1,y2,x1,x2,dy,dx,lev=3):
    """
        returns a pyramid hog features at the given position and 
        zeros in case the coordiantes are outside the borders
        Error codes:
        0:no error -1:totally out of scale -2:totally out of borders
        1:partially out of scale 2:partially out of borders
    """
    error=0
    pyr=[]
    vdy=0
    vdx=0
    if i<0 or i>len(fp.hog):
        return pyr,-1
    if (y1<0 and y2<0) or (x1<0 and x2<0) or (y1>fp.hog[i].shape[0] and y2>fp.hog[i].shape[0]) or (x1>fp.hog[i].shape[1] and x2>fp.hog[i].shape[1]):
        return pyr,-2
    for l in range(lev):
        b=numpy.zeros((abs(y2-y1)*2**l,abs(x2-x1)*2**l,fp.hog[0].shape[2]))
        vdy+=dy[l]
        vdx+=dx[l]
        if (i-l*fp.interv)<0:
            pyr.append(b)
            error+=1
        else:
            a=fp.hog[i-l*fp.interv]
            dimy=a.shape[0]
            dimx=a.shape[1]
            my1=2**l*(y1)+2**l-1+vdy;my2=2**l*(y2)+2**l-1+vdy;mx1=2**l*(x1)+2**l-1+vdx;mx2=2**l*(x2)+2**l-1+vdx
            py1=my1;py2=my2;px1=mx1;px2=mx2
            dy1=0;dy2=0;dx1=0;dx2=0
            if py1<0:
                py1=0
                dy1=py1-my1
                error=2
            if py2>=dimy:
                py2=dimy
                dy2=my2-py2
                error=2
            if px1<0:
                px1=0
                dx1=px1-mx1
                error=2
            if px2>=dimx:
                px2=dimx
                dx2=mx2-px2
                error=2
        #print dy1,dy2,py1
        #print dy1,dy2,py2
        #if numpy.array(a[py1:py2,px1:px2].shape).min()==0 or numpy.array(b[dy1:y2-y1-dy2,dx1:x2-x1-dx2].shape).min()==0:
            #print "Error in the requested shape: returning all zeros!!"
        #    print "Warning: features totally out of the borders!"
        #else:
            b[dy1:my2-my1-dy2,dx1:mx2-mx1-dx2]=a[py1:py2,px1:px2]
            pyr.append(b)
            vdy+=dy[l]
            vdx+=dx[l]
    return pyr,error


def getfeatPyrAdj(fp,i,y1,y2,x1,x2,ww,rho,lev=3,adjfrom=1):
    """
        returns a pyramid hog features at the given position and 
        zeros in case the coordiantes are outside the borders
        Error codes:
        0:no error -1:totally out of scale -2:totally out of borders
        1:partially out of scale 2:partially out of borders
    """
    d=numpy.array([[0,0],[-1,-1],[-1,0],[-1,1],[0,-1],[0,1],[1,-1],[1,0],[1,1]])
    error=0
    pyr=[]
    deform=[]
    if i<0 or i>=len(fp.hog):
        return pyr,deform,0,-1
    if (y1<=0 and y2<=0) or (x1<=0 and x2<=0) or (y1>=fp.hog[i].shape[0] and y2>=fp.hog[i].shape[0]) or (x1>=fp.hog[i].shape[1] and x2>=fp.hog[i].shape[1]):
        return pyr,deform,0,-2
    acc=numpy.zeros(2)
    totval=0
    for l in range(lev):
        b=numpy.zeros((abs(y2-y1)*2**l,abs(x2-x1)*2**l,fp.hog[0].shape[2]))
        if (i-l*fp.interv)<0:
            pyr.append(b)
            deform.append(d[4,:])
            error+=1
        else:
            #deform.append(numpy.zeros((2**l,2)))
            maxval=-10
            maxb=numpy.array([])
            dpos=-1
            #for sub in range(deform.shape[0]):
            npos=d.shape[0]
            if l<adjfrom:
                npos=1
            for dp in range(npos):
                #for dy in range(3):
                b=numpy.zeros((abs(y2-y1)*2**l,abs(x2-x1)*2**l,fp.hog[0].shape[2]))
                a=fp.hog[i-l*fp.interv]
                dimy=a.shape[0]
                dimx=a.shape[1]
                my1=2**l*y1+2**l-1+d[dp,0]+acc[0];my2=2**l*y2+2**l-1+d[dp,0]+acc[0];mx1=2**l*x1+2**l-1+d[dp,1]+acc[1];mx2=2**l*x2+2**l-1+d[dp,1]+acc[1]
                py1=my1;py2=my2;px1=mx1;px2=mx2
                dy1=0;dy2=0;dx1=0;dx2=0
                if py1<0:
                    py1=0
                    dy1=py1-my1
                    error=2
                if py2>=dimy:
                    py2=dimy
                    dy2=my2-py2
                    error=2
                if px1<0:
                    px1=0
                    dx1=px1-mx1
                    error=2
                if px2>=dimx:
                    px2=dimx
                    dx2=mx2-px2
                    error=2
                b[dy1:my2-my1-dy2,dx1:mx2-mx1-dx2]=a[py1:py2,px1:px2]
                val=numpy.sum(b*ww[0][l])
                #print d[dp,:],maxval,val
                if val>maxval:
                    maxval=val
                    dpos=dp
                    maxb=b.copy()
#            if l<adjfrom:
#                dpos=0
            deform.append(d[dpos,:])
            pyr.append(maxb)
            #acc=numpy.zeros(())
            acc+=d[dpos,:]*2
            totval+=maxval
            #print totval
    #print maxval
    return pyr,deform,totval-rho,error


from numpy import ctypeslib
from ctypes import c_int,c_double
#libmrf=ctypeslib.load_library("libmyrmf.so.1.0.1","")
#libmrf.mymrf.argtypes=[ctypeslib.ndpointer(c_double),ctypeslib.ndpointer(c_double),ctypeslib.ndpointer(c_double),ctypeslib.ndpointer(c_double),ctypeslib.ndpointer(c_double),ctypeslib.ndpointer(c_double),ctypeslib.ndpointer(c_int),c_int,c_int,c_int,c_double]
#libmrf.mymrf.restype=c_double
d=numpy.array([[0,0],[-1,-1],[-1,0],[-1,1],[0,-1],[0,1],[1,-1],[1,0],[1,1]])

def get4best(fp,i,y1,y2,x1,x2,wwl,ww,rho,deep,feat,featl,deform,deforml,hCue=[],vCue=[],hCue2=[],vCue2=[],k=10.0):
    #compute mrf
    print y1,y2,x1,x2
    if hCue==[]:
        lhCue=numpy.ones((2,2))
        lhCue[:,1]=0
    else:
        lhCue=numpy.array([[hCue[0][0],0.0],[hCue[0][1],0.0]])   
        #hCue[1,:]=0
    if vCue==[]:
        lvCue=numpy.ones((2,2))
        lvCue[:,1]=0
        #vCue[1,:]=0
    else:
        lvCue=numpy.array([[vCue[0][0],0.0],[vCue[0][1],0.0]])
    if hCue2==[]:
        lhCue2=numpy.ones((2,2))
        lhCue2[:,1]=0
    else:
        lhCue2=numpy.array([[hCue2[0][0],hCue2[0][1]],[0.0,0.0]])   
        #hCue[1,:]=0
    if vCue2==[]:
        lvCue2=numpy.ones((2,2))
        lvCue2[:,1]=0
        #vCue[1,:]=0
    else:
        lvCue2=numpy.array([[vCue2[0][0],vCue2[0][1]],[0.0,0.0]])
    dy=y2-y1
    dx=x2-x1
    D=numpy.zeros((2,2,d.shape[0]),dtype=c_double)
    V=numpy.zeros((2,2,d.shape[0]),dtype=c_double)
##    pylab.figure(100)
##    pylab.imshow(ww[:,:,10])
##    pylab.show()
    for py in range(2):
        for px in range(2):
            for l in range(d.shape[0]):
                D[py,px,l]=numpy.sum(ww[py*dy/2:(py+1)*dy/2,px*dx/2:(px+1)*dx/2,:]*getfeat(fp.hog[i],y1+py*dy/2+d[l,0],y1+(py+1)*dy/2+d[l,0],x1+px*dx/2+d[l,1],x1+(px+1)*dx/2+d[l,1]))
##            pylab.figure(101)
##            pylab.clf()
##            pylab.imshow(ww[py*dy/2:(py+1)*dy/2,px*dx/2:(px+1)*dx/2,10])
##            pylab.show()  
##            raw_input()              
    res=numpy.zeros((2,2),dtype=c_int)
    maxval=libmrf.mymrf(-D,V,lhCue,lvCue,lhCue2,lvCue2,res,2,2,d.shape[0],k)
    #deform.append([])
    #print res
    val=0
    bestfeat=numpy.zeros((dy,dx,ww.shape[2]))
    for py in range(2):
        for px in range(2):
            val+=D[py,px,res[py,px]]
            bestfeat[py*dy/2:(py+1)*dy/2,px*dx/2:(px+1)*dx/2,:]=getfeat(fp.hog[i],y1+py*dy/2+d[res[py,px],0],y1+(py+1)*dy/2+d[res[py,px],0],x1+px*dx/2+d[res[py,px],1],x1+(px+1)*dx/2+d[res[py,px],1])
    feat.append(bestfeat)
    deform.append(res)
    featl.append(bestfeat)
    deforml.append(res)
    score=val
    #print "Compare: ",val,maxval
    #next level
    feat.append([])
    featl.append([])
    deform.append([])
    deforml.append([])
    dfy=2**(deep+1)/2
    dfx=2**(deep+1)/2
    newfeat=numpy.zeros((2*dy,2*dx,feat[0].shape[2]))
    newdeform=numpy.zeros((2*dfy,2*dfx))
    if deep>0:
        #print deep
        if (i-fp.interv)<0:
            print "Warning: I can not reach the last level!!!"
        else:
            print res
            score+=get4best(fp,i-fp.interv,2*y1+1+2*d[res[0,0],0],2*y2+1-dy+2*d[res[0,0],0],2*x1+1+2*d[res[0,0],1],2*x2+1-dx+2*d[res[0,0],1],wwl[1:],wwl[1][:dy,:dx],rho,deep-1,feat[-1],featl[-1],deform[-1],deforml[-1],hCue=hCue[1][0:],vCue=vCue[1][0:],hCue2=hCue2[1][0:],vCue2=vCue2[1][0:])
            score+=get4best(fp,i-fp.interv,2*y1+1+2*d[res[1,0],0],2*y2+1-dy+2*d[res[1,0],0],2*x1+1+dx+2*d[res[1,0],1],2*x2+1+2*d[res[1,0],1],wwl[1:],wwl[1][:dy,dx:],rho,deep-1,feat[-1],featl[-1],deform[-1],deforml[-1],hCue=hCue[1][2:],vCue=vCue[1][2:],hCue2=hCue2[1][2:],vCue2=vCue2[1][2:])
            score+=get4best(fp,i-fp.interv,2*y1+1+dy+2*d[res[0,1],0],2*y2+1+2*d[res[0,1],0],2*x1+1+2*d[res[0,1],1],2*x2+1-dx+2*d[res[0,1],1],wwl[1:],wwl[1][dy:,:dx],rho,deep-1,feat[-1],featl[-1],deform[-1],deforml[-1],hCue=hCue[1][4:],vCue=vCue[1][4:],hCue2=hCue2[1][4:],vCue2=vCue2[1][4:])
            score+=get4best(fp,i-fp.interv,2*y1+1+dy+2*d[res[1,1],0],2*y2+1+2*d[res[1,1],0],2*x1+1+dx+2*d[res[1,1],1],2*x2+1+2*d[res[1,1],1],wwl[1:],wwl[1][dy:,dx:],rho,deep-1,feat[-1],featl[-1],deform[-1],deforml[-1],hCue=hCue[1][6:],vCue=vCue[1][6:],hCue2=hCue2[1][6:],vCue2=vCue2[1][6:])
##            print "hola",2*y1,2*y2-dy,2*x1,2*x2-dx
##            score+=get4best(fp,i-fp.interv,2*y1,2*y2-dy,2*x1,2*x2-dx,wwl[1:],wwl[1][:dy,:dx],rho,deep-1,feat[-1],featl[-1],deform[-1],deforml[-1])#,hCue=[],vCue=[]
##            score+=get4best(fp,i-fp.interv,2*y1+dy,2*y2,2*x1,2*x2-dx,wwl[1:],wwl[1][dy:,:dx],rho,deep-1,feat[-1],featl[-1],deform[-1],deforml[-1])
##            score+=get4best(fp,i-fp.interv,2*y1,2*y2-dy,2*x1+dx,2*x2,wwl[1:],wwl[1][:dy,dx:],rho,deep-1,feat[-1],featl[-1],deform[-1],deforml[-1])
##            score+=get4best(fp,i-fp.interv,2*y1+dy,2*y2,2*x1+dx,2*x2,wwl[1:],wwl[1][dy:,dx:],rho,deep-1,feat[-1],featl[-1],deform[-1],deforml[-1])
        if feat[-1]!=[]:
            count=0
            for py in range(2):
                for px in range(2):
                    newfeat[py*dy:(py+1)*dy,px*dx:(px+1)*dx,:]=feat[-1][2*count]
                    newdeform[py*dfy:(py+1)*dfy,px*dfx:(px+1)*dfx]=deform[-1][2*count]
                    count+=1
##            if deep==1:
##                pylab.figure(15)
##                pylab.clf()
##                drawHOG2(newfeat);
##                pylab.show()
##                #raw_input()
        del featl[-1]
        del deforml[-1]
        featl.append(newfeat)
        deforml.append(newdeform)
        #print newfeat.shape
        #print newdeform.shape
    return score
        


def getfeatPyrAdjMRF(fp,i,y1,y2,x1,x2,ww,rho,lev=3,adjfrom=1,hCue=[],vCue=[],hCue2=[],vCue2=[]):
    """
        returns a pyramid hog features at the given position and 
        zeros in case the coordiantes are outside the borders
        Error codes:
        0:no error -1:totally out of scale -2:totally out of borders
        1:partially out of scale 2:partially out of borders
    """
    #d=numpy.array([[0,0],[-1,-1],[-1,0],[-1,1],[0,-1],[0,1],[1,-1],[1,0],[1,1]])
    error=0
    pyr=[]
    deform=[]
    if i<0 or i>len(fp.hog):
        return pyr,defrom,0,-1
    if (y1<=0 and y2<=0) or (x1<=0 and x2<=0) or (y1>=fp.hog[i].shape[0] and y2>=fp.hog[i].shape[0]) or (x1>=fp.hog[i].shape[1] and x2>=fp.hog[i].shape[1]):
        return pyr,deform,[],[],0,-2
    acc=numpy.zeros(2)
    totval=0
    for l in range(adjfrom):
        b=numpy.zeros((abs(y2-y1)*2**l,abs(x2-x1)*2**l,fp.hog[0].shape[2]))
        if (i-l*fp.interv)<0:
            pyr.append(b)
            deform.append(d[4,:])
            error+=1
        else:
            #deform.append(numpy.zeros((2**l,2)))
            maxval=-10
            maxb=numpy.array([])
            dpos=-1
            #for sub in range(deform.shape[0]):
            npos=d.shape[0]
            if l<adjfrom:
                npos=1
            for dp in range(npos):
                #for dy in range(3):
                b=numpy.zeros((abs(y2-y1)*2**l,abs(x2-x1)*2**l,fp.hog[0].shape[2]))
                a=fp.hog[i-l*fp.interv]
                dimy=a.shape[0]
                dimx=a.shape[1]
                my1=2**l*y1+2**l-1+d[dp,0]+acc[0];my2=2**l*y2+2**l-1+d[dp,0]+acc[0];mx1=2**l*x1+2**l-1+d[dp,1]+acc[1];mx2=2**l*x2+2**l-1+d[dp,1]+acc[1]
                py1=my1;py2=my2;px1=mx1;px2=mx2
                dy1=0;dy2=0;dx1=0;dx2=0
                if py1<0:
                    py1=0
                    dy1=py1-my1
                    error=2
                if py2>=dimy:
                    py2=dimy
                    dy2=my2-py2
                    error=2
                if px1<0:
                    px1=0
                    dx1=px1-mx1
                    error=2
                if px2>=dimx:
                    px2=dimx
                    dx2=mx2-px2
                    error=2
                b[dy1:my2-my1-dy2,dx1:mx2-mx1-dx2]=a[py1:py2,px1:px2]
                val=numpy.sum(b*ww[0][l])
                #print d[dp,:],maxval,val
                if val>maxval:
                    maxval=val
                    dpos=dp
                    maxb=b.copy()
#            if l<adjfrom:
#                dpos=0
            deform.append(dpos)
            pyr.append(maxb)
            #acc=numpy.zeros(())
            acc+=d[dpos,:]*2
            totval+=maxval
    feat1=pyr;deform1=[numpy.zeros((1,1))]
    #feat2=pyr;deform2=[numpy.zeros((2,2))]
    feat2=pyr[:];deform2=[numpy.zeros((1,1))]
    if len(ww[0])>1:
        totval+=get4best(fp,i-fp.interv,1+2*y1,1+2*y2,1+2*x1,1+2*x2,ww[0][1:],ww[0][1],rho,lev-adjfrom-1,feat1,feat2,deform1,deform2,hCue=hCue,vCue=vCue,hCue2=hCue2,vCue2=vCue2)
    #raw_input()
        if feat2[-1]==[]:
            del feat2[-1]
        if deform2[-1]==[]:
            del deform2[-1]
    return feat1,feat2,deform1,deform2,totval-rho,error

def drawDef(pyr,i,py,px,sy,sx,feat,deform,sbin,lev):
    #for l in range(feat)
    s=sbin*2**(float(i)/pyr.interv)
    dy1=d[deform[0][0,0],0]
    dx1=d[deform[0][0,0],1]
    dy2=d[deform[0][0,1],0]
    dx2=d[deform[0][0,1],1]
    dy3=d[deform[0][1,0],0]
    dx3=d[deform[0][1,0],1]
    dy4=d[deform[0][1,1],0]
    dx4=d[deform[0][1,1],1]
    box((py+dy1)*s,(px+dx1)*s,(py+sy/2+dy1)*s,(px+sx/2+dx1)*s,col=colors[lev],lw=1.2)
    box((py+dy2)*s,(sx/2+px+dx2)*s,(py+sy/2+dy2)*s,(px+sx+dx2)*s,col=colors[lev],lw=1.2)
    box((sy/2+py+dy3)*s,(px+dx3)*s,(py+sy+dy3)*s,(px+sx/2+dx3)*s,col=colors[lev],lw=1.2)
    box((sy/2+py+dy4)*s,(sx/2+px+dx4)*s,(py+sy+dy4)*s,(px+sx+dx4)*s,col=colors[lev],lw=1.2)
    #raw_input()
    if deform[1]!=[]:
        drawDef(pyr,i-pyr.interv,(py+dy1)*2,(px+dx1)*2,sy,sx,feat,deform[1][0:],sbin,lev+1)
        drawDef(pyr,i-pyr.interv,(py+dy2)*2,sx+(px+dx2)*2,sy,sx,feat,deform[1][2:],sbin,lev+1)
        drawDef(pyr,i-pyr.interv,sy+(py+dy3)*2,(px+dx3)*2,sy,sx,feat,deform[1][4:],sbin,lev+1)
        drawDef(pyr,i-pyr.interv,sy+(py+dy4)*2,sx+(px+dx4)*2,sy,sx,feat,deform[1][6:],sbin,lev+1)

##def bulidb(pyr,b,bestdef):
##    dimy=b.shape[0]
##    dimx=b.shape[1]
##    dt=numpy.log2(bestdef.shape[1])
##    maxb=b.copy()
##    #for p in range(bestdef.shape[1]):
##    for py in range(dt):
##        for px in range(dt):
##            pos=px*dt+py
##            maxb[dimy/dt*py:dimy/dt*(py+1),dimx/dt*px:dimx/dt*(px+1)]=getfeat(pyr,bestdef[0,pos],bestdef[1,pos],bestdef[2,pos],bestdef[3,pos])
##    return maxb
##
##def propagate(acc,b,bestdef):
##    newacc=numpy.zeros((acc.shape[0],2,2,acc.shape[1],2))
##    for p in range(acc.shape[0]):
##        for px in range(2):
##            for py in range(2):
##                for parts in range(acc.shape[1]):
##                    newacc[p,py,px,parts,0]=acc[p,parts,0]#+bestdef[]
##                    newacc[p,py,px,parts,1]=acc[p,parts,1]+px
##    newacc=newacc.reshape(acc.shape[0]*2*2,acc.shape[1])
##    return newacc
##
##def getfeatPyrAdjMrf(fp,i,y1,y2,x1,x2,ww,rho,lev=3,adjfrom=1):
##    """
##        returns a pyramid hog features at the given position and 
##        zeros in case the coordiantes are outside the borders
##        Error codes:
##        0:no error -1:totally out of scale -2:totally out of borders
##        1:partially out of scale 2:partially out of borders
##    """
##    from numpy import ctypeslib
##    from ctypes import c_int,c_double
##    libmrf=ctypeslib.load_library("libmyrmf.so.1.0.1","")
##    libmrf.argtypes=[ctypeslib.ndpointer(),ctypeslib.ndpointer(),ctypeslib.ndpointer(),ctypeslib.ndpointer(),ctypeslib.ndpointer(),c_int,c_int,c_int,c_int]
##    libmrf.restype=c_double
##    d=numpy.array([[0,0],[-1,-1],[-1,0],[-1,1],[0,-1],[0,1],[1,-1],[1,0],[1,1]])
##    hCue=numpy.zeros((2,2))
##    vCue=numpy.zeros((2,2))
##    error=0
##    pyr=[]
##    deform=[]
##    if i<0 or i>len(fp.hog):
##        return pyr,defrom,0,-1
##    if (y1<=0 and y2<=0) or (x1<=0 and x2<=0) or (y1>=fp.hog[i].shape[0] and y2>=fp.hog[i].shape[0]) or (x1>=fp.hog[i].shape[1] and x2>=fp.hog[i].shape[1]):
##        return pyr,deform,0,-2
##    acc=numpy.zeros((1,1,2))
##    totval=0
##    for l in range(lev):
##        b=numpy.zeros((abs(y2-y1)*2**l,abs(x2-x1)*2**l,fp.hog[0].shape[2]))
##        dty=b.shape[0]/(2**(l-adjfrom))
##        dtx=b.shape[1]/(2**(l-adjfrom))
##        if (i-l*fp.interv)<0:
##            pyr.append(b)
##            deform.append(d[4,:])
##            error+=1
##        else:
##            #deform.append(numpy.zeros((2**l,2)))
##            #maxb=numpy.array([])
##            #dpos=-1
##            #for sub in range(deform.shape[0]):
##            npos=d.shape[0]
##            if l<adjfrom:
##                npos=1
##            numparts=4**(l-adjfrom)
##            maxval=numpy.zeros(numparts)
##            bestdef=numpy.zeros((4,numparts),dtype=numpy.int)
##            for part in range(numparts):
##                for p in range(4):
##                    D=numpy.zeros((4,d.shape[0]))
##                    for dp in range(npos):
##                        #for dy in range(3):
##                        b=numpy.zeros((abs(y2-y1)*2**l,abs(x2-x1)*2**l,fp.hog[0].shape[2]))
##                        a=fp.hog[i-l*fp.interv]
##                        dimy=a.shape[0]
##                        dimx=a.shape[1]
##                        my1=2**l*y1+2**l-1+d[dp,0]+acc[part,p,0];my2=2**l*y2+2**l-1+d[dp,0]+acc[part,p,0];mx1=2**l*x1+2**l-1+d[dp,1]+acc[part,p,1];mx2=2**l*x2+2**l-1+d[dp,1]+acc[part,p,1]
##                        py1=my1;py2=my2;px1=mx1;px2=mx2
##                        dy1=0;dy2=0;dx1=0;dx2=0
##                        if py1<0:
##                            py1=0
##                            dy1=py1-my1
##                            error=2
##                        if py2>=dimy:
##                            py2=dimy
##                            dy2=my2-py2
##                            error=2
##                        if px1<0:
##                            px1=0
##                            dx1=px1-mx1
##                            error=2
##                        if px2>=dimx:
##                            px2=dimx
##                            dx2=mx2-px2
##                            error=2
##                        b[dy1:my2-my1-dy2,dx1:mx2-mx1-dx2]=a[py1:py2,px1:px2]
##                        D[p,dp]=numpy.sum(b*ww[0][l])
##                maxval[part]=libmrf.myrmf(D,V,hCue,vCue,res,2,2,d.shape[0],1.0)
##                bestdef[:,part]=res
##                #mymrf(double *D, double *V, double *hCue, double *vCue,int *result, int sizeX,int sizeY,int numLabels, double valk)
##                        #print d[dp,:],maxval,val
####                        if val>maxval:
####                            maxval=val
####                            dpos=dp
####                            maxb=b.copy()
###            if l<adjfrom:
###                dpos=0
##            #deform.append(d[dpos,:])
##            maxb=bulidb(D,b,bestdef)
##            deform.append(d[bestdef,:])
##            pyr.append(maxb)
##            #acc=numpy.zeros(())
##            #acc+=d[dpos,:]*2
##            acc=propagate(acc,b,bestdef)
##            totval+=numpy.sum(maxval)
##            #print totval
##    #print maxval
##    return pyr,deform,totval-rho,error

##def getfeatPyrAdjMrf(fp,i,y1,y2,x1,x2,ww,lev=3,adjfrom=1):
##    """
##        returns a pyramid hog features at the given position and 
##        zeros in case the coordiantes are outside the borders
##        Error codes:
##        0:no error -1:totally out of scale -2:totally out of borders
##        1:partially out of scale 2:partially out of borders
##    """
##    from numpy import ctypeslib
##    from ctypes import c_int,c_double
##    libmrf=ctypeslib.load_library("libmyrmf.so.1.0.1","")
##    libmrf.argtypes=[ctypeslib.ndpointer(),ctypeslib.ndpointer(),ctypeslib.ndpointer(),ctypeslib.ndpointer(),ctypeslib.ndpointer(),c_int,c_int,c_int,c_int]
##    libmrf.restype=c_double
##    d=numpy.array([[0,0],[-1,-1],[-1,0],[-1,1],[0,-1],[0,1],[1,-1],[1,0],[1,1]])
##    error=0
##    pyr=[]
##    deform=[]
##    if i<0 or i>len(fp.hog):
##        return pyr,deform,0,-1
##    if (y1<=0 and y2<=0) or (x1<=0 and x2<=0) or (y1>=fp.hog[i].shape[0] and y2>=fp.hog[i].shape[0]) or (x1>=fp.hog[i].shape[1] and x2>=fp.hog[i].shape[1]):
##        return pyr,deform,0,-2
##    acc=numpy.zeros(2)
##    for l in range(lev):
##        b=numpy.zeros((abs(y2-y1)*2**l,abs(x2-x1)*2**l,fp.hog[0].shape[2]))
##        if (i-l*fp.interv)<0:
##            pyr.append(b)
##            deform.append(d[4,:])
##            error+=1
##        else:
##            maxval=-10
##            maxb=numpy.array([])
##            dpos=-1
##            for dp in range(d.shape[0]):
##                #for dy in range(3):
##                a=fp.hog[i-l*fp.interv]
##                dimy=a.shape[0]
##                dimx=a.shape[1]
##                my1=2**l*y1+2**l-1+d[dp,0]+acc[0];my2=2**l*y2+2**l-1+d[dp,0]+acc[0];mx1=2**l*x1+2**l-1+d[dp,1]+acc[1];mx2=2**l*x2+2**l-1+d[dp,1]+acc[1]
##                py1=my1;py2=my2;px1=mx1;px2=mx2
##                dy1=0;dy2=0;dx1=0;dx2=0
##                if py1<0:
##                    py1=0
##                    dy1=py1-my1
##                    error=2
##                if py2>=dimy:
##                    py2=dimy
##                    dy2=my2-py2
##                    error=2
##                if px1<0:
##                    px1=0
##                    dx1=px1-mx1
##                    error=2
##                if px2>=dimx:
##                    px2=dimx
##                    dx2=mx2-px2
##                    error=2
##                b[dy1:my2-my1-dy2,dx1:mx2-mx1-dx2]=a[py1:py2,px1:px2]
##                val=numpy.sum(b*ww[0][l])
##                if val>maxval:
##                    maxval=val
##                    dpos=dp
##                    maxb=b.copy()
##            if l<adjfrom:
##                dpos=4
##            deform.append(d[dpos,:])
##            pyr.append(maxb)
##            acc+=d[dpos,:]*2
##    return pyr,deform,maxval,error


##def getfeatpyr(p,i,y1,y2,x1,x2):
##    if i>len(a):
##        return 
##    getfeat(p[i],y1,y2,x1,x2)
##    return b,0


def upsample(win):
    newin=numpy.array([])
    if len(win.shape)>2:
        newin=numpy.zeros((win.shape[0]*2,win.shape[1]*2,win.shape[2]),dtype=win.dtype)
    else:
        newin=numpy.zeros((win.shape[0]*2,win.shape[1]*2),dtype=win.dtype)
    newin[::2,::2]=win[:,:]
    newin[1::2,:]+=0.5*newin[::2,:]
    newin[1:-2:2,:]+=0.5*newin[2::2,:]
    newin[-1,:]=newin[-2,:]
    newin[:,1::2]+=0.5*newin[:,::2]
    newin[:,1:-2:2]+=0.5*newin[:,2::2]
    newin[:,-1]=newin[:,-2]
    return newin[:,:]

def loadSvm(str,dir="./save/",lib="libsvm"):
    if lib=="libsvm":
        return loadSvmLib(str,dir)
    if lib=="linear":
        return loadSvmLin(str,dir)
    if lib=="linearblock":
        return loadSvmLin(str,dir)
    else:
        return loadSvmLight(str,dir)

def loadSvmLib(str,dir="./save/",skip=0):
    """
        Returns w and rho of the dir+str SVM model file
    """
    #str=fname+"_l%d.dat"%level
    h=open(dir+str)
    lin= h.readlines()
    maxsz=len(lin[8].split())
##    for l in lin:
##        if maxsz<len(l.split()):
##            maxsz=len(l.split())
    w=numpy.zeros(maxsz-1)
    rho=float(lin[4].split()[1])
    #print maxsz

    if skip==0:
        for csv in range(len(lin)-8):
            w1=lin[csv+8].split()
            c=float(w1[0])
            for count in range(1,len(w1)):
                w[count-1]=w[count-1]+c*float(w1[count].split(":")[1])
        #fd=open("./aux"+str,"wb")
        #pickle.dump(w,fd,2)
        #fd.close()
    else:
        fd=open("./aux"+str,"r")
        w=pickle.load(fd)
        fd.close()
    return (w,rho)
    #ww=w.reshape((numfy-3,numfx-3,4*obin))

def loadSvmLin(str,dir="./",bias=100):
    """
        Returns w and rho of the dir+str SVM model file
    """
    #str=fname+"_l%d.dat"%level
    h=open(dir+str)
    lin= h.readlines()
    w=numpy.zeros(len(lin)-7)
    b=-float(lin[-1])*bias
    for id,l in enumerate(lin[6:-1]):
        w[id]=float(l)
    return (w,b)

def loadSvmLinBlk(str,dir="./"):
    """
        Returns w and rho of the dir+str SVM model file
    """
    #str=fname+"_l%d.dat"%level
    h=open(dir+str)
    lin= h.readlines()
    w=numpy.zeros(len(lin)-7)
    b=-float(lin[6])*100
    for id,l in enumerate(lin[7:]):
        w[id]=float(l)
    return (w,b)


def loadSvmLibNolin(libsvmfile):
    """
        Returns w and rho of the dir+str SVM model file
    """

    h=open(libsvmfile)
    lin= h.readlines()
    maxsz=len(lin[8].split())
    numsv=len(lin)-8
    w=numpy.zeros((numsv,maxsz))
    rho=float(lin[4].split()[1])

    for csv in range(numsv):
        w1=lin[csv+8].split()
        w[csv,0]=float(w1[0])
        for count in range(1,len(w1)):
            w[csv,count]=float(w1[count].split(":")[1])

    return (w,rho)


def loadSvmLight(str,dir="./save/",skip=0):
    """
        Returns w and rho of the dir+str SVM model file
    """
    #str=fname+"_l%d.dat"%level
    h=open(dir+str)
    lin= h.readlines()
    maxsz=int(lin[7].split()[0])
##    for l in lin:
##        if maxsz<len(l.split()):
##            maxsz=len(l.split())
    w=numpy.zeros(maxsz)
    rho=float(lin[10].split()[0])
    #print maxsz

    if skip==0:
        for csv in range(len(lin)-11):
            w1=lin[csv+11].split()
            c=float(w1[0])
            for count in range(1,len(w1)-1):
                w[count-1]=w[count-1]+c*float(w1[count].split(":")[1])
        #fd=open("./aux"+str,"wb")
        #pickle.dump(w,fd,2)
        #fd.close()
    else:
        fd=open("./aux"+str,"r")
        w=pickle.load(fd)
        fd.close()
    return (w,rho)
    #ww=w.reshape((numfy-3,numfx-3,4*obin))

##def trainSvmRaw(posnfeat,negnfeat,str,dir="./save/",pc=0.1):
##    """
##        train a linear SVM with C=pc and save the results in dir+str
##    """
##    kernels = [LINEAR, POLY, RBF]
##    kname = ['linear','polynomial','rbf']
##    posntimes=posnfeat.shape[0]
##    negntimes=negnfeat.shape[0]
##    ntimes=posntimes+negntimes
##    possample=posnfeat.tolist()
##    negsample=negnfeat.tolist()
##    labels = numpy.concatenate((numpy.ones(posntimes),-numpy.ones(negntimes)),1).tolist()
##    samples = possample + negsample
##    problem = svm_problem(labels, samples);
##    size = len(samples)
##    print "Starting SVM"
##    param = svm_parameter(C = pc,kernel_type = LINEAR)#,nr_weight = 2,weight_label = [1,0]#,weight = [10,1])
##    model = svm_model(problem,param)
##    model.save(dir+str)
##    return model

def toSvmfile(posnfeat,negnfeat,str):
    """    
        Save the feautres into a files in a SVMlib format
    """    
    fd=open(str,"w")
    for n in range(posnfeat.shape[0]):
        print n
        fd.write("1")
        for d in range(posnfeat.shape[1]):
            fd.write(" %d:%g"%(d+1,posnfeat[n,d]))
        fd.write("\n")
    for n in range(negnfeat.shape[0]):
        print n
        fd.write("-1")
        for d in range(negnfeat.shape[1]):
            fd.write(" %d:%g"%(d+1,negnfeat[n,d]))
        fd.write("\n")
    fd.close()

def toSvmfileFast(posnfeat,negnfeat,str,bias=-1):
    """    
        Save the feautres into a files in a SVMlib format
    """    
    from ctypes import c_float,c_double,c_int,c_char_p
    import string
    print "Writing SVM file (%d pos,%d neg)"%(len(posnfeat),len(negnfeat))
    if type(posnfeat)==list:
        islist=True
    pvec2svm = numpy.ctypeslib.load_library("libvec2svm.so",".") 
    vec2svm = pvec2svm.vec2svm
    vec2svm.restype = None
    vec2svm.argtypes = [c_int,numpy.ctypeslib.ndpointer(dtype=c_float,ndim=1),c_char_p]
    fd=open(str,"w")
    if islist:
        i=len(posnfeat[0])
    else:
        i=posnfeat.shape[1]
    if bias>0:
        i+=1
    s=string.zfill("",i*20)
    #force the data to be contiguos to use vec2svm correctly
    #posnfeat=numpy.array(posnfeat,order="C")
    #negnfeat=numpy.array(negnfeat,order="C")
    for n in range(len(posnfeat)):
        #print n
        fd.write("1")      
        if bias>0:
            aux=numpy.concatenate((posnfeat[n],[numpy.float32(bias)]))
        else:
            aux=posnfeat[n]
        if islist:
            vec2svm(i,aux,s)
        else:
            aux=numpy.ascontiguousarray(aux)
            vec2svm(i,aux,s)
            #fd.write
        fd.write(s[:s.find("\n")+1])
    for n in range(len(negnfeat)):
        #print n
        if bias>0:
            aux=numpy.concatenate((negnfeat[n],[numpy.float32(bias)]))
        else:
            aux=negnfeat[n]
        fd.write("-1")
        if islist:
            vec2svm(i,aux,s)
        else:
            aux=numpy.ascontiguousarray(aux)
            vec2svm(i,aux,s)
        fd.write(s[:s.find("\n")+1])
    fd.close()

def trainSvmRaw(posnfeat,negnfeat,str,dir="./save/",pc=0.1,lib="libsvm",t=0,pr=0,g=1.0,big=False):
    if lib=="libsvm":
        trainSvmRawLib(posnfeat,negnfeat,str,dir,pc,t,pr,g=g,big=big)
    if lib=="svmlight":
        trainSvmRawLight(posnfeat,negnfeat,str,dir,pc)
    if lib=="shogun":
        trainSvmShogun(posnfeat,negnfeat,str,dir,pc)
    if lib=="linear":
        trainSvmRawLin(posnfeat,negnfeat,str,dir,pc)
    if lib=="pegasos":
        trainSvmRawPeg(posnfeat,negnfeat,str,dir,pc)
    if lib=="linearblock":
        trainSvmRawLinBlk(posnfeat,negnfeat,str,dir,pc)

def trainSvmRawLin(posnfeat,negnfeat,str,dir="./save/",pc=0.017,path="/home/marcopede/code/c/liblinear-1.7"):
    """
        The same as trainSVMRaw but it does use files instad of lists:
        it is slower but it needs less memory.
    """
    posntimes=len(posnfeat)
    negntimes=len(negnfeat)
    ntimes=posntimes+negntimes
    labels = numpy.concatenate((numpy.ones(posntimes),-numpy.ones(negntimes)),1)#.tolist()
    toSvmfileFast(posnfeat,negnfeat,dir+str+".data")
    del posnfeat,negnfeat #save memory but be carefull because it is not safe
    print "Starting Linear SVM training"
    import default 
    #cmd=path+"/train -s 3 -c %f -B -1 "%(pc)+dir+str+".data "+dir+str+"\n"
    cmd=path+"/train -s 1 -c %f -B 100 "%(pc)+dir+str+".data "+dir+str+"\n"
    print cmd
    call(cmd, shell = True, stdout = PIPE)
    import os
    os.remove(dir+str+".data")
    return 0

import ctypes
from ctypes import c_float,c_int,c_void_p,POINTER

ctypes.cdll.LoadLibrary("./libfastpegasos.so")
lpeg= ctypes.CDLL("libfastpegasos.so")
#void fast_pegasos(ftype *w,int wx,ftype *ex,int exy,ftype *label,ftype lambda,int iter,int part)
lpeg.fast_pegasos.argtypes=[
    numpy.ctypeslib.ndpointer(dtype=c_float,ndim=1,flags="C_CONTIGUOUS")#w
    ,c_int #sizew
    ,numpy.ctypeslib.ndpointer(dtype=c_float,ndim=2,flags="C_CONTIGUOUS")#samples
    ,c_int #numsamples
    ,numpy.ctypeslib.ndpointer(dtype=c_float,ndim=1,flags="C_CONTIGUOUS")#labels
    ,c_float #lambda
    ,c_int #iter
    ,c_int #part
    ]

maxcomp=10 #max 10 components
lpeg.fast_pegasos_comp.argtypes=[
    numpy.ctypeslib.ndpointer(dtype=c_float,ndim=1,flags="C_CONTIGUOUS")#w
    ,c_int #num comp
    ,POINTER(c_int) #compx
    ,POINTER(c_int) #compy 
    ,POINTER(c_void_p)#samples
    ,c_int #numsamples
    ,numpy.ctypeslib.ndpointer(dtype=c_int,ndim=1,flags="C_CONTIGUOUS")#labels
    ,numpy.ctypeslib.ndpointer(dtype=c_int,ndim=1,flags="C_CONTIGUOUS")#comp number
    ,c_float #lambda
    ,c_int #iter
    ,c_int #part
    ]

#ftype objective(ftype *w,int wx,ftype *ex, int exy,ftype *label,ftype lambda)
lpeg.objective.argtypes=[
    numpy.ctypeslib.ndpointer(dtype=c_float,ndim=1,flags="C_CONTIGUOUS")#w
    ,c_int #sizew
    ,numpy.ctypeslib.ndpointer(dtype=c_float,ndim=2,flags="C_CONTIGUOUS")#samples
    ,c_int #numsamples
    ,numpy.ctypeslib.ndpointer(dtype=c_float,ndim=1,flags="C_CONTIGUOUS")#labels
    ,c_float #lambda
    ]
lpeg.objective.restype=ctypes.c_float


def trainSvmRawPeg(posnfeat,negnfeat,fname,oldw=None,dir="./save/",pc=0.017,path="/home/marcopede/code/c/liblinear-1.7",maxtimes=100,eps=0.01,bias=100):
    """
        The same as trainSVMRaw but it does use files instad of lists:
        it is slower but it needs less memory.
    """
    ff=open(fname,"a")
    posntimes=len(posnfeat)
    negntimes=len(negnfeat)
    ntimes=posntimes+negntimes
    fdim=len(posnfeat[0])+1
    bigm=numpy.zeros((ntimes,fdim),dtype=numpy.float32)
    w=numpy.zeros(fdim,dtype=numpy.float32)
    if oldw!=None:
        w[:-1]=oldw
    for l in range(posntimes):
        bigm[l,:-1]=posnfeat[l]
        bigm[l,-1]=bias
    for l in range(negntimes):
        bigm[posntimes+l,:-1]=negnfeat[l]
        bigm[posntimes+l,-1]=bias
    labels=numpy.concatenate((numpy.ones(posntimes),-numpy.ones(negntimes))).astype(numpy.float32)
    print "Starting Pegasos SVM training"
    ff.write("Starting Pegasos SVM training\n")
    lamd=1/(pc*ntimes)
    obj=0.0
    for tt in range(maxtimes):
        lpeg.fast_pegasos(w,fdim,bigm,ntimes,labels,lamd,ntimes*10,tt)
        nobj=lpeg.objective(w,fdim,bigm,ntimes,labels,lamd)
        print "Objective Function:",nobj
        ff.write("Objective Function:%f\n"%nobj)
        ratio=abs(abs(obj/nobj)-1)
        print "Ratio:",ratio
        ff.write("Ratio:%f\n"%ratio)
        if ratio<eps:
            print "Converging after %d iterations"%tt
            break
        obj=nobj
        #sts.report(fname,"a","Training")
    b=-w[-1]*float(bias)
    #fast_pegasos(ftype *w,int wx,ftype *ex,int exy,ftype *label,ftype lambda,int iter,int part)
    return w[:-1],b

def trainSvmRawPegComp(trpos,trneg,trposcl,trnegcl,fname,oldw=None,dir="./save/",pc=0.017,path="/home/marcopede/code/c/liblinear-1.7",maxtimes=100,eps=0.01,bias=100):
    """
        The same as trainSVMRaw but it does use files instad of lists:
        it is slower but it needs less memory.
    """
    ff=open(fname,"a")
    numcomp=numpy.array(trposcl).max()+1
    trcomp=[]
    newtrcomp=[]
    trcompcl=[]
    alabel=[]
    label=[]
    for l in range(numcomp):
        trcomp.append([])#*numcomp
        label.append([])
    #trcomp=[trcomp]
    #trcompcl=[]
    #label=[[]]*numcomp
    compx=[0]*numcomp
    compy=[0]*numcomp
    for l in range(numcomp):
        compx[l]=len(trpos[numpy.where(numpy.array(trposcl)==l)[0][0]])+1
    for p,val in enumerate(trposcl):
        trcomp[val].append(numpy.concatenate((trpos[p],[1])).astype(c_float))
        #trcompcl.append(val)
        label[val].append(1)
        compy[val]+=1
    for p,val in enumerate(trnegcl):
        trcomp[val].append(numpy.concatenate((trneg[p],[1])).astype(c_float))        
        #trcompcl.append(val)
        label[val].append(-1)
        compy[val]+=1
    ntimes=len(trpos)+len(trneg)
    fdim=numpy.sum(compx)#len(posnfeat[0])+1
    #bigm=numpy.zeros((ntimes,fdim),dtype=numpy.float32)
    w=numpy.zeros(fdim,dtype=numpy.float32)
    if oldw!=None:
        w[:-1]=oldw
    #for l in range(posntimes):
    #    bigm[l,:-1]=posnfeat[l]
    #    bigm[l,-1]=bias
    #for l in range(negntimes):
    #    bigm[posntimes+l,:-1]=negnfeat[l]
    #    bigm[posntimes+l,-1]=bias
    #labels=numpy.concatenate((numpy.ones(posntimes),-numpy.ones(negntimes))).astype(numpy.float32)
    for l in range(numcomp):
        trcompcl=numpy.concatenate((trcompcl,numpy.ones(compy[l],dtype=numpy.int32)*l))
        alabel=numpy.concatenate((alabel,numpy.array(label[l])))
    trcompcl=trcompcl.astype(numpy.int32)
    alabel=alabel.astype(numpy.int32)
    arrint=(c_int*numcomp)
    arrfloat=(c_void_p*numcomp)
    #trcomp1=[list()]*numcomp
    for l in range(numcomp):#convert to array
        trcomp[l]=numpy.array(trcomp[l])
        newtrcomp.append(trcomp[l].ctypes.data_as(c_void_p))
    print "Clusters size:",compx
    print "Clusters elements:",compy
    print "Starting Pegasos SVM training"
    ff.write("Starting Pegasos SVM training\n")
    lamd=1/(pc*ntimes)
    obj=0.0
    ncomp=c_int(numcomp)
    
    #print "check"
    #pylab.figure(340);pylab.plot(alabel);pylab.show()
    #pylab.figure(350);pylab.plot(trcomp[0].sum(0));pylab.show()
    #pylab.figure(350);pylab.plot(trcomp[1].sum(0));pylab.show()
    #print "X0:",trcomp[0][0,:7],newtrcomp[0]
    #print "X1:",trcomp[1][0,:7],newtrcomp[1]
    #raw_input()

    for tt in range(maxtimes):
        lpeg.fast_pegasos_comp(w,ncomp,arrint(*compx),arrint(*compy),arrfloat(*newtrcomp),ntimes,alabel,trcompcl,lamd,ntimes*10,tt)
        #nobj=lpeg.objective(w,fdim,bigm,ntimes,labels,lamd)
        nobj=1
        print "Objective Function:",nobj
        ff.write("Objective Function:%f\n"%nobj)
        ratio=abs(abs(obj/nobj)-1)
        print "Ratio:",ratio
        ff.write("Ratio:%f\n"%ratio)
        #if ratio<eps:
        #    print "Converging after %d iterations"%tt
        #    break
        obj=nobj
        #sts.report(fname,"a","Training")
    #b=-w[-1]*float(bias)
    #fast_pegasos(ftype *w,int wx,ftype *ex,int exy,ftype *label,ftype lambda,int iter,int part)
    return w,0#w[:-1],b

def trainSvmRawLinBlk(negnfeat,posnfeat,str,dir="./save/",pc=0.017,path="/home/marcopede/code/c/liblinear-cdblock-1.7",parts=-1,maxsize=2):
    """
        The same as trainSVMRaw but it does use files instad of lists:
        it is slower but it needs less memory.
    """
    posntimes=len(posnfeat)
    negntimes=len(negnfeat)
    ntimes=posntimes+negntimes
    labels = numpy.concatenate((numpy.ones(posntimes),-numpy.ones(negntimes)),1)#.tolist()
    toSvmfileFast(posnfeat,negnfeat,dir+str+".data",bias=100)
    if parts<=0:
        print "Building Blocks of %d GB"%(maxsize)
        parts=(len(posnfeat)+len(negnfeat))*len(negnfeat[0])*4/(maxsize*1024*1024*1024)+1
    del posnfeat,negnfeat #save memory but be carefull because it is not safe
    print "Starting Linear Block SVM training in %d parts"%parts
    #cmd=path+"/train -s 3 -c %f -B 1 "%(pc)+dir+str+".data "+dir+str+"\n"  
    str1=str.split("/")[-1]
    cmd=path+"/blocksplit -m %d "%(parts)+dir+str+".data \n"
    print cmd
    call(cmd, shell = True, stdout = PIPE)
    #./blocktrain -a webtrain.cookie -m 2 webtrain.40 model
    #cmd=path+"/blocktrain -a pmodel -m 5 -M 1000 -B 100 -s 1 -c %f "%(pc)+dir+str1+".data.%d "%(parts)+dir+str+"\n"
    cmd=path+"/blocktrain -B -1 -E 0.01 -s 1 -c %f "%(pc)+dir+str1+".data.%d "%(parts)+dir+str+"\n"
    print cmd
    call(cmd, shell = True)#, stdout = PIPE)
    #removing tmp files
    import os
    os.remove(dir+str+".data")
    #os.remove("pmodel")
    call("rm -r "+dir+str1+".data.%d"%parts,shell=True,stdout = PIPE)
    return 0


def trainSvmRawLib(posnfeat,negnfeat,str,dir="./save/",pc=0.017,t=0,pr=0,g=1.0,big=False):
    """
        The same as trainSVMRaw but it does use files instad of lists:
        it is slower but it needs less memory.
    """
    #kernels = [LINEAR, POLY, RBF]
    #kname = ['linear','polynomial','rbf']
    posntimes=len(posnfeat)
    negntimes=len(negnfeat)
    ntimes=posntimes+negntimes
    #possample=posnfeat.tolist()
    #negsample=negnfeat.tolist()
    labels = numpy.concatenate((numpy.ones(posntimes),-numpy.ones(negntimes)),1)#.tolist()
    #samples = possample + negsample
    #samples=numpy.concatenate((posnfeat,negnfeat),0)
    #problem = svm_problem(labels, samples);
    #size = len(samples)
    #print "Starting SVM"
    #param = svm_parameter(C = pc,kernel_type = LINEAR)#,nr_weight = 2,weight_label = [1,0]#,weight = [10,1])
    #model = svm_model(problem,param)
    #toSvmfile(posnfeat,negnfeat,dir+str+".data")
    #raw_input()
    toSvmfileFast(posnfeat,negnfeat,dir+str+".data")
    del posnfeat,negnfeat #save memory but be carefull because it is not safe
    #cmd="./easy data."+str+" "+str 
    print "Starting SVM, pr=%d"%pr
    if big:
        import os
        myenv=os.environ
        myenv["OMP_NUM_THREADS"]="16"
        #myargs=["svm-train","-t %d"%t,"-c 0.017","-m 1000","-b %d"%pr,dir+str+".data",dir+str]
        #myargs=["svm-train","-t %d"%t,"-c 0.017","-m 1000","-b %d"%pr,dir+str+".data",dir+str]
        #myargs=["-t %d"%t,"-c 0.017","-b %d"%pr,dir+str+".data",dir+str]
        #os.execve("/home/marcopede/code/c/libsvm-2.89/svm-train",myargs,myenv)
        #os.execl("/home/marcopede/code/c/libsvm-2.89/svm-train","svm-train","-t %d"%t,"-c 0.017","-b %d"%pr,dir+str+".data",dir+str)
        #os.execl("/home/marcopede/code/c/libsvm-2.89/svm-train","svm-train","-t","%d"%t,"-b","%d"%pr,dir+str+".data",dir+str)
        print "svm-train -t %d -m 1000 -c %f -g %f -b %d"%(t,pc,g,pr)
        #os.execle("/home/marcopede/code/c/libsvm-2.89-threads/svm-train","svm-train","-t","%d"%t,"-m","1000","-c","%f"%pc,"-g","%f"%g,"-b","%d"%pr,dir+str+".data",dir+str,myenv)
        os.execle("/home/marcopede/code/c/libsvm-2.89-threads/svm-train","svm-train","-t","%d"%t,"-m","1000","-c","%f"%pc,"-g","%f"%g,"-b","%d"%pr,dir+str+".data",dir+str,myenv)
    else:
        import default 
        #cmd=default.svmpath+"/train -s 3 -c %f -B 1 "%(pc)+dir+str+".data "+dir+str+"\n" 
        cmd=default.svmpath+"/svm-train -t %d -c %f -m 1000 -b %d -g %f "%(t,pc,pr,g)+dir+str+".data "+dir+str+"\n" 
        call(cmd, shell = True, stdout = PIPE)
        import os
        os.remove(dir+str+".data")
    return 0

def trainSvmRawLibnfold(posnfeat,negnfeat,str,dir="./save/",pc=0.017,t=0,pr=0,g=1.0,big=False,nfold=5):
    """
        The same as trainSVMRaw but it does use files instad of lists:
        it is slower but it needs less memory.
    """
    posntimes=posnfeat.shape[0]
    negntimes=negnfeat.shape[0]
    dpos=posntimes/nfold
    dneg=negntimes/nfold
    for l in range(nfold):
        posfeat=numpy.concatenate((posnfeat[:dpos*(l),:],posnfeat[dpos*(l+1):,:]),0)
        negfeat=numpy.concatenate((negnfeat[:dneg*(l),:],negnfeat[dneg*(l+1):,:]),0)
        trainSvmRawLib(posnfeat,negnfeat,str,dir,pc,t,pr,g,big=False)
        postest=numpy.concatenate((numpy.zeros((dpos,1)),posnfeat[dpos*(l):dpos*(l+1),:]),1).astype(numpy.float32)
        negtest=numpy.concatenate((numpy.zeros((dneg,1)),negnfeat[dneg*(l):dneg*(l+1),:]),1).astype(numpy.float32)
        import mysvm
        m=mysvm.svm_model(dir+str)
        pp=m.predict_values(postest)
        nn=m.predict_values(negtest)
        print float(numpy.sum(pp>0)+numpy.sum(pp<0))/(dpos+dneg)
        raw_input()
    return 0

def trainSvmRawLight(posnfeat,negnfeat,str,dir="./save/",pc=0.1):
    """
        The same as trainSVMRaw but it does use files instad of lists:
        it is slower but it needs less memory.
    """
    #kernels = [LINEAR, POLY, RBF]
    #kname = ['linear','polynomial','rbf']
    posntimes=posnfeat.shape[0]
    negntimes=negnfeat.shape[0]
    ntimes=posntimes+negntimes
    #possample=posnfeat.tolist()
    #negsample=negnfeat.tolist()
    labels = numpy.concatenate((numpy.ones(posntimes),-numpy.ones(negntimes)),1)#.tolist()
    #samples = possample + negsample
    #samples=numpy.concatenate((posnfeat,negnfeat),0)
    #problem = svm_problem(labels, samples);
    #size = len(samples)
    print "Starting SVM"
    #param = svm_parameter(C = pc,kernel_type = LINEAR)#,nr_weight = 2,weight_label = [1,0]#,weight = [10,1])
    #model = svm_model(problem,param)
    toSvmfileFast(posnfeat,negnfeat,dir+str+".data")
    #toSvmfile(posnfeat,negnfeat,dir+str+".data")
    del posnfeat,negnfeat #save memory but be carefull because it is not safe
    #cmd="./easy data."+str+" "+str 
    print "Starting SVM..."
    #cmd="svm-train -t 0 -c 0.01 -w1 1 -w-1 3 "+dir+str+".data "+dir+str+"\n" 
    cmd="/home/marcopede/code/c/svmlight/svm_learn -t 0 -m 1000 "+dir+str+".data "+dir+str+"\n" 
    call(cmd, shell = True, stdout = PIPE)
    return 0

def trainSvmMKL(posnfeat,negnfeat,str,dir="./save/",pc=0.1):
    """
        The same as trainSVMRaw but it does use files instad of lists:
        it is slower but it needs less memory.
    """
    from sg import sg
    npos=posnfeat.shape[0]
    nneg=negnfeat.shape[0]
    weight=1.
    labels=numpy.concatenate((numpy.ones(npos), -numpy.ones(nneg)),0)
    features=numpy.concatenate((posnfeat,negnfeat),0).T.astype(numpy.float)

    #features[:50,:100]=features[:50,:100]*2

    sg('c', 0.002)
    sg('new_svm', 'SVMLIGHT')
    sg('use_mkl', True)

    sg('set_labels', 'TRAIN', labels)
    sg('add_features', 'TRAIN', features)
##    sg('add_features', 'TRAIN', features[50:,:])
##    sg('add_features', 'TRAIN', features[:,:])
##    sg('add_features', 'TRAIN', features[:80,:])
##    sg('add_features', 'TRAIN', features[20:,:])

    sg('threads', 2) 
    sg('set_kernel', 'COMBINED', 100)
    sg('add_kernel', weight, 'CHI2', 'REAL', 100 , 1.)
##    sg('add_kernel', weight, 'LINEAR', 'REAL', 100, )
##    sg('add_kernel', weight, 'LINEAR', 'REAL', 100, )
##    sg('add_kernel', weight, 'LINEAR', 'REAL', 100, )
##    sg('add_kernel', weight, 'LINEAR', 'REAL', 100, )
    sg('init_kernel', 'TRAIN')
    sg('train_classifier')
    [bias, alphas]=sg('get_svm')
    kw=sg('get_subkernel_weights')
    rho=bias[0,0]
    ww=numpy.sum(alphas[:,0]*features[:,alphas[:,1].astype(numpy.int)],1)
    print kw
    sg('clear') 
    return ww,rho

def trainSvmMKL2(posnfeat,negnfeat,str,dir="./save/",pc=0.1,threads=8,mklon=True):
    """
        The same as trainSVMRaw but it does use files instad of lists:
        it is slower but it needs less memory.
    """
    import util
    from sg import sg
    npos=posnfeat.shape[3]
    nneg=negnfeat.shape[3]
    dy=posnfeat.shape[0]
    dx=posnfeat.shape[1]
    df=posnfeat.shape[2]
    weight=1.
    labels=numpy.concatenate((numpy.ones(npos), -numpy.ones(nneg)),0)
    features=numpy.concatenate((posnfeat.reshape(dy*dx*df,npos),negnfeat.reshape(dy*dx*df,nneg)),1).astype(numpy.float)

    #features[:50,:100]=features[:50,:100]*2

    sg('c', 0.017)
    sg('new_svm', 'SVMLIGHT')
    sg('use_mkl', True)

    sg('set_labels', 'TRAIN', labels)
    ##sg('add_features', 'TRAIN', features)
    for it in range(dx*dy):
        sg('add_features', 'TRAIN', features[it*df:+(it+1)*df,:])
##    sg('add_features', 'TRAIN', features)
##    sg('add_features', 'TRAIN', features[50:,:])
##    sg('add_features', 'TRAIN', features[:,:])
##    sg('add_features', 'TRAIN', features[:80,:])
##    sg('add_features', 'TRAIN', features[20:,:])

    sg('threads', threads) 
    sg('set_kernel', 'COMBINED', 100)
    ##sg('add_kernel', weight, 'LINEAR', 'REAL', 100 , 1.)
    for it in range(dx*dy):
        sg('add_kernel', weight, 'LINEAR', 'REAL', 100 , 1.)
##    sg('add_kernel', weight, 'LINEAR', 'REAL', 100 , 1.)
##    sg('add_kernel', weight, 'LINEAR', 'REAL', 100, )
##    sg('add_kernel', weight, 'LINEAR', 'REAL', 100, )
##    sg('add_kernel', weight, 'LINEAR', 'REAL', 100, )
##    sg('add_kernel', weight, 'LINEAR', 'REAL', 100, )
    sg('init_kernel', 'TRAIN')
    sg('train_classifier')
    [bias, alphas]=sg('get_svm')
    kw=sg('get_subkernel_weights')
    rho=bias[0,0]
    ww=numpy.sum(alphas[:,0]*features[:,alphas[:,1].astype(numpy.int)],1)
    ww=ww.reshape((dy,dx,df))
    if mklon:
        ww=(ww.T*(kw.reshape((dy,dx))).T).T
    pylab.figure()
    util.drawHOG(ww,svm="pos")
    pylab.show()
    print "KW:",kw
    sg('clear') 
    return ww,rho

def trainSvmMKL3(posnfeat,negnfeat,str,dir="./save/",pc=0.1,threads=8,mklon=True):
    """
        The same as trainSVMRaw but it does use files instad of lists:
        it is slower but it needs less memory.
    """
    import util
    from sg import sg
    npos=posnfeat[0].shape[3]
    nneg=negnfeat[0].shape[3]
    dy=posnfeat[0].shape[0]
    dx=posnfeat[0].shape[1]
    df=posnfeat[0].shape[2]
    lev=len(posnfeat)
    weight=1.
    labels=numpy.concatenate((numpy.ones(npos), -numpy.ones(nneg)),0)
    pfeatures=posnfeat[0].reshape(dy*dx*df,npos)
    for l in range(1,lev):
        pfeatures=numpy.concatenate((pfeatures,posnfeat[l].reshape(posnfeat[l].shape[0]*posnfeat[l].shape[1]*df,npos)),0)
    nfeatures=negnfeat[0].reshape(dy*dx*df,nneg)
    for l in range(1,lev):
        nfeatures=numpy.concatenate((nfeatures,negnfeat[l].reshape(negnfeat[l].shape[0]*negnfeat[l].shape[1]*df,nneg)),0)
    features=numpy.concatenate((pfeatures,nfeatures),1).astype(numpy.float)
    #features=numpy.concatenate((posnfeat.reshape(dy*dx*df,npos),negnfeat.reshape(dy*dx*df,nneg)),1).astype(numpy.float)
    

    #features[:50,:100]=features[:50,:100]*2

    sg('c', 0.017)
    #sg('c', 0.1)
    #sg('svm_epsilon',1e-5)

    sg('new_svm', 'SVMLIGHT')
    sg('use_mkl', True)

    sg('set_labels', 'TRAIN', labels)
    ##sg('add_features', 'TRAIN', features)
    pos=0
    for it in range(lev):
        sh=posnfeat[it].shape[0]*posnfeat[it].shape[1]*df
        sg('add_features', 'TRAIN', features[pos:pos+sh,:])
        pos+=sh
##    sg('add_features', 'TRAIN', features)
##    sg('add_features', 'TRAIN', features[50:,:])
##    sg('add_features', 'TRAIN', features[:,:])
##    sg('add_features', 'TRAIN', features[:80,:])
##    sg('add_features', 'TRAIN', features[20:,:])

    sg('threads', threads) 
    sg('set_kernel', 'COMBINED', 100)
    ##sg('add_kernel', weight, 'LINEAR', 'REAL', 100 , 1.)
    for it in range(lev):
        sg('add_kernel', weight, 'LINEAR', 'REAL', 100 , 1.)
        #sg('add_kernel', weight, 'CHI2', 'REAL', 1000 , 1.)
##    sg('add_kernel', weight, 'LINEAR', 'REAL', 100 , 1.)
##    sg('add_kernel', weight, 'LINEAR', 'REAL', 100, )
##    sg('add_kernel', weight, 'LINEAR', 'REAL', 100, )
##    sg('add_kernel', weight, 'LINEAR', 'REAL', 100, )
##    sg('add_kernel', weight, 'LINEAR', 'REAL', 100, )
    sg('init_kernel', 'TRAIN')
    sg('train_classifier')
    [bias, alphas]=sg('get_svm')
    kw=sg('get_subkernel_weights')
    rho=-bias[0,0]
    w=numpy.sum(alphas[:,0]*features[:,alphas[:,1].astype(numpy.int)],1)
    ww=[]
    pos=0
    for it in range(lev):
        sh=posnfeat[it].shape[0]*posnfeat[it].shape[1]*df
        ww.append(w[pos:pos+sh].reshape(posnfeat[it].shape[0],posnfeat[it].shape[1],df))
        pos+=sh
    #pylab.figure()
    #pylab.show()
    #util.drawHOGmodelPyr(ww,svm="pos",type=1,kw=kw)
    for it in range(lev):
        if mklon:
            ww[it]*=kw[it]
    print kw
    sg('clear') 
    return ww,rho,kw

def trainSvmMKLCHI(posnfeat,negnfeat,posbow,negbow,filename,pc=0.1,threads=8,mklon=True):
    """
        
    """
    import util
    from sg import sg
    npos=posnfeat[0].shape[3]
    nneg=negnfeat[0].shape[3]
    dy=posnfeat[0].shape[0]
    dx=posnfeat[0].shape[1]
    df=posnfeat[0].shape[2]
    lev=len(posnfeat)
    weight=1.
    labels=numpy.concatenate((numpy.ones(npos), -numpy.ones(nneg)),0)
    pfeatures=posnfeat[0].reshape(dy*dx*df,npos)
    for l in range(1,lev):
        pfeatures=numpy.concatenate((pfeatures,posnfeat[l].reshape(posnfeat[l].shape[0]*posnfeat[l].shape[1]*df,npos)),0)
    #pfeatures=numpy.concatenate((pfeatures,numpy.concatenate((posbow,posbow),1)),0)
    pfeatures=numpy.concatenate((pfeatures,posbow),0)
    nfeatures=negnfeat[0].reshape(dy*dx*df,nneg)
    for l in range(1,lev):
        nfeatures=numpy.concatenate((nfeatures,negnfeat[l].reshape(negnfeat[l].shape[0]*negnfeat[l].shape[1]*df,nneg)),0)
    #nfeatures=numpy.concatenate((nfeatures,numpy.concatenate((negbow,negbow),1)),0)
    nfeatures=numpy.concatenate((nfeatures,negbow),0)
    features=numpy.concatenate((pfeatures,nfeatures),1).astype(numpy.float)
    #features=numpy.concatenate((posnfeat.reshape(dy*dx*df,npos),negnfeat.reshape(dy*dx*df,nneg)),1).astype(numpy.float)
    

    #features[:50,:100]=features[:50,:100]*2

    sg('c', 0.017)
    #sg('c', 0.1)
    #sg('svm_epsilon',1e-5)

    sg('new_svm', 'SVMLIGHT')
    sg('use_mkl', True)

    sg('set_labels', 'TRAIN', labels)
    ##sg('add_features', 'TRAIN', features)
    pos=0
    for it in range(lev):
        sh=posnfeat[it].shape[0]*posnfeat[it].shape[1]*df
        sg('add_features', 'TRAIN', features[pos:pos+sh,:])
        pos+=sh
    sg('add_features', 'TRAIN', features[pos:,:])        
##    sg('add_features', 'TRAIN', features)
##    sg('add_features', 'TRAIN', features[50:,:])
##    sg('add_features', 'TRAIN', features[:,:])
##    sg('add_features', 'TRAIN', features[:80,:])
##    sg('add_features', 'TRAIN', features[20:,:])

    sg('threads', threads) 
    sg('set_kernel', 'COMBINED', 100)
    ##sg('add_kernel', weight, 'LINEAR', 'REAL', 100 , 1.)
    for it in range(lev):
        sg('add_kernel', weight, 'CHI2', 'REAL', 100 , 1.)
    sg('add_kernel', weight, 'CHI2', 'REAL', 100 , 1.)
        #sg('add_kernel', weight, 'CHI2', 'REAL', 1000 , 1.)
##    sg('add_kernel', weight, 'LINEAR', 'REAL', 100 , 1.)
##    sg('add_kernel', weight, 'LINEAR', 'REAL', 100, )
##    sg('add_kernel', weight, 'LINEAR', 'REAL', 100, )
##    sg('add_kernel', weight, 'LINEAR', 'REAL', 100, )
##    sg('add_kernel', weight, 'LINEAR', 'REAL', 100, )
    sg('init_kernel', 'TRAIN')
    sg('train_classifier')
##    [bias, alphas]=sg('get_svm')
    kw=sg('get_subkernel_weights')
##    rho=-bias[0,0]
##    w=numpy.sum(alphas[:,0]*features[:,alphas[:,1].astype(numpy.int)],1)
##    ww=[]
##    pos=0
##    for it in range(lev):
##        sh=posnfeat[it].shape[0]*posnfeat[it].shape[1]*df
##        ww.append(w[pos:pos+sh].reshape(posnfeat[it].shape[0],posnfeat[it].shape[1],df))
##        pos+=sh
##    #pylab.figure()
##    #pylab.show()
##    #util.drawHOGmodelPyr(ww,svm="pos",type=1,kw=kw)
##    for it in range(lev):
##        if mklon:
##            ww[it]*=kw[it]
    print kw
    #sg('save_features', filename+".sv","REAL","TARGET")
    sg('save_classifier', filename+".sv")
    sg('save_kernel', filename+".ker")
    save(filename,features)
    #km=sg('get_kernel_matrix')
    #print km
    #fdsdf
    sg('clear') 
##    return ww,rho,kw


def propagate(parts,rparts=2,lev=1,shift=False):
    """generate a new body at the resolution lev using the relative position"""
    mul=2**lev
    if type(rparts)==int: #if rparts is a number split the previous level into rparts
        newparts=numpy.zeros((rparts,rparts,parts.shape[1]),dtype=part)
        for i in range(parts.shape[0]):
            for idy in range(rparts):
                for idx in range(rparts):
                    newparts[idy,idx,:]["itr"]=parts[i,:]["itr"]
                    newparts[idy,idx,:]["oct"]=parts[i,:]["oct"]-lev
                    newparts[idy,idx,:]["y"]=parts[i,:]["y"]*mul+idy*(parts[i,:]["sy"]*mul)/rparts
                    newparts[idy,idx,:]["x"]=parts[i,:]["x"]*mul+idx*(parts[i,:]["sx"]*mul)/rparts
                    newparts[idy,idx,:]["sy"]=(parts[i,:]["sy"]+1)*mul/rparts
                    newparts[idy,idx,:]["sx"]=(parts[i,:]["sx"]+1)*mul/rparts
        newparts=newparts.reshape((rparts*rparts,parts.shape[1]))
                    #newpos=Pos(item.pos.itr,item.pos.oct+1,(item.pos.y+1)*2-1+idy*(item.pos.sy*2)/rparts-1,(item.pos.x+1)*2-1+idx*(item.pos.sx*2)/rparts-1,(item.pos.sy+1)*2/rparts,(item.pos.sx+1)*2/rparts)
                    #newpart=Part(newpos)
                    #newbody.parts.append(newpart)
    else: #if rparts is a list of parts use it as new level structure for every part
        newparts=numpy.zeros((rparts.shape[0],parts.shape[0],parts.shape[1]),dtype=part)
        for i in range(parts.shape[0]):
            for id in range(rparts.shape[0]):
                newparts[id,i,:]["itr"]=parts[i,:]["itr"]
                newparts[id,i,:]["oct"]=parts[i,:]["oct"]-lev
                if shift:
                    newparts[id,i,:]["y"]=(parts[i,:]["y"])*mul+rparts[id]["y"]+1
                    newparts[id,i,:]["x"]=(parts[i,:]["x"])*mul+rparts[id]["x"]+1
                else:
                    newparts[id,i,:]["y"]=(parts[i,:]["y"])*mul+rparts[id]["y"]
                    newparts[id,i,:]["x"]=(parts[i,:]["x"])*mul+rparts[id]["x"]
                newparts[id,i,:]["sy"]=rparts[id]["sy"]
                newparts[id,i,:]["sx"]=rparts[id]["sx"]
        newparts=newparts.reshape((rparts.shape[0]*parts.shape[0],parts.shape[1]))
                #newpos=Pos(item.pos.itr,item.pos.oct+1,(item.pos.y+1)*2-1+item2.pos.y,(item.pos.x+1)*2-1+item2.pos.x,item2.pos.sy,item2.pos.sx)
                #newpart=Part(newpos)
                #print newpart
                #newbody.parts.append(newpart)
    return newparts

def propagate2(parts,interval=10,rparts=2,lev=1,shift=False):
    """generate a new body at the resolution lev using the relative position"""
    mul=2**lev
    if type(rparts)==int: #if rparts is a number split the previous level into rparts
        newparts=numpy.zeros((rparts,rparts,parts.shape[0],parts.shape[1]),dtype=part)
        for i in range(parts.shape[0]):
            for idx in range(rparts):
                for idy in range(rparts):
                    newparts[idy,idx,i,:]["itr"]=parts[i,:]["itr"]-10*lev
                    newparts[idy,idx,i,:]["oct"]=i
                    filter=(parts[i,:]["sy"]*mul)%rparts==0
                    newparts["sy"][idy,idx,i,filter]=parts[i,filter]["sy"]*mul/rparts
                    newparts["sy"][idy,idx,i,(filter==False)]=(parts[i,(filter==False)]["sy"]*mul/rparts)+1
                    newparts["sx"][idy,idx,i,filter]=parts[i,filter]["sx"]*mul/rparts
                    newparts["sx"][idy,idx,i,(filter==False)]=(parts[i,(filter==False)]["sx"]*mul/rparts)+1
                    newparts["y"][idy,idx,i,filter]=parts[i,filter]["y"]*mul+idx*newparts[idy,idx,i,filter]["sy"]+1
                    newparts["y"][idy,idx,i,filter==False]=parts[i,filter==False]["y"]*mul+idy*(newparts[idy,idx,i,filter==False]["sy"]-1)+1
                    newparts["x"][idy,idx,i,filter]=parts[i,filter]["x"]*mul+idy*newparts[idy,idx,i,filter]["sx"]+1
                    newparts["x"][idy,idx,i,filter==False]=parts[i,filter==False]["x"]*mul+idx*(newparts[idy,idx,i,filter==False]["sx"]-1)+1
##                    newparts[idy,idx,:]["sy"]=((parts[i,:]["sy"]*mul).astype(numpy.float)/rparts)
##                    newparts[idy,idx,:]["sx"]=((parts[i,:]["sx"]*mul).astype(numpy.float)/rparts)
        newparts=newparts.reshape((rparts*rparts*parts.shape[0],parts.shape[1]))
                    #newpos=Pos(item.pos.itr,item.pos.oct+1,(item.pos.y+1)*2-1+idy*(item.pos.sy*2)/rparts-1,(item.pos.x+1)*2-1+idx*(item.pos.sx*2)/rparts-1,(item.pos.sy+1)*2/rparts,(item.pos.sx+1)*2/rparts)
                    #newpart=Part(newpos)
                    #newbody.parts.append(newpart)
    else: #if rparts is a list of parts use it as new level structure for every part
        newparts=numpy.zeros((rparts.shape[0],parts.shape[0],parts.shape[1]),dtype=part)
        for i in range(parts.shape[0]):
            for id in range(rparts.shape[0]):
                newparts[id,i,:]["itr"]=parts[i,:]["itr"]
                newparts[id,i,:]["oct"]=parts[i,:]["oct"]-lev
                if shift:
                    newparts[id,i,:]["y"]=(parts[i,:]["y"])*mul+rparts[id]["y"]+1
                    newparts[id,i,:]["x"]=(parts[i,:]["x"])*mul+rparts[id]["x"]+1
                else:
                    newparts[id,i,:]["y"]=(parts[i,:]["y"])*mul+rparts[id]["y"]
                    newparts[id,i,:]["x"]=(parts[i,:]["x"])*mul+rparts[id]["x"]
                newparts[id,i,:]["sy"]=rparts[id]["sy"]
                newparts[id,i,:]["sx"]=rparts[id]["sx"]
        newparts=newparts.reshape((rparts.shape[0]*parts.shape[0],parts.shape[1]))
                #newpos=Pos(item.pos.itr,item.pos.oct+1,(item.pos.y+1)*2-1+item2.pos.y,(item.pos.x+1)*2-1+item2.pos.x,item2.pos.sy,item2.pos.sx)
                #newpart=Part(newpos)
                #print newpart
                #newbody.parts.append(newpart)
    return newparts

def packold(fparts,delta=[],obin=16,add=12,defmode="pedro"):
    #for the moment delta is not considered
    import scipy.ndimage.filters as flt
    totsize=0
    totdef=0
    sz=[]
    szdef=[]
    auxdelta=[]
    for pid,p in enumerate(fparts):
        sz.append(numpy.prod(p.shape[:-1]))
        totsize+=numpy.prod(p.shape[:-1])
        szdef.append(2*4**pid)
        totdef+=2*4**pid
    packfeat=numpy.zeros((p.shape[-1],totsize),numpy.float32)
    pt=0
    #to avoid to have features for the root filter
    for id,p in enumerate(fparts):
        packfeat[:,pt:pt+sz[id]]=p.T.reshape((p.shape[-1],sz[id]))
        pt=pt+sz[id]
    if delta!=[]:
        pt=0
        deffeat=numpy.zeros((fparts[0].shape[-1],totdef-2))
        if defmode=="pede" or defmode=="all":
            for l in range(1,len(fparts)):
                #deffeat=flt.correlate(delta[l][:,:,0,:],[[1][-1]])
                v=flt.correlate1d(delta[l][:,:,0,:],[1,-1],0)
                v2=flt.correlate1d(delta[l][:,:,0,:],[1,-1],1)
                h2=flt.correlate1d(delta[l][:,:,1,:],[1,-1],1)
                h=flt.correlate1d(delta[l][:,:,1,:],[1,-1],0)
                #deffeat=numpy.concatenate((v[1::2,:,:].T.reshape(delta[l]**2/2),h[:,1::2,:].T.reshape(delta[l]**2/2)),0),)
                deffeat[:,pt:pt+szdef[l]/2]=numpy.concatenate(((v2[:,1::2,:]).T.reshape(fparts[0].shape[-1],4**l/2)**2,((h2[:,1::2,:]).T.reshape(fparts[0].shape[-1],4**l/2))**2),1)
                deffeat[:,pt+szdef[l]/2:pt+szdef[l]]=numpy.concatenate((numpy.swapaxes(v[1::2,:,:],0,1).T.reshape(fparts[0].shape[-1],4**l/2)**2,(numpy.swapaxes(h[1::2,:,:],0,1).T.reshape(fparts[0].shape[-1],4**l/2))**2),1)
                pt+=szdef[l]
                print deffeat
                #raw_input()
            #packfeat=numpy.concatenate((packfeat,-deffeat/10.0),1).astype(numpy.float32)
        pt=0
        deffeat2=numpy.zeros((fparts[0].shape[-1],totdef-2))
        if defmode=="pedro" or defmode=="all":
            for l in range(1,len(fparts)):
                deffeat2[:,pt:pt+szdef[l]]=(delta[l][:,:,:,:]**2).reshape((szdef[l],fparts[0].shape[-1])).T
                pt+=szdef[l]
            #deffeat2[numpy.all(deffeat2==0,1),:]=1
            print "Pedro def:",deffeat2
            print deffeat2.mean(0)
            #note the next line
            #deffeat2=numpy.zeros((fparts[0].shape[-1],totdef-2))
        packfeat=numpy.concatenate((packfeat,deffeat/3.0,deffeat2/3.0),1).astype(numpy.float32)
        #raw_input()
    return packfeat

def pack(fparts,delta=[],obin=16,add=12,defmode="pedro"):
    #for the moment delta is not considered
    import scipy.ndimage.filters as flt
    totsize=0
    totdef=0
    sz=[]
    szdef=[]
    auxdelta=[]
    for pid,p in enumerate(fparts):
        sz.append(numpy.prod(p.shape[:-1]))
        totsize+=numpy.prod(p.shape[:-1])
        szdef.append(2*4**pid)
        totdef+=2*4**pid
    packfeat=numpy.zeros((p.shape[-1],totsize),numpy.float32)
    pt=0
    #to avoid to have features for the root filter
    for id,p in enumerate(fparts):
        packfeat[:,pt:pt+sz[id]]=p.T.reshape((p.shape[-1],sz[id]))
        pt=pt+sz[id]
    if delta!=[]:
        pt=0
        deffeat=numpy.zeros((fparts[0].shape[-1],totdef-2),numpy.float32)
        if defmode=="pede" or defmode=="all":
            for l in range(1,len(fparts)):
                #valid only for l<=2
                #deffeat=flt.correlate(delta[l][:,:,0,:],[[1][-1]])
                v=flt.correlate1d(delta[l][:,:,0,:],[1,-1],0)[1,:,:].T.squeeze()**2
                v2=flt.correlate1d(delta[l][:,:,0,:],[1,-1],1)[:,1,:].T.squeeze()**2
                h2=flt.correlate1d(delta[l][:,:,1,:],[1,-1],1)[:,1,:].T.squeeze()**2
                h=flt.correlate1d(delta[l][:,:,1,:],[1,-1],0)[1,:,:].T.squeeze()**2
                deffeat[:,pt:pt+szdef[l]]=numpy.concatenate((v2,h2,v,h),1)
                pt+=szdef[l]
                print deffeat
                #raw_input()
            #packfeat=numpy.concatenate((packfeat,-deffeat/10.0),1).astype(numpy.float32)
        pt=0
        deffeat2=numpy.zeros((fparts[0].shape[-1],totdef-2),numpy.float32)
        if defmode=="pedro" or defmode=="all":
            for l in range(1,len(fparts)):
                deffeat2[:,pt:pt+szdef[l]]=(delta[l][:,:,:,:]**2).reshape((szdef[l],fparts[0].shape[-1])).T
                pt+=szdef[l]
            #deffeat2[numpy.all(deffeat2==0,1),:]=1
            print "Pedro def:",deffeat2
            print deffeat2.mean(0)
            #note the next line
            #deffeat2=numpy.zeros((fparts[0].shape[-1],totdef-2))
        packfeat=numpy.concatenate((packfeat,deffeat/3.0,deffeat2/3.0),1)
        #raw_input()
    return packfeat


def unpackPyrold(packfeat,cfg,delta=[]):
    #for the moment delta is not considered
    st=[]
    sz=[]
    szdef=[]
    feat=[]
    #totdef=0
    for l in range(cfg.lev):
        st.append(numpy.prod(cfg.fx*2**l*cfg.fy*2**l*(cfg.ds)))
        sz.append((cfg.fy*2**l,cfg.fx*2**l,cfg.ds))
        szdef.append(4**l)
        #totdef+=4**l
    pt=0
    for id,p in enumerate(st):
        feat.append(packfeat[pt:pt+p].reshape(sz[id][2],sz[id][1],sz[id][0]).T)
        pt+=p
    if delta:
        print "Before Clip:",packfeat[pt:]
        packfeat[pt:]=numpy.clip(packfeat[pt:],-1,-0.001)#numpy.abs(packfeat[pt:])
        hCue=[]
        vCue=[]
        hCue2=[]
        vCue2=[]
        vCue.append(packfeat[pt:pt+szdef[1]/2])
        vCue.append([])
        pt+=szdef[1]/2
        hCue.append(packfeat[pt:pt+szdef[1]/2])
        hCue.append([])       
        pt+=szdef[1]/2
        vCue2.append(packfeat[pt:pt+szdef[1]/2])
        vCue2.append([])
        pt+=szdef[1]/2
        hCue2.append(packfeat[pt:pt+szdef[1]/2])
        hCue2.append([])       
        pt+=szdef[1]/2
        for l in range(2,cfg.lev):
            vCue[-1].append(packfeat[pt:pt+2])
            vCue[-1].append([])
            vCue[-1].append(packfeat[4+pt:pt+6])
            vCue[-1].append([])
            vCue[-1].append(packfeat[2+pt:pt+4])
            vCue[-1].append([])
            vCue[-1].append(packfeat[6+pt:pt+8])
            vCue[-1].append([])
            pt+=8
            hCue[-1].append(packfeat[pt:pt+2])
            hCue[-1].append([])
            hCue[-1].append(packfeat[4+pt:pt+6])
            hCue[-1].append([])
            hCue[-1].append(packfeat[2+pt:pt+4])
            hCue[-1].append([])
            hCue[-1].append(packfeat[6+pt:pt+8])
            hCue[-1].append([])
            pt+=8
            vCue2[-1].append(packfeat[pt:pt+2])
            vCue2[-1].append([])
            vCue2[-1].append(packfeat[2+pt:pt+4])
            vCue2[-1].append([])
            vCue2[-1].append(packfeat[4+pt:pt+6])
            vCue2[-1].append([])
            vCue2[-1].append(packfeat[6+pt:pt+8])
            vCue2[-1].append([])
            pt+=8
            hCue2[-1].append(packfeat[pt:pt+2])
            hCue2[-1].append([])
            hCue2[-1].append(packfeat[2+pt:pt+4])
            hCue2[-1].append([])
            hCue2[-1].append(packfeat[4+pt:pt+6])
            hCue2[-1].append([])
            hCue2[-1].append(packfeat[6+pt:pt+8])
            hCue2[-1].append([])
            pt+=8
        #pedro feat
        packfeat[pt:]=numpy.clip(packfeat[pt:],-1,-0.001)#numpy.abs(packfeat[pt:])
        pdef=[]
        pdef.append(packfeat[pt:pt+2])
        pdef.append([])
        pt+=2
        pdef.append(packfeat[pt:pt+2])
        pdef.append([])
        pt+=2
        pdef.append(packfeat[pt:pt+2])
        pdef.append([])
        pt+=2
        pdef.append(packfeat[pt:pt+2])
        pdef.append([])
        pt+=2
        for l in range(2,cfg.lev):
            pdef[1]=[]
            pdef[1].append(packfeat[pt:pt+2])
            pdef[1].append([])
            pt+=2
            pdef[1].append(packfeat[pt:pt+2])
            pdef[1].append([])
            pt+=2
            pdef[3]=[]
            pdef[3].append(packfeat[pt:pt+2])
            pdef[3].append([])
            pt+=2
            pdef[3].append(packfeat[pt:pt+2])
            pdef[3].append([])
            pt+=2
            pdef[1].append(packfeat[pt:pt+2])
            pdef[1].append([])
            pt+=2
            pdef[1].append(packfeat[pt:pt+2])
            pdef[1].append([])
            pt+=2
            pdef[3].append(packfeat[pt:pt+2])
            pdef[3].append([])
            pt+=2
            pdef[3].append(packfeat[pt:pt+2])
            pdef[3].append([])
            pt+=2
        
            pdef[5]=[]
            pdef[5].append(packfeat[pt:pt+2])
            pdef[5].append([])
            pt+=2
            pdef[5].append(packfeat[pt:pt+2])
            pdef[5].append([])
            pt+=2
            pdef[7]=[]
            pdef[7].append(packfeat[pt:pt+2])
            pdef[7].append([])
            pt+=2
            pdef[7].append(packfeat[pt:pt+2])
            pdef[7].append([])
            pt+=2
            pdef[5].append(packfeat[pt:pt+2])
            pdef[5].append([])
            pt+=2
            pdef[5].append(packfeat[pt:pt+2])
            pdef[5].append([])
            pt+=2
            pdef[7].append(packfeat[pt:pt+2])
            pdef[7].append([])
            pt+=2
            pdef[7].append(packfeat[pt:pt+2])
            pdef[7].append([])
            pt+=2
        delta=[hCue,vCue,hCue2,vCue2,pdef]
#    if delta==[]:
        print delta
        print packfeat[-40:]
        #raw_input()
        return feat,delta
    else:
        return feat

def unpackPyr(packfeat,cfg,delta=[]):
    #for the moment delta is not considered
    st=[]
    sz=[]
    szdef=[]
    feat=[]
    #totdef=0
    for l in range(cfg.lev):
        st.append(numpy.prod(cfg.fx*2**l*cfg.fy*2**l*(cfg.ds)))
        sz.append((cfg.fy*2**l,cfg.fx*2**l,cfg.ds))
        szdef.append(4**l)
        #totdef+=4**l
    pt=0
    for id,p in enumerate(st):
        feat.append(packfeat[pt:pt+p].reshape(sz[id][2],sz[id][1],sz[id][0]).T)
        pt+=p
    if delta:
        print "Before Clip:",packfeat[pt:]
        packfeat[pt:]=numpy.clip(packfeat[pt:],-1,-0.001)#numpy.abs(packfeat[pt:])
        hCue=[]
        vCue=[]
        hCue2=[]
        vCue2=[]
        hCue.append(packfeat[pt+numpy.array([0,1])])
        hCue.append([])       
        vCue.append(packfeat[pt+numpy.array([2,3])])
        vCue.append([])
        hCue2.append(packfeat[pt+numpy.array([4,5])])
        hCue2.append([])       
        vCue2.append(packfeat[pt+numpy.array([6,7])])
        vCue2.append([])
        pt=pt+8
        for l in range(2,cfg.lev):
            vCue[-1].append(packfeat[pt:pt+2])
            vCue[-1].append([])
            vCue[-1].append(packfeat[4+pt:pt+6])
            vCue[-1].append([])
            vCue[-1].append(packfeat[2+pt:pt+4])
            vCue[-1].append([])
            vCue[-1].append(packfeat[6+pt:pt+8])
            vCue[-1].append([])
            pt+=8
            hCue[-1].append(packfeat[pt:pt+2])
            hCue[-1].append([])
            hCue[-1].append(packfeat[4+pt:pt+6])
            hCue[-1].append([])
            hCue[-1].append(packfeat[2+pt:pt+4])
            hCue[-1].append([])
            hCue[-1].append(packfeat[6+pt:pt+8])
            hCue[-1].append([])
            pt+=8
            vCue2[-1].append(packfeat[pt:pt+2])
            vCue2[-1].append([])
            vCue2[-1].append(packfeat[2+pt:pt+4])
            vCue2[-1].append([])
            vCue2[-1].append(packfeat[4+pt:pt+6])
            vCue2[-1].append([])
            vCue2[-1].append(packfeat[6+pt:pt+8])
            vCue2[-1].append([])
            pt+=8
            hCue2[-1].append(packfeat[pt:pt+2])
            hCue2[-1].append([])
            hCue2[-1].append(packfeat[2+pt:pt+4])
            hCue2[-1].append([])
            hCue2[-1].append(packfeat[4+pt:pt+6])
            hCue2[-1].append([])
            hCue2[-1].append(packfeat[6+pt:pt+8])
            hCue2[-1].append([])
            pt+=8
        #pedro feat
        packfeat[pt:]=numpy.clip(packfeat[pt:],-1,-0.001)#numpy.abs(packfeat[pt:])
        pdef=[]
        pdef.append(packfeat[pt:pt+2])
        pdef.append([])
        pt+=2
        pdef.append(packfeat[pt:pt+2])
        pdef.append([])
        pt+=2
        pdef.append(packfeat[pt:pt+2])
        pdef.append([])
        pt+=2
        pdef.append(packfeat[pt:pt+2])
        pdef.append([])
        pt+=2
        for l in range(2,cfg.lev):
            pdef[1]=[]
            pdef[1].append(packfeat[pt:pt+2])
            pdef[1].append([])
            pt+=2
            pdef[1].append(packfeat[pt:pt+2])
            pdef[1].append([])
            pt+=2
            pdef[3]=[]
            pdef[3].append(packfeat[pt:pt+2])
            pdef[3].append([])
            pt+=2
            pdef[3].append(packfeat[pt:pt+2])
            pdef[3].append([])
            pt+=2
            pdef[1].append(packfeat[pt:pt+2])
            pdef[1].append([])
            pt+=2
            pdef[1].append(packfeat[pt:pt+2])
            pdef[1].append([])
            pt+=2
            pdef[3].append(packfeat[pt:pt+2])
            pdef[3].append([])
            pt+=2
            pdef[3].append(packfeat[pt:pt+2])
            pdef[3].append([])
            pt+=2
        
            pdef[5]=[]
            pdef[5].append(packfeat[pt:pt+2])
            pdef[5].append([])
            pt+=2
            pdef[5].append(packfeat[pt:pt+2])
            pdef[5].append([])
            pt+=2
            pdef[7]=[]
            pdef[7].append(packfeat[pt:pt+2])
            pdef[7].append([])
            pt+=2
            pdef[7].append(packfeat[pt:pt+2])
            pdef[7].append([])
            pt+=2
            pdef[5].append(packfeat[pt:pt+2])
            pdef[5].append([])
            pt+=2
            pdef[5].append(packfeat[pt:pt+2])
            pdef[5].append([])
            pt+=2
            pdef[7].append(packfeat[pt:pt+2])
            pdef[7].append([])
            pt+=2
            pdef[7].append(packfeat[pt:pt+2])
            pdef[7].append([])
            pt+=2
        delta=[hCue,vCue,hCue2,vCue2,pdef]
#    if delta==[]:
        print delta
        print packfeat[-40:]
        #raw_input()
        return feat,delta
    else:
        return feat



def packold(fparts,delta=[],obin=16,add=12):
    #for the moment delta is not considered
    totsize=0
    sz=[]
    auxdelta=[]
    if delta==[]:
        dspace=0
    else:
        auxdelta=numpy.zeros(delta.shape)
        if len(fparts)>1:
            auxdelta=delta
        dspace=3
    for p in fparts:
        sz.append(numpy.prod(p.shape[:-1])+dspace)
        totsize+=numpy.prod(p.shape[:-1])+dspace
    packfeat=numpy.zeros((p.shape[-1],totsize),numpy.float32)
    pt=0
    #to avoid to have features for the root filter
    for id,p in enumerate(fparts):
        if auxdelta==[]:
            packfeat[:,pt:pt+sz[id]]=p.T.reshape((p.shape[-1],sz[id]))
        else:
            #notice that I am using the abs of delta
            packfeat[:,pt:pt+sz[id]]=numpy.concatenate((p.T.reshape((p.shape[-1],sz[id]-dspace)),numpy.abs(auxdelta[id,:,:].T)),1)#p.T.reshape((p.shape[-1],sz[id]-3))
        pt=pt+sz[id]
    return packfeat


def packMKL(fparts,delta=[],obin=16,add=12):
    #for the moment delta is not considered
    totsize=0
    sz=[]
    auxdelta=[]
    if delta==[]:
        dspace=0
    else:
        auxdelta=numpy.zeros(delta.shape)
        if len(fparts)>1:
            auxdelta=delta
        dspace=3
    for p in fparts:
        sz.append(numpy.prod(p.shape[:-1])+dspace)
        totsize+=numpy.prod(p.shape[:-1])+dspace
    packfeat=numpy.zeros((p.shape[-1],totsize),numpy.float32)
    pt=0
    #to avoid to have features for the root filter
    for id,p in enumerate(fparts):
        if auxdelta==[]:
            packfeat[:,pt:pt+sz[id]]=p.T.reshape((p.shape[-1],sz[id]))
            sdf
        else:
            #notice that I am using the abs of delta
            packfeat[:,pt:pt+sz[id]]=numpy.concatenate((p.T.reshape((p.shape[-1],sz[id]-dspace)),numpy.abs(auxdelta[id,:,:].T)),1)#p.T.reshape((p.shape[-1],sz[id]-3))
        pt=pt+sz[id]
    return packfeat

##def packParts(fparts,delta=[],obin=16,add=12):
##    #for the moment delta is not considered
##    totsize=0
##    sz=[]
##    if delta==[]:
##        dspace=0
##    else:
##        dspace=3
##    for p in fparts:
##        sz.append(numpy.prod(p.shape[:-1])+dspace)
##        totsize+=numpy.prod(p.shape[:-1])+dspace
##    packfeat=numpy.zeros((p.shape[-1],totsize),numpy.float32)
##    pt=0
##    for id,p in enumerate(fparts):
##        if delta==[]:
##            packfeat[:,pt:pt+sz[id]]=p.T.reshape((p.shape[-1],sz[id]))
##        else:
##            #notice that I am using the abs of delta
##            packfeat[:,pt:pt+sz[id]]=numpy.concatenate((p.T.reshape((p.shape[-1],sz[id]-dspace)),numpy.abs(delta[id,:,:].T)),1)#p.T.reshape((p.shape[-1],sz[id]-3))
##        pt=pt+sz[id]
##    return packfeat

def unpack(packfeat,struct,delta=[],obin=16,add=12):
    #for the moment delta is not considered
    st=[]
    sz=[]
    feat=[]
    if delta==True:
        delta=numpy.zeros((len(struct),3))
        dspace=3
    else:
        dspace=0
    for p in struct:
        st.append(numpy.prod(p["sx"]*p["sy"]*(obin+add))+dspace)
        sz.append((p["sy"],p["sx"],(obin+add)))
        #totsize+=numpy.prod(sz[])
    pt=0
    for id,p in enumerate(st):
        #feat.append(packfeat[pt:pt+p-dspace].reshape(sz[id][2],sz[id][1],sz[id][0]).T)
        feat.append(packfeat[pt:pt+p-dspace].reshape(sz[id][2],sz[id][1],sz[id][0]).T)
        if delta!=[]:
            delta[id,:]=packfeat[pt+p-dspace:pt+p]
        pt+=p
    if delta==[]:
        return feat
    else:
        return feat,delta

def unpackPyrOld(packfeat,cfg,delta=[]):
    #for the moment delta is not considered
    st=[]
    sz=[]
    feat=[]
    if delta==True:
        delta=numpy.zeros((len(struct),3))
        dspace=3
    else:
        dspace=0
    for l in range(cfg.lev):
        st.append(numpy.prod(cfg.fx*2**l*cfg.fy*2**l*(cfg.ds))+dspace)
        sz.append((cfg.fy*2**l,cfg.fx*2**l,cfg.ds))
        #totsize+=numpy.prod(sz[])
    pt=0
    for id,p in enumerate(st):
        #feat.append(packfeat[pt:pt+p-dspace].reshape(sz[id][2],sz[id][1],sz[id][0]).T)
        feat.append(packfeat[pt:pt+p-dspace].reshape(sz[id][2],sz[id][1],sz[id][0]).T)
        if delta!=[]:
            delta[id,:]=packfeat[pt+p-dspace:pt+p]
        pt+=p
    if delta==[]:
        return feat
    else:
        return feat,delta


##def unpackParts(packfeat,struct,delta=[],obin=16,add=12):
##    #for the moment delta is not considered
##    st=[]
##    sz=[]
##    feat=[]
##    if delta==True:
##        delta=numpy.zeros((len(struct),3))
##        dspace=3
##    else:
##        dspace=0
##    for p in struct:
##        st.append(numpy.prod(p["sx"]*p["sy"]*(obin+add))+dspace)
##        sz.append((p["sy"],p["sx"],(obin+add)))
##        #totsize+=numpy.prod(sz[])
##    pt=0
##    for id,p in enumerate(st):
##        #feat.append(packfeat[pt:pt+p-dspace].reshape(sz[id][2],sz[id][1],sz[id][0]).T)
##        feat.append(packfeat[pt:pt+p-dspace].reshape(sz[id][2],sz[id][1],sz[id][0]).T)
##        if delta!=[]:
##            delta[id,:]=packfeat[pt+p-dspace:pt+p]
##        pt+=p
##    if delta==[]:
##        return feat
##    else:
##        return feat,delta

def drawGrid(img,pix=16,col="b"):
    """
        draw a grid in the image with pix size
    """
    for py in range(1,int(img.shape[0]/pix)+2):
        pylab.plot([0,(img.shape[1]/pix+2)*pix-pix],[py*pix-pix/2,py*pix-pix/2],col,lw=1)
    for px in range(1,int(img.shape[1]/pix)+2):
        pylab.plot([px*pix-pix/2,px*pix-pix/2],[0,(img.shape[0]/pix+2)*pix-pix],col,lw=1)

def drawGrid2(img,pix=16,col="b"):
    """
        draw a grid in the image with pix size
    """
    #pix=int(pix)
    for py in range(2,int(round(img.shape[0]/pix))+1):
        #horizontal lines
        pylab.plot([pix,round((img.shape[1])/pix)*pix-pix],[py*pix-pix,py*pix-pix],col,lw=1)
    for px in range(2,int(round(img.shape[1]/pix))+1):
        #vertical lines
        pylab.plot([px*pix-pix,px*pix-pix],[pix,round((img.shape[0])/pix)*pix-pix],col,lw=1)

def drawHOG(feat,obin=18,svm=None,sy=0,sx=0,vmin=None,vmax=None):
    """
        Draw the HOG features
    """
    k=1
    pylab.ioff()
    if svm==None:
        min=feat[:,:,:obin].min()
        max=feat[:,:,:obin].max()
    elif svm=="pos":
        min=0
        max=feat[:,:,:obin].max()
    elif svm=="neg":
        min=0
        k=-1
        max=-feat[:,:,:obin].min()
    if vmin!=None:
        min=vmin
    if vmax!=None:
        max=vmax
    d=max-min#feat[:,:,:obin].max()-feat[:,:,:obin].min()
    print d
    for y in range(feat.shape[0]):
        for x in range(feat.shape[1]):
            for l in range(obin):
                if feat[y,x,l]*k>0:
                    pylab.plot([sx+x,sx+x+0.4*numpy.cos(l*(2*numpy.pi)/obin)],[sy+y,sy+y+0.4*numpy.sin(l*(2*numpy.pi)/obin)],color=(abs(1.0-(k*feat[y,x,l]-min)/d),abs(1.0-(k*feat[y,x,l]-min)/d),abs(1.0-(k*feat[y,x,l]-min)/d)),lw=2)
        #plot([x,x+0.4*numpy.cos(l*(2*numpy.pi)/feat.shape[2])],[y,y+0.4*numpy.sin(l*(2*numpy.pi)/feat.shape[2])],color=(1-x,1-y,1-x))
        #plot([x,x+0.5],[y,y+0.5],"b")
    pylab.xticks([])
    pylab.yticks([])
    pylab.gca().set_ylim(pylab.gca().get_ylim()[::-1])
    #pylab.gca().set_xlim(pylab.gca().get_xlim()[::-1])
    pylab.axis("image")
    
def drawHOG9(mfeat,obin=18,svm=None,sy=0,sx=0,vmin=None,vmax=None):
    """
        Draw the HOG features
    """
    vv=1
    feat=(numpy.abs(mfeat[:,:,obin:obin+obin/2])+numpy.abs(mfeat[:,:,:obin/2])+numpy.abs(mfeat[:,:,obin/2:obin]))**2
    feat=feat/feat.sum()
    min=0
    max=feat.max()
    k=1
    pylab.ioff()
    d=feat.max()
    #print d
    for y in range(feat.shape[0]):
        for x in range(feat.shape[1]):
            sort=numpy.argsort(feat[y,x,:])
            for l in sort:#range(obin/2):
                if feat[y,x,l]*k>0:
                    pylab.plot([sx+x,sx+x+0.4*numpy.cos(l*(2*numpy.pi)/obin-numpy.pi/2)],[sy+y,sy+y+0.4*numpy.sin(l*(2*numpy.pi)/obin-numpy.pi/2)],color=(abs(1.0-(k*feat[y,x,l]-min)/d)**vv,abs(1.0-(k*feat[y,x,l]-min)/d)**vv,abs(1.0-(k*feat[y,x,l]-min)/d)**vv),lw=2)
                    pylab.plot([sx+x,sx+x+0.4*numpy.cos(l*(2*numpy.pi)/obin+numpy.pi-numpy.pi/2)],[sy+y,sy+y+0.4*numpy.sin(l*(2*numpy.pi)/obin+numpy.pi-numpy.pi/2)],color=(abs(1.0-(k*feat[y,x,l]-min)/d)**vv,abs(1.0-(k*feat[y,x,l]-min)/d)**vv,abs(1.0-(k*feat[y,x,l]-min)/d)**vv),lw=2)
        #plot([x,x+0.4*numpy.cos(l*(2*numpy.pi)/feat.shape[2])],[y,y+0.4*numpy.sin(l*(2*numpy.pi)/feat.shape[2])],color=(1-x,1-y,1-x))
        #plot([x,x+0.5],[y,y+0.5],"b")
    pylab.xticks([])
    pylab.yticks([])
    pylab.gca().set_ylim(pylab.gca().get_ylim()[::-1])
    #pylab.gca().set_xlim(pylab.gca().get_xlim()[::-1])
    pylab.axis("image")


def drawDef(dfeat,dy,dx,mindef=0.001,distr="father"):
    from matplotlib.patches import Ellipse
    #kd=100
    #dx=-dx
    #dy=-dy
    pylab.ioff()
    if distr=="father":
        py=[0,0,2,2];px=[0,2,0,2]
    if distr=="child":
        py=[0,1,1,2];px=[1,2,0,1]
    ordy=[0,0,1,1];ordx=[0,1,0,1]
    #cc=0
    x1=-0.5+dx;x2=2.5+dx
    y1=-0.5+dy;y2=2.5+dy
    if distr=="father":       
        pylab.fill([x1,x1,x2,x2,x1],[y1,y2,y2,y1,y1],"r", alpha=0.15, edgecolor="b",lw=1)    
    for l in range(len(py)):
        aux=dfeat[ordy[l],ordx[l],:].clip(-1,-mindef)
        wh=numpy.exp(-mindef/aux[0])/numpy.exp(1);hh=numpy.exp(-mindef/aux[1])/numpy.exp(1)
        #print "Dx",wh,"Dy",hh
        e=Ellipse(xy=[(px[l]+dx),(py[l]+dy)], width=wh, height=hh, alpha=0.35)
        x1=-0.75+dx+px[l];x2=0.75+dx+px[l]
        y1=-0.76+dy+py[l];y2=0.75+dy+py[l]
        col=numpy.array([wh*hh]*3).clip(0,1)
        if distr=="father":
            col[0]=0       
        e.set_facecolor(col)
        pylab.gca().add_artist(e)
        if distr=="father":       
            pylab.fill([x1,x1,x2,x2,x1],[y1,y2,y2,y1,y1],"b", alpha=0.15, edgecolor="b",lw=1)            
        #cc=cc+1

def drawDeform(dfeat,mindef=0.001):
    from matplotlib.patches import Ellipse
    lev=len(dfeat)
    if 1:
        sy=1
        sx=lev
    else:
        sy=lev
        sx=1
    pylab.subplot(sy,sx,1)
    x1=-0.5;x2=0.5
    y1=-0.5;y2=0.5
    pylab.fill([x1,x1,x2,x2,x1],[y1,y2,y2,y1,y1],"b", alpha=0.15, edgecolor="b",lw=1) 
    pylab.fill([x1,x1,x2,x2,x1],[y1,y2,y2,y1,y1],"r", alpha=0.15, edgecolor="r",lw=1)       
    wh=numpy.exp(-mindef/dfeat[0][0,0,0])/numpy.exp(1);hh=numpy.exp(-mindef/dfeat[0][0,0,1])/numpy.exp(1)
    e=Ellipse(xy=[0,0], width=wh, height=hh , alpha=0.35)
    col=numpy.array([wh*hh]*3).clip(0,1)
    col[0]=0
    e.set_facecolor(col)
    pylab.axis("off")
    pylab.gca().add_artist(e)
    pylab.gca().set_ylim(-0.5,0.5)
    pylab.gca().set_xlim(-0.5,0.5)
    for l in range(1,lev):
        pylab.subplot(sy,sx,l+1)
        for ry in range(2**(l-1)):
            for rx in range(2**(l-1)): 
                drawDef(dfeat[l][ry*2:(ry+1)*2,rx*2:(rx+1)*2,2:]*4**l,4*ry,4*rx,distr="child")
                drawDef(dfeat[l][ry*2:(ry+1)*2,rx*2:(rx+1)*2,:2]*4**l,ry*2**(l),rx*2**(l),mindef=mindef,distr="father")
        #pylab.gca().set_ylim(-0.5,(2.6)**l)
        pylab.axis("off")
        pylab.gca().set_ylim((2.6)**l,-0.5)
        pylab.gca().set_xlim(-0.5,(2.6)**l)

def drawModel(mfeat,mode="black"):
    import drawHOG
    lev=len(mfeat)
    if mfeat[0].shape[0]>mfeat[0].shape[1]:
        sy=1
        sx=lev
    else:
        sy=lev
        sx=1
    for l in range(lev):
        pylab.subplot(sy,sx,l+1)
        if mode=="white":
            drawHOG9(mfeat[l])
        elif mode=="black":
            img=drawHOG.drawHOG(mfeat[l])
            pylab.axis("off")
            pylab.imshow(img,cmap=pylab.cm.gray,interpolation="nearest")

def drawContrast(feat,obin=18,svm=None):
    im=numpy.zeros((feat.shape[0]*2,feat.shape[1]*2))
    im[::2,::2]=feat[:,:,obin+obin/2]
    im[1::2,::2]=feat[:,:,obin+obin/2+1]
    im[::2,1::2]=feat[:,:,obin+obin/2+2]
    im[1::2,1::2]=feat[:,:,obin+obin/2+3]
    pylab.imshow(im,interpolation="nearest")

def drawHOGmodel(feat,delta,model,obin=16,svm=None):
    """
        Draw the HOG model
    """
    if delta==[]:
        for c,p in enumerate(model):
            drawHOG(feat[c],obin,svm,sy=p["y"],sx=p["x"])
            box(p["y"]-0.5,p["x"]-0.5,p["y"]+p["sy"]-0.5,p["x"]+p["sx"]-0.5)
    else:
        #if model["sy"]>model["sx"]:
        if maxdimy(model)>maxdimx(model):
            pylab.subplot(1,2,1)
        else:
            pylab.subplot(2,1,1)
        for c,p in enumerate(model):
            drawHOG(feat[c],obin,svm,sy=p["y"],sx=p["x"])
            box(p["y"]-0.5,p["x"]-0.5,p["y"]+p["sy"]-0.5,p["x"]+p["sx"]-0.5)
        if maxdimy(model)>maxdimx(model):
            pylab.subplot(1,2,2)
        else:
            pylab.subplot(2,1,2)
        for c in range(delta.shape[0]):
            pylab.arrow(float(model[c][0]["x"]+model[c][0]["sx"]/2.0-0.5),float(model[c][0]["y"]+model[c][0]["sy"]/2.0-0.5),1+delta[c,2]*10,.0,width=0.03)
            pylab.arrow(float(model[c][0]["x"]+model[c][0]["sx"]/2.0-0.5),float(model[c][0]["y"]+model[c][0]["sy"]/2.0-0.5),-(1+delta[c,2]*10),.0,width=0.03)
            pylab.arrow(float(model[c][0]["x"]+model[c][0]["sx"]/2.0-0.5),float(model[c][0]["y"]+model[c][0]["sy"]/2.0-0.5),.0,1+delta[c,1]*10,width=0.03)
            pylab.arrow(float(model[c][0]["x"]+model[c][0]["sx"]/2.0-0.5),float(model[c][0]["y"]+model[c][0]["sy"]/2.0-0.5),.0,-(1+delta[c,1]*10),width=0.03)
            box(model[c][0]["y"]-0.5,model[c][0]["x"]-0.5,model[c][0]["y"]+model[c][0]["sy"]-0.5,model[c][0]["x"]+model[c][0]["sx"]-0.5)
    pylab.xticks([])
    pylab.yticks([])
    pylab.gca().set_ylim(pylab.gca().get_ylim()[::-1])
    #pylab.gca().set_xlim(pylab.gca().get_xlim()[::-1])
    pylab.axis("image")

def drawHOGmodelPyrold(feat,delta=[],svm=None,type=2,kw=[]):
    """
        Draw the HOG model
    """
    if delta==[]:
        vmin=numpy.zeros(len(feat))
        vmax=numpy.zeros(len(feat))
        for id,f in enumerate(feat):
            vmin[id]=f.min()
            vmax[id]=f.max()
        nx=len(feat)
        ny=1
        if feat[0].shape[0]<feat[0].shape[1]:
            nx=1
            ny=len(feat)
        #pylab.figure()
        for id,f in enumerate(feat):
            #pylab.figure()
            pylab.subplot(ny,nx,id+1)
            if type==1:
                drawHOG(f,svm=svm,vmin=0,vmax=vmax.max())
            else:
                drawHOG2(f,svm=svm,vmin=0,vmax=vmax.max())
            if kw!=[]:
                pylab.title("%.3f"%kw[id])
            #box(p["y"]-0.5,p["x"]-0.5,p["y"]+p["sy"]-0.5,p["x"]+p["sx"]-0.5)
    else:
        #if model["sy"]>model["sx"]:
        if maxdimy(model)>maxdimx(model):
            pylab.subplot(1,2,1)
        else:
            pylab.subplot(2,1,1)
        for c,p in enumerate(model):
            drawHOG(feat[c],obin,svm,sy=p["y"],sx=p["x"])
            box(p["y"]-0.5,p["x"]-0.5,p["y"]+p["sy"]-0.5,p["x"]+p["sx"]-0.5)
        if maxdimy(model)>maxdimx(model):
            pylab.subplot(1,2,2)
        else:
            pylab.subplot(2,1,2)
        for c in range(delta.shape[0]):
            pylab.arrow(float(model[c][0]["x"]+model[c][0]["sx"]/2.0-0.5),float(model[c][0]["y"]+model[c][0]["sy"]/2.0-0.5),1+delta[c,2]*10,.0,width=0.03)
            pylab.arrow(float(model[c][0]["x"]+model[c][0]["sx"]/2.0-0.5),float(model[c][0]["y"]+model[c][0]["sy"]/2.0-0.5),-(1+delta[c,2]*10),.0,width=0.03)
            pylab.arrow(float(model[c][0]["x"]+model[c][0]["sx"]/2.0-0.5),float(model[c][0]["y"]+model[c][0]["sy"]/2.0-0.5),.0,1+delta[c,1]*10,width=0.03)
            pylab.arrow(float(model[c][0]["x"]+model[c][0]["sx"]/2.0-0.5),float(model[c][0]["y"]+model[c][0]["sy"]/2.0-0.5),.0,-(1+delta[c,1]*10),width=0.03)
            box(model[c][0]["y"]-0.5,model[c][0]["x"]-0.5,model[c][0]["y"]+model[c][0]["sy"]-0.5,model[c][0]["x"]+model[c][0]["sx"]-0.5)
    pylab.xticks([])
    pylab.yticks([])
    #pylab.gca().set_ylim(pylab.gca().get_xlim()[::-1])
    #pylab.axis("image")

def drawHOGmodelPyr(feat,delta=[],svm=None,type=1,kw=[]):
    """
        Draw the HOG model
    """
    if delta==[]:
        vmin=numpy.zeros(len(feat))
        vmax=numpy.zeros(len(feat))
        for id,f in enumerate(feat):
            vmin[id]=f.min()
            vmax[id]=f.max()
        nx=len(feat)
        ny=1
        if feat[0].shape[0]<feat[0].shape[1]:
            nx=1
            ny=len(feat)
        #pylab.figure()
        for id,f in enumerate(feat):
            #pylab.figure()
            pylab.subplot(ny,nx,id+1)
            if type==1:
                #drawHOG(f,svm=svm,vmin=0,vmax=vmax.max())
                drawHOG9(f)
            else:
                #drawHOG2(f,svm=svm,vmin=0,vmax=vmax.max())
                drawHOG9(f)
            if kw!=[]:
                pylab.title("%.3f"%kw[id])
            #box(p["y"]-0.5,p["x"]-0.5,p["y"]+p["sy"]-0.5,p["x"]+p["sx"]-0.5)
    else:
        from matplotlib.patches import Ellipse
        vmin=numpy.zeros(len(feat))
        vmax=numpy.zeros(len(feat))
        for id,f in enumerate(feat):
            vmin[id]=f.min()
            vmax[id]=f.max()
        nx=len(feat)
        ny=1
        if feat[0].shape[0]<feat[0].shape[1]:
            nx=1
            ny=len(feat)
        #pylab.figure()
        nw=0
        for id in range(len(feat)):
            nw+=numpy.sum(numpy.abs(feat[id]))
        #pylab.title("Norm W:%f"%nw)
        pylab.subplots_adjust(left=0.1,right=0.9,bottom=0.1,top=0.9,wspace=0.4,hspace=0.4)
        for id,f in enumerate(feat):
            py=f.shape[0]/(2**id)
            px=f.shape[1]/(2**id)
            pylab.subplot(ny,nx,id+1)
            for divy in range(2**id):
                for divx in range(2**id):
            #pylab.figure()
                    if type==1:
                        if id>0:
                            f1=f#.copy()
                            #f1=f[::-1,:,:]
                            #f1[:,:,:9]=f[:,:,8::-1]#[::-1,:,:]
                            #f1=f1[::-1,:,:]
                        else:
                            f1=f
                        #drawHOG(f1[py*(divy):max(py*(divy+1),0),px*divx:px*(divx+1)],svm=svm,vmin=0,vmax=vmax.max(),sy=(py+2)*divy,sx=(px+2)*divx)
                        drawHOG9(f1[py*(divy):max(py*(divy+1),0),px*divx:px*(divx+1)],sy=(py+2)*divy,sx=(px+2)*divx)
                        box((py+2)*divy-0.5,(px+2)*divx-0.5,(py+2)*(divy+1)-2.5,(px+2)*(divx+1)-2.5,lw=1.0,col="black")
                    #else:
                    #    drawHOG2(f[:,:],svm=svm,vmin=0,vmax=vmax.max())
                    if kw!=[]:
                        pylab.title("%.3f"%kw[id])
            #box(p["y"]-0.5,p["x"]-0.5,p["y"]+p["sy"]-0.5,p["x"]+p["sx"]-0.5)
            #pylab.subplot(ny,nx,id+1)
            #if id==1:
            #    for divy in range(2**id-1):
            #        for divx in range(2**id):
            #            hh=2-100*delta[0][id-1][0]
            #            wh=2-100*delta[1][id-1][1]
            #            e=Ellipse(xy=[(px+2)*(divx)+(px/2.0)-0.5,(py+2)*(divy)+py+0.5], width=wh, height=hh)
            #            e.set_facecolor([max(0,wh/2.0*hh/2.0)]*3)
            #            pylab.gca().add_artist(e)
            #    for divy in range(2**id):
            #        for divx in range(2**id-1):
            #            hh=2-100*delta[2][id-1][0]
            #            wh=2-100*delta[3][id-1][1]
            #            e=Ellipse(xy=[(px+2)*(divx)+px+0.5,(py+2)*(divy)+(py/2.0)-0.5], width=wh, height=hh)
            #            e.set_facecolor([max(0,wh/2.0*hh/2.0)]*3)
            #            pylab.gca().add_artist(e)
            sd=0
            if id==1:
                drawdef(delta[0][0],delta[1][0],delta[2][0],delta[3][0],py,px,0,0)
                sd+=delta[0][0][0]+delta[1][0][0]+delta[2][0][0]+delta[3][0][0]
                sd+=delta[0][0][1]+delta[1][0][1]+delta[2][0][1]+delta[3][0][1]
            if id==2:
                for x in range(2):
                    for y in range(2):
                        print f.shape[0]
                        drawdef(delta[0][1][2*(x+y*2)],delta[1][1][2*(x+y*2)],delta[2][1][2*(x+y*2)],delta[3][1][2*(x+y*2)],py,px,((2*py+4))*1-((2*py+4))*(y),(2*px+4)*x)
                        sd+=delta[0][1][2*(x+y*2)][0]+delta[1][1][2*(x+y*2)][0]+delta[2][1][2*(x+y*2)][0]+delta[3][1][2*(x+y*2)][0]
                        sd+=delta[0][1][2*(x+y*2)][1]+delta[1][1][2*(x+y*2)][1]+delta[2][1][2*(x+y*2)][1]+delta[3][1][2*(x+y*2)][1]
            app=numpy.sum(numpy.abs(feat[id]))
            pylab.title("Norm W: %.3f Tot: %.3f \nApp: %.3f %.3f Def: %.3f %.3f\n"%(nw,app+sd,app,app/(app+sd)*100,sd,sd/(app+sd)*100),fontsize='small')                
            if id>0:
                pylab.gca().set_ylim(pylab.gca().get_ylim()[::-1])
                #pylab.gca().set_ylim
    pylab.xticks([])
    pylab.yticks([])
    #pylab.gca().set_ylim(pylab.gca().get_xlim()[::-1])
    #pylab.axis("image")

def drawPedroDef(pdef):
    """
        Draw deformation
    """
    pylab.figure(120)
    pylab.clf()
    if pdef[1]!=[]:
        pylab.subplot(2,1,1)
    drawpedrodef(pdef,0,0)
    pylab.gca().set_ylim([-1,3])
    pylab.gca().set_xlim([-1,3])
    if pdef[1]!=[]:
        pylab.subplot(2,1,2)
        drawpedrodef(pdef[1],0,0)
        drawpedrodef(pdef[3],0,2)
        drawpedrodef(pdef[5],2,0)
        drawpedrodef(pdef[7],2,2)
        pylab.gca().set_ylim([-1,7])
        pylab.gca().set_xlim([-1,7])
    pylab.show()
    pylab.draw()


def drawpedrodef(pdef,dy,dx):
    from matplotlib.patches import Ellipse
    #kd=100
    #dx=-dx
    #dy=-dy
    cc=0
    for py in range(2):
        for px in range(2):
            wh=numpy.exp(-0.001/pdef[2*cc][0])/numpy.exp(1);hh=numpy.exp(-0.001/pdef[2*cc][1])/numpy.exp(1)
            e=Ellipse(xy=[2*(px+dx),2*(py+dy)], width=wh, height=hh)
            e.set_facecolor([wh*hh]*3)
            pylab.gca().add_artist(e)
            cc=cc+1


def drawdef(v1,h1,v2,h2,dimy,dimx,py,px,kd=100):
    from matplotlib.patches import Ellipse
    #wh=max(2-kd*v1[1],0);hh=max(2-kd*h1[1],0)
    wh=-0.002/v1[1];hh=-0.002/h1[1]
    e=Ellipse(xy=[px+dimx+0.5,py+dimy/2.0-0.5], width=wh, height=hh)
    e.set_facecolor([wh/2.0*hh/2.0]*3)
    pylab.gca().add_artist(e)
    #wh=max(2-kd*v1[0],0);hh=max(2-kd*h1[0],0)
    wh=-0.002/v1[0];hh=-0.002/h1[0]
    e=Ellipse(xy=[px+dimx+0.5,py+dimy+2+dimy/2.0-0.5], width=wh, height=hh)
    #e.set_facecolor([max(min(1,wh/2.0*hh/2.0),0)]*3)
    e.set_facecolor([wh/2.0*hh/2.0]*3)
    pylab.gca().add_artist(e)
    #wh=max(2-kd*v2[0],0);hh=max(2-kd*h2[0],0)
    wh=-0.002/v2[0];hh=-0.002/h2[0]
    e=Ellipse(xy=[px+dimx/2.0-0.5,py+dimy+1.5-1], width=wh, height=hh)
    #e.set_facecolor([max(min(1,wh/2.0*hh/2.0),0)]*3)
    e.set_facecolor([wh/2.0*hh/2.0]*3)
    pylab.gca().add_artist(e)
    #wh=max(2-kd*v2[1],0);hh=max(2-kd*h2[1],0)
    wh=-0.002/v2[1];hh=-0.002/h2[1]
    e=Ellipse(xy=[px+dimx/2.0+2+dimx-0.5,py+dimy+1.5-1], width=wh, height=hh)
    #e.set_facecolor([max(min(1,wh/2.0*hh/2.0),0)]*3)
    e.set_facecolor([wh/2.0*hh/2.0]*3)
    pylab.gca().add_artist(e)
    pylab.show()
    pylab.draw()


def drawHOG2(feat1,obin=18,opt=None,svm=None,vmin=None,vmax=None):
    """
    Draw the hog features in a faster way
    """
    if vmax==None:
        vmax=1
    if vmin==None:
        vmin=0
    if opt==None:
        feat2=feat1[:,:,:obin]
        phi=0
    if opt=="img":
        feat2=(feat1[:,:,0:obin/2]+feat1[:,:,obin/2:obin])/2
        #obin=2*obin
        phi=numpy.pi/2
    if opt=="imgn":
        feat2=(feat1[:,:,0:obin/2]+feat1[:,:,obin/2:obin])/2
        #obin=2*obin
        vmin=feat1.min()
        vmax=feat1.max()
        phi=numpy.pi/2
    feat=numpy.zeros((feat2.shape[0]+1,feat2.shape[1]+1,feat2.shape[2]))
    feat[-1,-1,:]=numpy.ones(feat2.shape[2])*vmax
    feat[:-1,:-1]=feat2
    dimy=feat.shape[1]
    dimx=feat.shape[0]
    my=numpy.zeros((dimy,dimx))
    mx=numpy.zeros((dimy,dimx))
    for l in range(feat.shape[2]):
        #if feat[y,x,l]>0:
        #pylab.plot([x,x+0.4*numpy.cos(l*(2*numpy.pi)/obin)],[y,y+0.4*numpy.sin(l*(2*numpy.pi)/obin)],color=(1.0-(feat[y,x,l]-min)/d,1.0-(feat[y,x,l]-min)/d,1.0-(feat[y,x,l]-min)/d),lw=2)
        my=numpy.ones((dimy,dimx))*0.4*numpy.sin(l*(2*numpy.pi)/obin+phi)
        mx=-numpy.ones((dimy,dimx))*0.4*numpy.cos(l*(2*numpy.pi)/obin+phi)
        col=vmax-feat[:,:,l]+vmin
        pylab.quiver(mx,my,col,headwidth=1,headlength=0,units='x',scale=1, cmap=pylab.cm.gray,lw=0)
        if opt=="img" or opt=="imgn":
            pylab.quiver(-mx,-my,col,headwidth=1,headlength=0,units='x',scale=1, cmap=pylab.cm.gray,lw=0)
    pylab.xticks([])
    pylab.yticks([])
    pylab.gca().set_ylim(dimy-0.5,-0.5)
    pylab.gca().set_xlim(-0.5,dimx-0.5)
    #pylab.axis('equal')
    #pylab.gca().set_ylim(pylab.gca().get_xlim()[::-1])
    pylab.axis("image")
    pylab.gca().set_ylim(dimx-1.5,-0.5)
    pylab.gca().set_xlim(-0.5,dimy-1.5)
    
def filter(nfeat,ww,rho):
    res=numpy.zeros((nfeat[0].shape[3],len(ww)))
    #sel=[]
    #resl=[]
    for m in range(len(ww)):
        for id,p in enumerate(nfeat):
            res[:,m]+=numpy.sum(numpy.sum(numpy.sum(p[:,:,:,:].T*ww[m][id][:,:,:].T,1),1),1)
        res[:,m]-=rho[m]
    #    sel.append(numpy.logical_and(res<1,res>-1))
    bres=numpy.max(res,1)
    bcl=numpy.argmax(res,1)
    #sel1=numpy.logical_and(bres<=1,bres>=-1)# old version until 19/01/2010
    sel=bres>=-1
    #print bres
    #print "Hard Negative:",numpy.sum(sel>0),numpy.sum(sel1>0)
    #raw_input()
    return sel,bcl

def filter2(nfeat,ww,rho):
    res=numpy.zeros((nfeat[0].shape[3]))
    for id,p in enumerate(nfeat):
        res=res+numpy.sum(numpy.sum(numpy.sum(p[:,:,:,:].T*ww[id][:,:,:].T,1),1),1)
    res=res-rho
    sel=res>=-1
    print "Scores:",res
    #print bres
    #print "Hard Negative:",numpy.sum(sel>0),"/",nfeat[0].shape[3]
    #raw_input()
    return sel


def prepareSV(alpha,sv):
    sortsv=numpy.sort((sv.T*alpha).T,0)
    sumsv=numpy.cumsum(sortsv,0)
    return sortsv,sumsv

def kernelFast(v,sortsv,cumsv):
    v=v.reshape((1,v.size))
    #vlen=v.shape[1]
    numsv=sortsv.shape[0]
    r=numpy.zeros(v.shape)
    for col in range(sortsv.shape[1]):
        pos=numpy.searchsorted(sortsv[:,col],v[0,col],"left")
        r[0,col]=cumsv[min(pos,numsv-1),col]+v[0,col]*(numsv-pos-1)
        #r[0,col]=cumsv[pos,col]+v[0,col]*(vlen-pos)
        #print vlen,pos
    return r

def kernel(v,alpha,sv):
    aux=numpy.zeros((2,v.size))
    aux[0,:]=v
    r=numpy.zeros(v.shape)
    for l in range(sv.shape[0]):
        aux[1,:]=sv[l,:]
        r+=aux.min(0)*alpha[l]
        #print aux.min(0)
    return r

def toint8(a):
    return (a*512).clip(0,255).astype(numpy.uint8)

def fromint8(a):
    return a.astype(numpy.float32)/512.0
    
def objf(w,b,pos,neg,c):
    m=pos.shape[0]+neg.shape[0]
    errp=numpy.maximum(0,1-numpy.sum(pos*w,1)-b)
    errn=numpy.maximum(0,1+numpy.sum(neg*w,1)-b)
    print errp,errn
    mw=0.5*numpy.sqrt(numpy.sum((w*w)))
    tote=(numpy.sum(errp)+numpy.sum(errn))/float(m)
    f=mw+c*tote
    print "w",mw,"err",tote,"f",f
    return f

def showExamples(ex,fy,fx,w=None,r=None):
    import drawHOG
    n=len(ex)
    for l in range(n):
        pylab.subplot(int(n**0.5),int((n**0.5))+1,l+1)
        img=drawHOG.drawHOG(ex[l][:fy*fx*31].reshape(fy,fx,31))
        pylab.axis("off")
        if w!=None:
            pylab.text(0,0,"%.3f"%(numpy.sum(ex[l]*w)-r))
        pylab.imshow(img,cmap=pylab.cm.gray,interpolation="nearest")



    

