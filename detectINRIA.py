#import matplotlib
#matplotlib.use("Agg") #if run out of ipython do not show any graph
import util
import pyrHOG2
import time
import itertools

class config(object):
    pass

def detectWrap(a):
    t=time.time()
    i=a[0]
    imname=a[1]
    bbox=a[2]
    m=a[3]
    cfg=a[4]
    if cfg.show:
        import pylab
        img=util.myimread(imname)
        pylab.figure(10)
        pylab.ioff()
        pylab.clf()
        axes=pylab.Axes(pylab.gcf(), [.0,.0,1.0,1.0])
        pylab.gcf().add_axes(axes) 
        pylab.axis("off")
        pylab.imshow(img,interpolation="nearest",animated=True) 
        #pylab.axis((0,img.shape[1],img.shape[0],0))
    if bbox!=None:
        gtbbox=[{"bbox":x} for x in bbox]   
    else:
        gtbbox=None
    f=pyrHOG2.pyrHOG(imname,interv=10,savedir=cfg.savedir+"/hog/",notload=not(cfg.loadfeat),notsave=not(cfg.savefeat),hallucinate=cfg.hallucinate,cformat=True)
    res=pyrHOG2.detect(f,m,gtbbox,hallucinate=cfg.hallucinate,initr=cfg.initr,ratio=cfg.ratio,deform=cfg.deform,posovr=cfg.posovr,bottomup=cfg.bottomup,usemrf=cfg.usemrf,numneg=cfg.numneg,thr=cfg.thr,inclusion=cfg.inclusion,small=cfg.small,show=cfg.show,usefather=cfg.usefather,useprior=cfg.useprior,nms=cfg.ovr,K=cfg.k)
    if cfg.show:
        pylab.axis((0,img.shape[1],img.shape[0],0))
        pylab.draw()
        pylab.show()
        #raw_input()
    #print "Detect Wrap:",time.time()-t
    return res

testname="./data/INRIA/inria_bothfull";it=6
import sys
if len(sys.argv)>1:
    imname=sys.argv[1]
else:
    imname="000061.jpg"
    #it=int(sys.argv[1])
cfg=util.load(testname+".cfg")
cfg.maxtest=1000#100
cfg.numneg=1000
cfg.ovr=0.5
cfg.posovr=0.7
cfg.thr=-2
#cfg.deform=True
cfg.bottomup=False
cfg.loadfeat=False
cfg.savefeat=False
cfg.multipr=4
cfg.inclusion=False
cfg.dbpath="/home/marcopede/databases/"
cfg.auxdir=""
cfg.show=True
cfg.mythr=-10
cfg.small=False
cfg.useprior=False
cfg.k=1.0

m=util.load("%s%d.model"%(testname,it))

if 0:
    print "Show model"
    pylab.figure(100)
    pylab.clf()
    util.drawModel(m["ww"])
    pylab.figure(101)
    pylab.clf()
    util.drawDeform(m["df"])
    pylab.draw()
    pylab.show()

res=detectWrap([0,imname,None,m,cfg])
print "---- Image %s----"%imname
#print "Detections:", len(res[1]) 
dettime=res[2]
numhog=res[3]
#print "Detection Time:",dettime
print "Number of computed HOGs:",numhog
#print "Press a key..."
#raw_input()



