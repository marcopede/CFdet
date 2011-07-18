import util
import pyrHOG2
import pyrHOG2RL
import time
import pylab
    
if __name__=="__main__":

    import sys
    if len(sys.argv)>1:
        cls=sys.argv[1]
    else:
        cls="car"

    if cls=="all":
        cls=["aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair","cow","diningtable",
                "dog","horse","motorbike","person","pottedplant","sheep","sofa","train","tvmonitor"]
    else:
        cls=[cls]

    if len(sys.argv)>2:
        imname=sys.argv[2]
    else:
        imname="000004.jpg"
    
    #configuration class
    class config(object):
        pass
    cfg=config()
    cfg.testname="./data/finalRL/%s2_test"
    cfg.resize=1.0
    cfg.hallucinate=True
    cfg.initr=1
    cfg.ratio=1
    cfg.deform=True
    #cfg.show=False
    cfg.bottomup=False
    cfg.usemrf=True

    #read the image
    img=util.myimread(imname,resize=cfg.resize)    
    #compute the hog pyramid
    f=pyrHOG2.pyrHOG(img,interv=10,savedir="",notsave=True,notload=True,hallucinate=cfg.hallucinate,cformat=True)
    #show the image
    fig=pylab.figure(20)
    pylab.ioff()
    axes=pylab.Axes(fig, [.0,.0,1.0,1.0]) # [left, bottom, width, height] where each value is between 0 and 1
    fig.add_axes(axes) 
    pylab.imshow(img,interpolation="nearest",animated=True)
        
    t=time.time()
    #for each class
    for ccls in cls:
        print
        print "Class: %s"%ccls
        #load the class model
        m=util.load("%s%d.model"%(cfg.testname%ccls,7))
        res=[]
        t1=time.time()
        #for each aspect
        for clm,m in enumerate(m):
            #scan the image with left and right models
            res.append(pyrHOG2RL.detectflip(f,m,None,hallucinate=cfg.hallucinate,initr=cfg.initr,ratio=cfg.ratio,deform=cfg.deform,bottomup=cfg.bottomup,usemrf=cfg.usemrf,small=False,cl=clm))
        fuse=[]
        numhog=0
        #fuse the detections
        for mix in res:
            tr=mix[0]
            fuse+=mix[1]
            numhog+=mix[3]
        rfuse=tr.rank(fuse,maxnum=300)
        nfuse=tr.cluster(rfuse,ovr=0.3,inclusion=False)
        pylab.axis("off")
        #show detections
        tr.show(nfuse,parts=True,thr=-0.99,scr=True,maxnum=2,cls=ccls)
        pylab.axis((0,img.shape[1],img.shape[0],0))
        print "Number of computed HOGs:",numhog
        #print "Extra time",time.time()-t1
        print "Elapsed time",time.time()-t
        pylab.ion()
        pylab.draw()
    pylab.show()  



