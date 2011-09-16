import util2
import pyrHOG2
import pyrHOG2RL
import time
import pylab
    
if __name__=="__main__":

    import sys
    #first argument: object class
    if len(sys.argv)>1:
        cls=sys.argv[1]
    else:
        cls="car"

    if cls=="all":
        cls=["aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair","cow","diningtable",
                "dog","horse","motorbike","person","pottedplant","sheep","sofa","train","tvmonitor"]
    else:
        cls=[cls]

    #second argument: image name
    if len(sys.argv)>2:
        imname=sys.argv[2]
    else:
        imname="test2.png"
    
    #configuration class
    class config(object):
        pass
    cfg=config()
    cfg.testname="./data/finalRL/%s2_test"  #object model
    cfg.resize=1.0                          #resize the input image
    cfg.hallucinate=True                    #use HOGs up to 4 pixels
    cfg.initr=1                             #initial radious of the CtF search
    cfg.ratio=1                             #radious at the next levels
    cfg.deform=True                         #use deformation
    cfg.bottomup=False                      #use complete search
    cfg.usemrf=True                         #use lateral constraints

    #read the image
    img=util2.myimread(imname,resize=cfg.resize)    
    #compute the hog pyramid
    f=pyrHOG2.pyrHOG(img,interv=10,savedir="",notsave=True,notload=True,hallucinate=cfg.hallucinate,cformat=True)
    #show the image
    fig=pylab.figure(20)
    pylab.ioff()
    axes=pylab.Axes(fig, [.0,.0,1.0,1.0]) 
    fig.add_axes(axes) 
    pylab.imshow(img,interpolation="nearest",animated=True)
        
    t=time.time()
    #for each class
    for ccls in cls:
        print
        print "Class: %s"%ccls
        #load the class model
        m=util2.load("%s%d.model"%(cfg.testname%ccls,7))
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
        tr.show(nfuse,parts=True,thr=-0.98,scr=True,maxnum=10,cls=ccls)
        pylab.axis((0,img.shape[1],img.shape[0],0))
        print "Number of computed HOGs:",numhog
        print "Elapsed time: %.3f s"%(time.time()-t)
        pylab.ion()
        pylab.draw()
    pylab.show()  



