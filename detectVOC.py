#import matplotlib
#matplotlib.use("Agg") #if run out of ipython do not show any graph
#from multiprocessing import Pool
import util
import pyrHOG2
import pyrHOG2RL
#import VOCpr
import time
import copy
import itertools
#from trainPASCALkeep2RL import *

def detectWrap(a):
    i=a[0]
    imname=a[1]
    bbox=a[2]
    models=a[3]
    cfg=a[4]
    if len(a)<=5:
        imageflip=False
    else:
        imageflip=a[5]
    img=util.myimread(imname,resize=cfg.resize)
    if imageflip:
        img=util.myimread(imname,True,resize=cfg.resize)
        if bbox!=None:
             bbox = util.flipBBox(img,bbox)
    if bbox!=None:
        gtbbox=[{"bbox":x,"img":imname.split("/")[-1]} for x in bbox]   
    else:
        gtbbox=None
    if cfg.show:
        import pylab
        img=util.myimread(imname,imageflip,resize=cfg.resize)
        pylab.figure(10)
        pylab.ioff()
        pylab.clf()
        pylab.axis("off")
        pylab.imshow(img,interpolation="nearest",animated=True) 
    notsave=False
    #if cfg.__dict__.has_key("test"):
    #    notsave=cfg.test
    #f=pyrHOG2.pyrHOG(imname,interv=10,savedir=cfg.auxdir+"/hog/",notsave=not(cfg.savefeat),notload=not(cfg.loadfeat),hallucinate=cfg.hallucinate,cformat=True,flip=imageflip,resize=cfg.resize)
    f=pyrHOG2.pyrHOG(img,interv=10,savedir=cfg.auxdir+"/hog/",notsave=not(cfg.savefeat),notload=not(cfg.loadfeat),hallucinate=cfg.hallucinate,cformat=True)#,flip=imageflip,resize=cfg.resize)
    res=[]
    for clm,m in enumerate(models):
        if cfg.useRL:
            res.append(pyrHOG2RL.detectflip(f,m,gtbbox,hallucinate=cfg.hallucinate,initr=cfg.initr,ratio=cfg.ratio,deform=cfg.deform,bottomup=cfg.bottomup,usemrf=cfg.usemrf,numneg=cfg.numneg,thr=cfg.thr,posovr=cfg.posovr,minnegincl=cfg.minnegincl,small=cfg.small,show=cfg.show,cl=clm,mythr=cfg.mythr,mpos=cfg.mpos,usefather=cfg.usefather,useprior=cfg.useprior,K=cfg.k))
        else:
            res.append(pyrHOG2.detect(f,m,gtbbox,hallucinate=cfg.hallucinate,initr=cfg.initr,ratio=cfg.ratio,deform=cfg.deform,bottomup=cfg.bottomup,usemrf=cfg.usemrf,numneg=cfg.numneg,thr=cfg.thr,posovr=cfg.posovr,minnegincl=cfg.minnegincl,small=cfg.small,show=cfg.show,cl=clm,mythr=cfg.mythr,mpos=cfg.mpos,usefather=cfg.usefather,useprior=cfg.useprior,emptybb=False,K=cfg.k))
    if cfg.show:
        pylab.draw()
        pylab.show()
    return res


def loadcfg(place,name,cls,select):
    class config(object):
        pass

    cfg=config()
    #cfg.cls=cls#"bicycle"
    cfg.numcl=2
    #if len(sys.argv)>2:
    #    it=int(sys.argv[2])
    #testname="./data/11_02_26/%s_%d_comp_bias2"%(cfg.cls,cfg.numcl)
    #testname="./data/11_03_28/%s%d_1aspect_last_sort"%(cfg.cls,cfg.numcl)
    testname="%s/%s%d_%s"%(place,cls,cfg.numcl,name)
    cfg.testname=testname
    import util
    cfg=util.load(testname+".cfg")
    try:
        num=int(select)
        cfg.maxtest=num
        select="pos"
    except (ValueError):
        pass  
  
    cfg.select=select
    #cfg.mythr=-10
    #cfg.mpos=1
    #if len(sys.argv)>3:
    #    cfg.mythr=float(sys.argv[3])
    #cfg.mythr=thr
    #cfg.bottomup=False
    cfg.maxtest=5000
    cfg.small=False
    #cfg.year="2007"
    #cfg.maxtest=16#5000
    #cfg.initr=0
    cfg.show=False
    if cfg.show:
        cfg.multipr=False
    else:
        cfg.multipr=4
    cfg.savefeat=False
    cfg.loadfeat=False
    cfg.thr=-2
    cfg.resize=1.0
    cfg.auxdir="/home/marcopede/databases/VOC2007/VOCdevkit/local/VOC2007/"#"/state/partition1/marcopede/"
    cfg.dbpath="/home/marcopede/databases/"
    return cfg

def test(thr,cfg):
    
    #cfg.test=True
    import util
    it=7
    models=util.load("%s%d.model"%(cfg.testname,it))
    w=[]
    rho=[]
    cfg.mythr=thr
    #for l in range(cfg.numcl):
    #    w.append(util.ModeltoW(models[l],cfg.usemrf,cfg.usefather,cfg.k,lastlev=1))
    #    rho.append(models[l]["rho"])
    #cfg.mythr=cfg.mythr*numpy.mean([numpy.sum(x**2) for x in w])#-numpy.mean(rho)
    #raw_input()    
    #cfg.mythr=cfg.mythr#-numpy.mean(rho)
    if cfg.multipr==1:
        numcore=None
    else:
        numcore=cfg.multipr

    mypool = Pool(numcore)
    
    if cfg.cls=="inria":
        if cfg.select=="pos":
            tsImages=getRecord(InriaTestData(basepath=cfg.dbpath),cfg.maxtest)
        else:
            tsImages=getRecord(InriaTestFullData(basepath=cfg.dbpath),cfg.maxtest)
    else:
        if select=="dir":#"." or select[0]=="/"
            import glob
            #lst=glob.glob("/home/marcopede/ivan/zebra/CVC_Zebra1/*.jpeg")[150:]
            lst=glob.glob("/media/ca0567b8-ee6d-4590-8462-0d093addb4cf/video/*.png")
            lst.sort()
            lst=lst[3900:]
            total=len(lst)
            tsImages=numpy.zeros(total,dtype=[("id",numpy.int32),("name",object),("bbox",list)])
            for idl,l in enumerate(lst):
                tsImages[idl]["id"]=idl
                tsImages[idl]["name"]=l#l.split("/")[-1]
                tsImages[idl]["bbox"]=None             
        else:
            tsImages=getRecord(VOC07Data(select=cfg.select,cl="%s_test.txt"%cfg.cls,
                    basepath=cfg.dbpath,#"/home/databases/",
                    usetr=True,usedf=False),cfg.maxtest)
        
    mypool = Pool(numcore)
 
    print "Test"
    print "Pruning Threshold:",cfg.mythr
    numhog=0
    initime=time.time()
    detlist=[]
    mycfg=copy.copy(cfg)
    mycfg.numneg=0
    arg=[[i,tsImages[i]["name"],None,models,mycfg] for i in range(len(tsImages))]
    t=time.time()
    if not(cfg.multipr):
        itr=itertools.imap(detectWrap,arg)        
    else:
        itr=mypool.imap(detectWrap,arg)
    for ii,res in enumerate(itr):
        totneg=0
        fuse=[]
        for mix in res:
            tr=mix[0]
            fuse+=mix[1]
            numhog+=mix[3]
        #for h in fuse:
        #    h["scr"]+=models[h["cl"]]["ra"]
        rfuse=tr.rank(fuse,maxnum=300)
        nfuse=tr.cluster(rfuse,ovr=0.3,inclusion=False)
        #print "----Test Image %d----"%ii
        print "----Test Image %s----"%tsImages[ii]["name"].split("/")[-1]
        for l in nfuse:
            detlist.append([tsImages[ii]["name"].split("/")[-1].split(".")[0],l["scr"],l["bbox"][1],l["bbox"][0],l["bbox"][3],l["bbox"][2]])
        print "Detections:",len(nfuse)
        if cfg.show:
            if cfg.show==True:
                showlabel="Parts"
            else:
                showlabel=False
            pylab.figure(20)
            pylab.ioff()
            pylab.clf()
            pylab.axis("off")
            img=util.myimread(tsImages[ii]["name"],resize=cfg.resize)
            pylab.imshow(img,interpolation="nearest",animated=True)
            pylab.gca().set_ylim(0,img.shape[0])
            pylab.gca().set_xlim(0,img.shape[1])
            pylab.gca().set_ylim(pylab.gca().get_ylim()[::-1])
            tr.show(nfuse,parts=showlabel,thr=-1.0,maxnum=0)           
            pylab.draw()
            pylab.show()
            #raw_input()
        if True:
            #RSZ=0.5
            f=pylab.figure(11,figsize=(9,5))
            pylab.ioff()
            pylab.clf()
            axes=pylab.Axes(f, [.0,.0,1.0,1.0]) # [left, bottom, width, height] where each value is between 0 and 1
            f.add_axes(axes) 
            img=util.myimread(tsImages[ii]["name"],resize=cfg.resize)
            pylab.imshow(img,interpolation="nearest",animated=True)
            #raw_input()
            pylab.axis("off")
            tr.show(nfuse,parts=True,thr=-0.99,scr=True)
            pylab.axis((0,img.shape[1],img.shape[0],0))
            pylab.ion()
            pylab.draw()
            pylab.show()  
            #raw_input()
            pylab.savefig("/media/ca0567b8-ee6d-4590-8462-0d093addb4cf/video/det/"+tsImages[ii]["name"].split("/")[-1].split(".")[0]+".png")  
    del itr
    
    #tp,fp,scr,tot=VOCpr.VOCprlistfastscore(tsImages,detlist,numim=cfg.maxpostest,show=False,ovr=0.5)
    #tp,fp,scr,tot=VOCpr.VOCprRecord_wrong(tsImages,detlist,show=False,ovr=0.5)
    tp,fp,scr,tot=VOCpr.VOCprRecord(tsImages,detlist,show=False,ovr=0.5)
    pylab.figure(15)
    pylab.clf()
    rc,pr,ap=VOCpr.drawPrfast(tp,fp,tot)
    pylab.draw()
    pylab.show()
    #pylab.savefig("%s_ap%d_test%s%.1f.png"%(testname,it,select,cfg.mythr))
    tottime=((time.time()-initime))
    print "Threshold used:",cfg.mythr
    print "Total number of HOG:",numhog
    print "AP(it=",it,")=",ap
    print "Testing Time: %.3f s"%tottime#/3600.0)
    #results={"det":detlist,"ap":ap,"tp":tp,"fp":fp,"pr":pr,"rc":rc,"numhog":numhog,"mythr":cfg.mythr,"time":tottime}
    #util.savemat("%s_ap%d_test_thr_%.3f.mat"%(cfg.testname,it,cfg.mythr),results)
    #util.save("%s_ap%d_test_thr_%.3f.dat"%(testname,it,cfg.mythr),results)
    #util.savedetVOC(detlist,"%s_ap%d_test_thr_%.3f.txt"%(testname,it,cfg.mythr))
    #fd=open("%s_ap%d_test%s.txt"%(cfg.testname,it,select),"a")
    #fd.write("Threshold used:%f\n"%cfg.mythr)
    #fd.write("Total number of HOG:%d\n"%numhog)
    #fd.write("Average precision:%f\n"%ap)
    #fd.write("Testing Time: %.3f s\n\n"%tottime)
    #fd.close()
    return ap,numhog


if __name__=="__main__":

    import sys
    if len(sys.argv)>1:
        cls=sys.argv[1]
    else:
        cls="car"

    if len(sys.argv)>2:
        imname=sys.argv[2]
    else:
        imname="000034.jpg"
    
    cfg=loadcfg("./data/finalRL","test",cls,"all")
    
    if len(sys.argv)>3:
        tmin=float(sys.argv[3])
        tmax=float(sys.argv[4])
    else:
        tmin=-2
        tmax=2        

    #lthr=[tmin,tmax]
    #dap=[]
    #speed=[]
    #pmin=0
    #pmax=1
    #ap1,hog1=test(-10,cfg)
    #print ap1,hog1
    cfg.show=False
    m=util.load("%s%d.model"%(cfg.testname,7))
    res=detectWrap([0,imname,None,m,cfg])
    fuse=[]
    numhog=0
    for mix in res:
        tr=mix[0]
        fuse+=mix[1]
        numhog+=mix[3]
    #for h in fuse:
    #    h["scr"]+=models[h["cl"]]["ra"]
    rfuse=tr.rank(fuse,maxnum=300)
    nfuse=tr.cluster(rfuse,ovr=0.3,inclusion=False)
    import pylab
    f=pylab.figure(20)#,figsize=(9,5))
    pylab.ioff()
    pylab.clf()
    axes=pylab.Axes(f, [.0,.0,1.0,1.0]) # [left, bottom, width, height] where each value is between 0 and 1
    f.add_axes(axes) 
    img=util.myimread(imname,resize=cfg.resize)
    pylab.imshow(img,interpolation="nearest",animated=True)
    #raw_input()
    pylab.axis("off")
    tr.show(nfuse,parts=True,thr=-0.99,scr=True,maxnum=2)
    pylab.axis((0,img.shape[1],img.shape[0],0))
    pylab.ion()
    pylab.draw()
    pylab.show()  
    print "---- Image %s----"%imname
    #print "Detections:", len(res[1]) 
    #dettime=res[2]
    #numhog=res[3]
    #print "Detection Time:",dettime
    print "Number of computed HOGs:",numhog
    #print "Press a key..."
    #raw_input()


