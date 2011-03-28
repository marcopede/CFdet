import matplotlib
matplotlib.use("Agg") #if run out of ipython do not show any graph
#from procedures import *
from database import *
from multiprocessing import Pool
import util
import pyrHOG2
import VOCpr
import time
import copy
import itertools
from trainPASCAL import *


if __name__=="__main__":

    cfg=config()

    cfg.cls="bicycle"
    it=7
    import sys
    if len(sys.argv)>1:
        cfg.cls=sys.argv[1]
    cfg.numcl=2
    if len(sys.argv)>2:
        it=int(sys.argv[2])
    #testname="./data/11_02_26/%s_%d_comp_bias2"%(cfg.cls,cfg.numcl)
    testname="./data/11_03_23/%s%d_keepdef"%(cfg.cls,cfg.numcl)
    cfg=util.load(testname+".cfg")
    cfg.mythr=-10
    #cfg.mpos=1
    if len(sys.argv)>3:
        cfg.mythr=float(sys.argv[3])
    if len(sys.argv)>4:
        select=sys.argv[4]
    else:
        select="all"
    #cfg.bottomup=False
    #cfg.year="2007"
    cfg.maxtest=5000
    #cfg.initr=0
    cfg.show=False
    if cfg.show:
        cfg.multipr=False
    else:
        cfg.multipr=8
    cfg.savefeat=False
    cfg.loadfeat=False
    cfg.thr=-2
    cfg.auxdir="/home/databases/VOC2007/VOCdevkit/local/VOC2007/"#"/state/partition1/marcopede/"
    #cfg.test=True
    models=util.load("%s%d.model"%(testname,it))

    if cfg.multipr==1:
        numcore=None
    else:
        numcore=cfg.multipr

    mypool = Pool(numcore)
    
    if cfg.cls=="inria":
        tsImages=getRecord(InriaTestFullData(basepath=cfg.dbpath),5000)
    else:
        tsImages=getRecord(VOC07Data(select=select,cl="%s_test.txt"%cfg.cls,
                    basepath=cfg.dbpath,#"/home/databases/",
                    usetr=True,usedf=False),cfg.maxtest)
        
    mypool = Pool(numcore)
 
    print "Test"
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
        print "----Test Image %d----"%ii
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
            img=util.myimread(tsImages[ii]["name"])
            pylab.imshow(img,interpolation="nearest",animated=True)
            pylab.gca().set_ylim(0,img.shape[0])
            pylab.gca().set_xlim(0,img.shape[1])
            pylab.gca().set_ylim(pylab.gca().get_ylim()[::-1])
            tr.show(nfuse,parts=showlabel,thr=-0.8,maxnum=10)           
            pylab.show()
            #raw_input()
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
    results={"det":detlist,"ap":ap,"tp":tp,"fp":fp,"pr":pr,"rc":rc,"numhog":numhog,"mythr":cfg.mythr,"time":tottime}
    #util.savemat("%s_ap%d_test_thr_%.3f.mat"%(testname,it,cfg.mythr),results)
    #util.save("%s_ap%d_test_thr_%.3f.dat"%(testname,it,cfg.mythr),results)
    util.savedetVOC(detlist,"%s_ap%d_test_thr_%.3f.txt"%(testname,it,cfg.mythr))
    fd=open("%s_ap%d_test%s.txt"%(testname,it,select),"a")
    fd.write("Threshold used:%f\n"%cfg.mythr)
    fd.write("Total number of HOG:%d\n"%numhog)
    fd.write("Average precision:%f\n"%ap)
    fd.write("Testing Time: %.3f s\n\n"%tottime)
    fd.close()



