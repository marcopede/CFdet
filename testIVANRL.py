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
from trainCVC02keep2RL import *

def loadcfg(place,name,cls,select,numcl=4):
    cfg=config()
    #cfg.cls=cls#"bicycle"
    cfg.numcl=numcl
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
    #cfg.bottomup=True
    cfg.maxtest=2500
    #cfg.small=False
    #cfg.year="2007"
    #cfg.maxtest=16#5000
    #cfg.initr=0
    cfg.show=True
    if cfg.show:
        cfg.multipr=4
    else:
        cfg.multipr=4
    cfg.savefeat=False
    cfg.loadfeat=False
    cfg.thr=-2
    #cfg.resize=0.5
    #cfg.deform=False
    #cfg.small=False
    #cfg.occl=True
    cfg.resize=1.0
    cfg.auxdir="/home/marcopede/databases/VOC2007/VOCdevkit/local/VOC2007/"#"/state/partition1/marcopede/"
    cfg.dbpath="/home/marcopede/databases/"
    return cfg

def test(thr,cfg,it=9):
    
    #cfg.test=True
    import util
    #it=it
    models=util.load("%s%d.model"%(cfg.testname,it))
    #skip occlusion
    #for l in models:
    #    del l["occl"]

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
    elif cfg.cls=="cvc02":
        #tsImages=getRecord(CVC02test(),cfg.maxtest)
        tsImages=getRecord(CVC02test(basepath="/media/OS/data/DATASET-CVC-02/CVC-02-System/",
                images="/sequence-15/color/",
                annotations="/sequence-15/annotations/",),cfg.maxtest)
        tsImagesFull=tsImages
    elif cfg.cls=="ivan":
        stest=10000
        #tsImages=getRecord(ImgFile("/media/OS/data/PVTRA101/CLEAR06_PVTRA101a01_502_Bbox.txt",imgpath="/media/OS/data/PVTRA101/images/"),stest+cfg.maxtest)[stest:]
        #tsImages=getRecord(ImgFile("/media/OS/data/PVTRA101a19/GrTr_CLEAR06_PVTRA101a19.txt",imgpath="/media/OS/data/PVTRA101a19/CLEAR06_PVTRA101a19/",sort=True,amin=1000),cfg.maxtest)[:600]#[:1950]#the other frames do not have GT
        #tsImages=getRecord(ImgFile("/media/OS/data/PVTRA101a19/GrTr_CLEAR06_PVTRA101a19_only12.txt",imgpath="/media/OS/data/PVTRA101a19/CLEAR06_PVTRA101a19/",sort=True,amin=100),cfg.maxtest)[:(1950/12)]#the other frames do not have GT
        #tsImages=getRecord(ImgFile("/media/OS/data/PVTRA101a19/CLEAR06_PVTRA101a19_people_Celik_allfr.txt",imgpath="/media/OS/data/PVTRA101a19/CLEAR06_PVTRA101a19/",sort=True,amin=100),cfg.maxtest)#pedestrian
        tsImages=getRecord(ImgFile("/media/OS/data/PVTRA101a19/CLEAR06_PVTRA101a19_vehicles_Celik_allfr.txt",imgpath="/media/OS/data/PVTRA101a19/CLEAR06_PVTRA101a19/",sort=True,amin=100),cfg.maxtest)#vechicles
        #tsImages=getRecord(ImgFile("/media/OS/data/PVTRA101a19/CLEAR06_PVTRA101a19_PV_Celik_allfr.txt",imgpath="/media/OS/data/PVTRA101a19/CLEAR06_PVTRA101a19/",sort=True,amin=100),cfg.maxtest)#pedestrian+vechicles
        tsImagesFull=tsImages
        
    #mypool = Pool(numcore)
 
    print "Test"
    print "Pruning Threshold:",cfg.mythr
    numhog=0
    initime=time.time()
    detlist=[]
    mycfg=copy.copy(cfg)
    mycfg.numneg=0
    mycfg.show=False
    arg=[[i,tsImages[i]["name"],None,models,mycfg] for i in range(len(tsImages))]
    t=time.time()
    if not(cfg.multipr):
        itr=itertools.imap(detectWrap,arg)        
    else:
        itr=mypool.imap(detectWrap,arg)
    pylab.figure(20)
    pylab.clf()
    ax = pylab.subplot(111)
    img=util.myimread(tsImages[0]["name"],resize=cfg.resize)
    pylab.ioff()
    axes=pylab.Axes(pylab.gcf(), [.0,.0,1.0,1.0]) # [left, bottom, width, height] where each value is between 0 and 1
    pylab.gcf().add_axes(axes) 
    pylab.axis("off")          
    aim=pylab.imshow(img,interpolation="nearest",animated=True)
    pylab.axis((0,img.shape[1],img.shape[0],0))           
    pylab.draw()
    pylab.show()
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
        nfuse=tr.cluster(rfuse,ovr=cfg.ovrasp,inclusion=False)
        #print "----Test Image %d----"%ii
        print "----Test Image %s----"%tsImages[ii]["name"].split("/")[-1]
        for idl,l in enumerate(nfuse):
            #print "DET:",l["bbox"]
            #raw_input()
            #if l["bbox"][0]/cfg.resize<125:
            #    nfuse.pop(idl)
            #    continue
            detlist.append([tsImages[ii]["name"].split("/")[-1].split(".")[0],l["scr"],l["bbox"][1]/cfg.resize,l["bbox"][0]/cfg.resize,l["bbox"][3]/cfg.resize,l["bbox"][2]/cfg.resize])
        print "Detections:",len(nfuse)
        if cfg.show:
            if cfg.show==True:
                showlabel=False#"Parts"#False#"Parts"
            else:
                showlabel=False
            pylab.figure(20)
            xx=pylab.gca()            
            del xx.texts[:]
            del xx.lines[:]
            #print pylab.gca().images
            #pylab.ioff()
#            axes=pylab.Axes(pylab.gcf(), [.0,.0,1.0,1.0]) # [left, bottom, width, height] where each value is between 0 and 1
#            pylab.gcf().add_axes(axes) 
#            pylab.axis("off")
            #img=util.myimread(tsImages[ii]["name"],resize=cfg.resize)
            img=pylab.imread(tsImages[ii]["name"])#myimread has memory problems!!!!!!!
            #pylab.clf()
            aim.set_array(img)
            #ims=pylab.imshow(img,interpolation="nearest",animated=True)
            tr.show(nfuse,parts=showlabel,thr=0,maxnum=100)
            pylab.axis((0,img.shape[1],img.shape[0],0))           
            pylab.draw()
            pylab.show()
            #print ax.images
            #raw_input()
        showGT=True
        if showGT:
            bb=tsImages[ii]["bbox"]
            for mbb in bb:
                util.box(mbb[0], mbb[1], mbb[2], mbb[3], col='r', lw=2)
            pylab.axis((0,img.shape[1],img.shape[0],0))
            pylab.draw()
            #raw_input()
        if select=="dir":
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
    pylab.figure(16)
    pylab.clf()
    fppi,miss,ap=VOCpr.drawMissRatePerImage(tp,fp,tot,len(tsImages))
    pylab.draw()
    pylab.show()
    sfdsd
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

    if len(sys.argv)>2:
        select=sys.argv[2]
    else:
        select="all"
    
    #cfg=loadcfg("./data/CVC02/11_07_21/","full",cls,select)
    #cfg=loadcfg("./data/CVC02/11_07_22/","fullOccl",cls,select)
    #cfg=loadcfg("./data/IVAN/11_07_27/","test3",cls,select)
    #cfg=loadcfg("./data/IVAN/11_07_28/","test3",cls,select)
    #cfg=loadcfg("./data/INRIA/11_07_28/","RLright",cls,select)
    #cfg=loadcfg("./data/IVAN/11_09_06/","test3",cls,select)
    #cfg=loadcfg("./data/IVAN/11_09_06/","testCLUSTERIVAN",cls,select)
    #cfg=loadcfg("./data/IVAN/11_09_06/","testFAST_real",cls,select)
    #cfg=loadcfg("./data/IVAN/11_09_06/","testFAST",cls,select)
    cfg=loadcfg("./data/IVAN/11_10_03/","cars",cls,select,numcl=1)
    #cfg=loadcfg("./data/IVAN/11_10_03/","car2aspect",cls,select,numcl=3)
    
    if len(sys.argv)>3:
        tmin=float(sys.argv[3])
        tmax=float(sys.argv[4])
    else:
        tmin=-2
        tmax=2        

    lthr=[tmin,tmax]
    dap=[]
    speed=[]
    pmin=0
    pmax=1
    ap1,hog1=test(-10,cfg,6)
    print ap1,hog1
    sdad
    ap2,hog2=test(lthr[pmax],cfg)
    lap=[ap1,ap2]
    lhog=[hog1,hog2]
    nlhog=[]
    gap=ap1-ap2
    it=0
    bound1=lthr[0]
    bound2=lthr[1]
    bpos1=1
    bpos2=1
    while (gap>(ap1-ap2)*0.1) and it<30:
        it+=1
        athr=(lthr[pmin]+lthr[pmax])/2.0
        print "AP",lap
        print "DELATAP",dap
        print "THR",lthr
        print "NHOG",nlhog
        print "HOG",lhog
        print "SPEEED",speed
        print "New Thr",athr 
        #raw_input()
        ap,hog=test(athr,cfg)
        if ap>=ap1:
            bound1=athr
            bpos1=pmin
        if ap<=ap2:
            bound2=athr     
            bpos2=pmin+1
        lhog.insert(pmin+1,hog)
        lap.insert(pmin+1,ap)
        lthr.insert(pmin+1,athr)
        pylab.figure(30)
        nlhog=numpy.array(lhog)/float(lhog[0])
        pylab.clf()
        pylab.plot(lhog,lap,"-",lw=2)
        pylab.figure(31)
        speed=1.0/(nlhog)
        pylab.clf()
        #pylab.semilogx(speed*12,lap,"-",lw=2)
        pylab.plot(speed*12,lap,"-",lw=2)
        pylab.xlim(0,200)
        pylab.show()
        print "Thr",athr 
        dap=[lap[idx]-lap[idx+1] for idx,x in enumerate(lap[1:])]
        #dap=[lhog[idx-1]-x for idx,x in enumerate(lhog[1:])]
        #dap=[lhog[idx]-x for idx,x in enumerate(lhog[1:])]
        #dap=[]
        #for idx,x in enumerate(lhog[1:]):
        #    dap.append(lhog[idx-1]-lhog[idx])
        #gap=numpy.max(dap[bpos1-1:bpos2])
        pmin=numpy.argmax(dap)
        pmax=pmin+1
        print "Saving partial results!"
        util.savemat(cfg.testname+"_speed.mat",{"AP":lap,"DAP":dap,"THR":lthr,"NHOG":nlhog,"HOG":lhog,"SPEED":speed,"BOUND":[bound1,bound2]})
    print bound1,bound2

