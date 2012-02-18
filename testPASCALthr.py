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
#from trainPASCALkeep2RL import *
from trainBOW import *

def loadcfg(place,name,cls,select):
    cfg=config()
    #cfg.cls=cls#"bicycle"
    cfg.numcl=2
    #if len(sys.argv)>2:
    #    it=int(sys.argv[2])
    #testname="./data/11_02_26/%s_%d_comp_bias2"%(cfg.cls,cfg.numcl)
    #testname="./data/11_03_28/%s%d_1aspect_last_sort"%(cfg.cls,cfg.numcl)
    testname="%s/%s%d_%s"%(place,cls,cfg.numcl,name)
    import util
    cfg=util.load(testname+".cfg")
    cfg.testname=testname
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
        cfg.multipr=8
    cfg.savefeat=False
    cfg.loadfeat=False
    cfg.thr=-2
    cfg.auxdir=""#"/home/marcopede/databases/VOC2007/VOCdevkit/local/VOC2007/"#"/state/partition1/marcopede/"
    return cfg

def test(thr,cfg,show=True):
    
    #cfg.test=True
    import util
    it=9
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

    #mypool = Pool(numcore)
    
    if cfg.cls=="inria":
        if cfg.select=="pos":
            tsImages=getRecord(InriaTestData(basepath=cfg.dbpath),cfg.maxtest)
        else:
            tsImages=getRecord(InriaTestFullData(basepath=cfg.dbpath),cfg.maxtest)
    else:
        tsImages=getRecord(VOC07Data(select=cfg.select,cl="%s_test.txt"%cfg.cls,
                    basepath=cfg.dbpath,#"/home/databases/",
                    usetr=True,usedf=False),cfg.maxtest)
        
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
        print "----Test Image %d----"%ii
        for l in nfuse:
            detlist.append([tsImages[ii]["name"].split("/")[-1].split(".")[0],l["scr"],l["bbox"][1],l["bbox"][0],l["bbox"][3],l["bbox"][2]])
        print "Detections:",len(nfuse)
        if 0:#show:#cfg.show:
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
            pylab.draw()
            pylab.show()
            #raw_input()
    del itr
    
    #tp,fp,scr,tot=VOCpr.VOCprlistfastscore(tsImages,detlist,numim=cfg.maxpostest,show=False,ovr=0.5)
    #tp,fp,scr,tot=VOCpr.VOCprRecord_wrong(tsImages,detlist,show=False,ovr=0.5)
    tp,fp,scr,tot,pdet=VOCpr.VOCprRecordthr(tsImages,detlist,show=False,ovr=0.5)
#    pdet.sort()
#    print pdet
#    if len(pdet)>0:
#        nthr=10*(pdet[int(len(pdet)*0.9)]-pdet[int(len(pdet)*0.1)])/11+pdet[int(len(pdet)*0.1)]#each time takes 10% of recall
#        if nthr==thr:
#            print "More drastic thr"
#            nthr=20*(pdet[int(len(pdet)*0.9)]-pdet[int(len(pdet)*0.1)])/21+pdet[int(len(pdet)*0.1)]
#        elif nthr<thr:
#            nthr=thr+0.01
#    else:
#        nthr=100
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
    return ap,numhog#,nthr


if __name__=="__main__":

    import sys
    if len(sys.argv)>1:
        cls=sys.argv[1]

    if len(sys.argv)>2:
        select=sys.argv[2]
    else:
        select="all"

    cfg=loadcfg("./data/PASCAL/12_01_31_VOC07_baseline","Normal",cls,select)
    cfg.multipr=8
    if cfg.multipr==1:
        numcore=None
    else:
        numcore=cfg.multipr

    mypool = Pool(numcore)
    cfg.usebow=False
    cfg.multipr=4
    cfg.maxtest=5000
    speed=[]
    thr=0#-0.05# for tvmonitor
    lhog=[]
    lap=[]
    lthr=[thr]
    dt=0.01
    for l in range(10):
        ap,hog=test(thr,cfg,show=True)
        if len(lap)>0:
            dap=abs(lap[-1]-ap)
            dhog=float(lhog[-1])/hog
            print "DAP",dap
            print "DHOG",dhog
            if dap>2 or dhog>2:#go slower
                print "SLOWER"
                dt=dt/2
            if dap<0.5 and dhog<1.5:#go faster
                print "FASTER"
                dt=dt*2
            print "DT",dt
            print "THR",thr
            thr=thr+dt
        lhog.append(hog)
        lap.append(ap)
        pylab.figure(30)
        nlhog=numpy.array(lhog)/float(lhog[0])
        pylab.clf()
        pylab.plot(lhog,lap,"-",lw=2)
        pylab.title("Number of HOG")
        pylab.figure(31)
        speed=1.0/(nlhog)
        pylab.clf()
        #pylab.semilogx(speed*12,lap,"-",lw=2)
        pylab.plot(speed*12,lap,"-",lw=2)
        pylab.title("Speed-up")
        pylab.xlim(0,400)
        pylab.ylim(0,max(lap)+0.01)
        pylab.draw()
        pylab.show()
        print "AP:",lap
        print "THR:",lthr
        print "NHOG:",nlhog
        print "SPEED:",speed
        print "Saving partial results!"
        util.savemat(cfg.testname+"_speed.mat",{"AP":lap,"THR":lthr,"NHOG":nlhog,"HOG":lhog,"SPEED":speed
})
        if ap==0:
            print "Finished!!"
            break
        lthr.append(thr)
        #raw_input()






#    ap2,hog2=test(lthr[pmax],cfg)
#    lap=[ap1,ap2]
#    lhog=[hog1,hog2]
#    nlhog=[]
#    gap=ap1-ap2
#    it=0
#    bound1=lthr[0]
#    bound2=lthr[1]
#    bpos1=1
#    bpos2=1
#    while (gap>(ap1-ap2)*0.1) and it<30:
#        it+=1
#        athr=(lthr[pmin]+lthr[pmax])/2.0
#        print "AP",lap
#        print "DELATAP",dap
#        print "THR",lthr
#        print "NHOG",nlhog
#        print "HOG",lhog
#        print "SPEEED",speed
#        print "New Thr",athr 
#        #raw_input()
#        ap,hog=test(athr,cfg)
#        if ap>=ap1:
#            bound1=athr
#            bpos1=pmin
#        if ap<=ap2:
#            bound2=athr     
#            bpos2=pmin+1
#        lhog.insert(pmin+1,hog)
#        lap.insert(pmin+1,ap)
#        lthr.insert(pmin+1,athr)
#        pylab.figure(30)
#        nlhog=numpy.array(lhog)/float(lhog[0])
#        pylab.clf()
#        pylab.plot(lhog,lap,"-",lw=2)
#        pylab.figure(31)
#        speed=1.0/(nlhog)
#        pylab.clf()
#        #pylab.semilogx(speed*12,lap,"-",lw=2)
#        pylab.plot(speed*12,lap,"-",lw=2)
#        pylab.xlim(0,200)
#        pylab.show()
#        print "Thr",athr 
#        dap=[lap[idx]-lap[idx+1] for idx,x in enumerate(lap[1:])]
#        #dap=[lhog[idx-1]-x for idx,x in enumerate(lhog[1:])]
#        #dap=[lhog[idx]-x for idx,x in enumerate(lhog[1:])]
#        #dap=[]
#        #for idx,x in enumerate(lhog[1:]):
#        #    dap.append(lhog[idx-1]-lhog[idx])
#        #gap=numpy.max(dap[bpos1-1:bpos2])
#        pmin=numpy.argmax(dap)
#        pmax=pmin+1
#        print "Saving partial results!"
#        util.savemat(cfg.testname+"_speed.mat",{"AP":lap,"DAP":dap,"THR":lthr,"NHOG":nlhog,"HOG":lhog,"SPEED":speed,"BOUND":[bound1,bound2]})
#    print bound1,bound2

