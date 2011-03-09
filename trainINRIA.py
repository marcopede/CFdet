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
import pegasos
import time

def detectWrap(a):
    t=time.time()
    i=a[0]
    imname=a[1]
    bbox=a[2]
    m=a[3]
    cfg=a[4]
    if cfg.show:
        img=util.myimread(imname)
        pylab.figure(10)
        pylab.ioff()
        pylab.clf()
        pylab.axis("off")
        pylab.imshow(img,interpolation="nearest",animated=True) 
    if bbox!=None:
        gtbbox=[{"bbox":x} for x in bbox]   
    else:
        gtbbox=None
    f=pyrHOG2.pyrHOG(imname,interv=10,savedir=cfg.savedir+"/hog/",notload=not(cfg.loadfeat),notsave=not(cfg.savefeat),hallucinate=cfg.hallucinate,cformat=True)
    res=pyrHOG2.detect(f,m,gtbbox,hallucinate=cfg.hallucinate,initr=cfg.initr,ratio=cfg.ratio,deform=cfg.deform,posovr=cfg.posovr,bottomup=cfg.bottomup,usemrf=cfg.usemrf,numneg=cfg.numneg,thr=cfg.thr,inclusion=cfg.inclusion,small=False,show=cfg.show,usefather=cfg.usefather)
    if cfg.show:
        pylab.show()
    print "Detect Wrap:",time.time()-t
    return res

#class config(object):
#    pass
        
class stats(object):
    """
    keep track of interesting variables    
    """
    def __init__(self,l):
        self.l=l

    def listvar(self,l):
        """
        set the list of variables [{"name":name1,"fct":fct1,"txt":descr1},[name2,fct2,descr2],..,[nameN,fctN,descrN]]
        """
        self.l=l

    def report(self,filename,mode="a",title=None):
                
        f=open(filename,mode)
        if title!=None:
            f.write("---%s "%title+time.asctime()+"------\n")
        else:
            f.write("---Stats on "+time.asctime()+"------\n")
        for v in self.l:     
            #if v["name"] in dir()       
            if v.has_key("fnc"):
                exec("value=str(%s(%s))"%(v["fnc"],v["name"]))
            else:
                if v["name"][-1]=="*":
                    exec("value=str(%s.__dict__)"%(v["name"][:-1]))
                else:
                    exec("value=str(%s)"%(v["name"]))
            #else 
            #    value="Not Exists!"
            if v.has_key("txt"):
                f.write("%s=%s\n"%(v["txt"],value))
            else:
                f.write("%s=%s\n"%(v["name"],value))
        f.close()

def unique_slow(old,new):
    if old==[]:
        return new
    unew=[]
    mold=numpy.array(old)
    for ep,e in enumerate(new):
        #print numpy.sum(numpy.abs(e-mold),1)
        #raw_input()
        print ep,"/",len(new)
        if numpy.all(numpy.sum(numpy.abs(e-mold),1)>0.1):
            unew.append(e)
    print "Doules",len(new)-len(unew)
    #raw_input()
    return unew


def unique(old,new):
    if old==[]:
        return new,0
    unew=[]
    mold=numpy.array(old)
    for ep,e in enumerate(new):
        #print numpy.sum(numpy.abs(e-mold),1)
        #raw_input()
        print ep,"/",len(new)
        apr=numpy.sum(numpy.abs(e[:100]-mold[:,:100]),1)
        #print apr
        #raw_input()
        if numpy.all(apr>0.1):
            unew.append(e)    
        else:
            if numpy.all(numpy.sum(numpy.abs(e-mold[apr<=0.1,:]),1)>0.1):
                unew.append(e)
    doubles=len(new)-len(unew)
    print "Doubles",doubles
    #raw_input()
    return unew,doubles


if __name__=="__main__":
#    cfg=config()

    try:
        from config_local import * #your own configuration
    except:
        print "config_local.py is not present loading configdef.py"
        from config import * #default configuration  
        
    if cfg.savedir=="":
        cfg.savedir=InriaPosData(basepath=cfg.dbpath).getStorageDir() #where to save

    util.save(cfg.testname+".cfg",cfg)

#    dbpath="/home/databases/"
#    cfg.fy=8#remember small
#    cfg.fx=3
#    cfg.lev=3
#    cfg.interv=10
#    cfg.ovr=0.45
#    cfg.sbin=8
#    cfg.maxpos=2000#120
#    cfg.maxtest=2000#100
#    cfg.maxneg=2000#120
#    cfg.maxexamples=10000
#    cfg.deform=True
#    cfg.usemrf=False
#    cfg.usefather=True
#    cfg.bottomup=False
#    cfg.initr=1
#    cfg.ratio=1
#    cfg.hallucinate=1
#    cfg.numneginpos=5
#    cfg.useflipos=True
#    cfg.useflineg=True
#    cfg.svmc=0.001#0.001#0.002#0.004
#    cfg.show=True
#    cfg.thr=-2
#    cfg.multipr=4
#    cfg.numit=20#10
#    cfg.comment="I shuld get more than 84... hopefully"
#    cfg.numneg=0#not used but necessary
#    testname="./data/11_02_28/inria_rightpedro"
#    cfg.savefeat=False #save precomputed features 
#    cfg.savedir=InriaPosData(basepath=dbpath).getStorageDir() #where to save
#    mydebug=False
#    if mydebug:
#        cfg.multipr=False
#        cfg.maxpos=10
#        cfg.maxneg=10
#        cfg.maxtest=10
#        cfg.maxexamples=1000

#    util.save(testname+".cfg",cfg)

    if cfg.multipr==1:
        numcore=None
    else:
        numcore=cfg.multipr

    mypool = Pool(numcore)

    trPosImages=getRecord(InriaPosData(basepath=cfg.dbpath),cfg.maxpos)
    trNegImages=getRecord(InriaNegData(basepath=cfg.dbpath),cfg.maxneg)
    tsImages=getRecord(InriaTestFullData(basepath=cfg.dbpath),cfg.maxtest)

    stcfg=stats([{"name":"cfg*"}])
    stcfg.report(cfg.testname+".rpt.txt","w","Initial Configuration")

    sts=stats(
        [{"name":"it","txt":"Iteration"},
        {"name":"nit","txt":"Negative Iteration"},
        {"name":"trpos","fnc":"len","txt":"Positive Examples"},
        {"name":"trneg","fnc":"len","txt":"Negative Examples"},
        {"name":"nSupportVectorsPos","txt":"Positive Support Vectors"},
        {"name":"nSupportVectorsNeg","txt":"Negative Support Vectors"},
        {"name":"doubles","txt":"Double Examples"}]
        )

    rpres=stats([{"name":"tinit","txt":"Time from the beginning"},
                {"name":"tpar","txt":"Time last iteration"},
                {"name":"ap","txt":"Average precision: "}])

    import time
    initime=time.time()
    #intit model

    fy=cfg.fy
    fx=cfg.fx
    #lowf1=numpy.random.random((fy,fx,31)).astype(numpy.float32)
    lowf1=numpy.ones((fy,fx,31)).astype(numpy.float32)
    lowf=lowf1/numpy.sqrt(numpy.sum(lowf1**2))/float(fx*fy)
    #lowd=numpy.random.random((1,1,4)).astype(numpy.float32)
    lowd=-numpy.ones((1,1,4)).astype(numpy.float32)
    #midf1=numpy.random.random((2*fy,2*fx,31)).astype(numpy.float32)
    midf1=numpy.ones((2*fy,2*fx,31)).astype(numpy.float32)
    midf=midf1/numpy.sqrt(numpy.sum(midf1**2))/float(4*fx*fy)
    #midd=numpy.random.random((2,2,4)).astype(numpy.float32)
    midd=-numpy.ones((2,2,4)).astype(numpy.float32)
    #higf1=numpy.random.random((4*fy,4*fx,31)).astype(numpy.float32)
    higf1=numpy.ones((4*fy,4*fx,31)).astype(numpy.float32)
    higf=higf1/numpy.sqrt(numpy.sum(higf1**2))/float(16*fx*fy)
    #higd=numpy.random.random((4,4,4)).astype(numpy.float32)
    higd=-numpy.ones((4,4,4)).astype(numpy.float32)
    vhigf1=numpy.ones((8*fy,8*fx,31)).astype(numpy.float32)
    vhigf=vhigf1/numpy.sqrt(numpy.sum(vhigf1**2))/float(64*fx*fy)
    #higd=numpy.random.random((4,4,4)).astype(numpy.float32)
    vhigd=-numpy.ones((8,8,4)).astype(numpy.float32)
    ww3=[lowf,midf,higf]
    wwh=[lowf,midf,higf,vhigf]
    ww2=[midf,higf]
    ww1=[higf]
    ww4=[lowf,midf]
    ww5=[midf]
    ww6=[lowf]
    dd1=[lowd]
    dd2=[lowd,midd]
    dd3=[lowd,midd,higd]
    ddh=[lowd,midd,higd,vhigd]
    rho=0
    model1={"ww":ww1,"rho":rho,"df":dd1,"fy":ww1[0].shape[0],"fx":ww1[0].shape[1]}
    model2={"ww":ww2,"rho":rho,"df":dd2,"fy":ww2[0].shape[0],"fx":ww2[0].shape[1]}
    model3={"ww":ww3,"rho":rho,"df":dd3,"fy":ww3[0].shape[0],"fx":ww3[0].shape[1]}
    modelh={"ww":wwh,"rho":rho,"df":ddh,"fy":wwh[0].shape[0],"fx":wwh[0].shape[1]}
    model4={"ww":ww4,"rho":rho,"df":dd2,"fy":ww4[0].shape[0],"fx":ww4[0].shape[1]}
    model5={"ww":ww5,"rho":rho,"df":dd1,"fy":ww5[0].shape[0],"fx":ww5[0].shape[1]}
    model6={"ww":ww6,"rho":rho,"df":dd1,"fy":ww6[0].shape[0],"fx":ww6[0].shape[1]}
    lmodels=[model1,model2,model3]
    m=model3#model3

    trpos=[]
    trneg=[]
    mpos=0.5
    for it in range(cfg.numit):
        trpos=[]
        newtrneg=[]
        #trneg=[]#only for a test
        cfgpos=copy.copy(cfg)
        cfgpos.numneg=cfg.numneginpos
        partime=time.time()
        print "---Positive Images:----"
    #    if it==0:#force to take best overlapping
    #        mpos=10
    #    else:
    #        mpos=0.5
        arg=[[i,trPosImages[i]["name"],trPosImages[i]["bbox"],m,cfgpos] for i in range(len(trPosImages))]
        t=time.time()
        #mypool = Pool(numcore)
        if not(cfg.multipr):
            itr=itertools.imap(detectWrap,arg)        
        else:
            #res=mypool.map(detectWrap,arg)
            itr=mypool.imap(detectWrap,arg)
        for ii,res in enumerate(itr):
        #res=it.next()
            trpos+=res[3]
            newtrneg+=res[4]
            print "----Pos Image %d----"%ii
            print "Pos:",len(res[3]),"Neg:",len(res[4])
            print "Tot Pos:",len(trpos)," Neg:",len(newtrneg)      
        del itr
        newtrneg2,doubles=unique(trneg,newtrneg)
        trneg=trneg+newtrneg2
        print "Time:",time.time()-t
        #for im in res:
        #    trpos+=im[3]            
        #    trneg+=im[4]
        print "---Positive Images:----"
        print "   Tot Pos:",len(trpos)
        print "   Tot Neg:",len(trneg)      
        #negative retraining
        trneglen=1
        cfgneg=copy.copy(cfg)
        cfgneg.numneg=10
        cfgneg.thr=-1.002
        nparts=20
        newPositives=True
        t=time.time()
        nSupportVectorsPos = 0
        nSupportVectorsNeg = 0
        for nit in range(5):
            newtrneg=[]
            print "---Negative Images It %d -----"%nit
            limit=False
            order=range(nparts)#numpy.random.permutation(range(nparts))
            for rr in order:
                arg=[[i,trNegImages[i]["name"],[],m,cfgneg] for i in range(rr*len(trNegImages)/nparts,(rr+1)*len(trNegImages)/nparts)]
                print "PART:%d Elements:%d"%(rr,len(arg))
                #for i in range(len(trNegImages)):
                #    print "------Neg Image ",i,"------"
                #arg=[[i,trNegImages[i]["name"],[],m,cfgneg] for i in range((it+1)*len(trNegImages)/(10))]
                #t=time.time()
                #mypool = Pool(numcore)
                if not(cfg.multipr):
                    itr=itertools.imap(detectWrap,arg)        
                else:
                    itr=mypool.imap(detectWrap,arg)
                for ii,res in enumerate(itr):
                    trpos+=res[3]#not necessary cause no positives
                    newtrneg+=res[4]
                    if (nit>0 and res[4]!=[]):
                        scr=res[2][0]["scr"]
                        dense=numpy.sum(res[4][0]*w)-r
                        #raw_input()
                        if abs(scr-dense)>0.0001:
                            print "Warning: the two scores must be equal!!!"
                            print "Scr:",scr,"DesneSCR:",dense,"Diff:",abs(scr-dense)
                            raw_input()
                    print "----Neg Image %d----"%ii
                    print "Pos:",len(res[3]),"Neg:",len(res[4])
                    print "Tot Pos:",len(trpos)," Neg:",len(newtrneg)      
                if len(trneg)+len(newtrneg)+len(trpos)>cfg.maxexamples:
                    print "Cache Limit Reached!"
                    limit=True
                    break
            del itr
            #check for doubles
            newtrneg2,doubles=unique(trneg,newtrneg)
            trneg=trneg+newtrneg2
            #mypool.close()            
            print "Time:",time.time()-t
            tr=res[0]#to build the model in the learning
            print "---Negative Images It %d:----"%nit
            print "   Tot Pos:",len(trpos)
            print "   Tot Neg:",len(trneg)
            raw_input()

            #print len(trneg),trneglen
            #if len(trneg)/float(trneglen)<1.05 and not(limit):
            #if len(newtrneg2)<5 and not(newPositives):
            if (float(len(newtrneg2))<=0.05*nSupportVectorsNeg) and not(newPositives) and not(limit):
                print "Not enough negatives, convergence!"
                break
            newPositives=False

            sts.report(cfg.testname+".rpt.txt","a","Before Training")

            print "SVM learning"
            svmname="%s.svm"%cfg.testname
            #lib="libsvm"
            lib="linear"
            #pc=0.008 #single resolution
            pc=cfg.svmc #high res
            #util.trainSvmRaw(trpos,trneg,svmname,dir="",pc=pc,lib=lib)
            #w,r=util.loadSvm(svmname,dir="",lib=lib)
            import time
            tt1=time.time()
            #w1,r1=pegasos.train(trpos,trneg,svmname,dir="",pc=pc)
            w,r,prloss=pegasos.trainComp(trpos,trneg,svmname,dir="",pc=pc)
            r=-w[-1]*100;w=w[:-1]
            #print "Results"
            #raw_input()
            #w,r=pegasos.trainkeep(trpos,trneg,svmname,dir="",pc=pc)

            m=tr.model(w,r,len(m["ww"]),31,usemrf=cfg.usemrf,usefather=cfg.usefather)
            util.save("%s%d.model"%(cfg.testname,it),m)
            if cfg.deform:
                print m["df"]

            print "Show model"
            pylab.figure(100)
            pylab.clf()
            util.drawModel(m["ww"])
            pylab.draw()
            pylab.show()
            pylab.savefig("%s_hog%d.png"%(cfg.testname,it))
            if cfg.deform:
                pylab.figure(101)
                pylab.clf()
                util.drawDeform(m["df"])
                pylab.draw()
                pylab.show()
                pylab.savefig("%s_def%d.png"%(cfg.testname,it))
            #raw_input()

            print "Filter Data"
            print "Length before:",len(trneg)
            for p,d in enumerate(trneg):
                if numpy.sum(d*w)-r<-1:
                    if (len(trneg)+len(trpos))>((cfg.maxexamples)/2):
                        trneg.pop(p)
            print "Length after:",len(trneg)
            trneglen=len(trneg)

            nSupportVectorsPos = 0
            nSupportVectorsNeg = 0
            for p,d in enumerate(trpos):
                if (numpy.sum(d*w)-r<= 1):
                #if (pegasos.predict2(d,w,r) <= 1):                
                    nSupportVectorsPos = nSupportVectorsPos + 1
                    
            for p,d in enumerate(trneg):
                if (numpy.sum(d*w)-r>=-1):
                #if (pegasos.predict2(d,w,r) >=-1):                
                    nSupportVectorsNeg = nSupportVectorsNeg + 1      

            sts.report(cfg.testname+".rpt.txt","a","After Filtering")
     
        print "Test"
        detlist=[]
        print "---Test Images----"
        arg=[[i,tsImages[i]["name"],None,m,cfg] for i in range(len(tsImages))]
        #t=time.time()
        #mypool = Pool(numcore)
        if not(cfg.multipr):
            itr=itertools.imap(detectWrap,arg)        
        else:
            itr=mypool.imap(detectWrap,arg)
        for i,im in enumerate(itr):
            print "---- Image %d----"%i
            print "Detections:", len(im[1])    
            for l in im[1]:
                detlist.append([tsImages[i]["name"].split("/")[-1].split(".")[0],l["scr"],l["bbox"][1],l["bbox"][0],l["bbox"][3],l["bbox"][2]])
        del itr
        #mypool.close()
        tp,fp,scr,tot=VOCpr.VOCprRecord(tsImages,detlist,show=False,ovr=0.5)
        pylab.figure(15)
        pylab.clf()
        rc,pr,ap=VOCpr.drawPrfast(tp,fp,tot)
        pylab.draw()
        pylab.show()
        pylab.savefig("%s_ap%d.png"%(cfg.testname,it))
        util.savemat("%s_ap%d.mat"%(cfg.testname,it),{"tp":tp,"fp":fp,"scr":scr,"tot":tot,"rc":rc,"pr":pr,"ap":ap})
        tinit=((time.time()-initime)/3600.0)
        tpar=((time.time()-partime)/3600.0)
        print "AP(it=",it,")=",ap
        print "Partial Time: %.3f h"%tpar
        print "Training Time: %.3f h"%tinit
        rpres.report(cfg.testname+".rpt.txt","a","Results")
        



