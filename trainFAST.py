import matplotlib
matplotlib.use("Agg") #if run out of ipython do not show any graph
#from procedures import *
from database import *
from multiprocessing import Pool
import util
import pyrHOG2
import pyrHOG2RL
import VOCpr
import time
import copy
import itertools

class config(object):
    pass

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

def remove_empty(seq):
    newseq=[]
    for l in seq:
        if l!=[]:
            newseq.append(l)
    return newseq

#get image name and bbox
def extractInfo(trPosImages,maxnum=-1,usetr=True,usedf=False):
    bb=numpy.zeros((len(trPosImages)*20,4))#as maximum 5 persons per image in average
    name=[]
    cnt=0
    tot=0
    if maxnum==-1:
        tot=len(trPosImages)
    else:
        tot=min(maxnum,trPosImages)
    for idx in range(tot):
        #print trPosImages.getImageName(idx)
        #img=trPosImages.getImage(idx)
        rect=trPosImages[idx]["bbox"]#.getBBox(idx,usetr=usetr,usedf=usedf)
        for r in rect:
            bb[cnt,:]=r[:4]
            name.append(trPosImages[idx]["name"])#.getImageName(idx))
            cnt+=1
        #img=pylab.imread("circle.png")
        util.pdone(idx,tot)
    ratio=(bb[:,2]-bb[:,0])/(bb[:,3]-bb[:,1])
    area=(bb[:,2]-bb[:,0])*(bb[:,3]-bb[:,1])
    return name,bb[:cnt,:],ratio[:cnt],area[:cnt]

def buildense(trpos,trposcl,cumsize,bias=100):
    ftrpos=[]
    for iel,el in enumerate(trpos):
        ftrpos.append(numpy.zeros(cumsize[-1],dtype=numpy.float32))
        ftrpos[-1][cumsize[trposcl[iel]]:cumsize[trposcl[iel]+1]-1]=trpos[iel]
        #bias
        ftrpos[-1][cumsize[trposcl[iel]+1]-1]=bias
    return ftrpos    

def clear(keep=("__builtins__", "clear")):
    keeps = {}
    for name, value in globals().iteritems():
        if name in keep: keeps[name] = value
        globals().clear()
        for name, value in keeps.iteritems():
            globals()[name] = value

#def detectWrap(a):
#    i=a[0]
#    imname=a[1]
#    bbox=a[2]
#    models=a[3]
#    cfg=a[4]
#    if len(a)<=5:
#        imageflip=False
#    else:
#        imageflip=a[5]
#    img=util.myimread(imname,resize=cfg.resize)
#    if imageflip:
#        img=util.myimread(imname,True,resize=cfg.resize)
#        if bbox!=None:
#             bbox = util.flipBBox(img,bbox)
#    if bbox!=None:
#        gtbbox=[{"bbox":x,"img":imname.split("/")[-1]} for x in bbox]   
#    else:
#        gtbbox=None
#    if cfg.show:
#        img=util.myimread(imname,imageflip,resize=cfg.resize)
#        pylab.figure(10)
#        pylab.ioff()
#        pylab.clf()
#        pylab.axis("off")
#        pylab.imshow(img,interpolation="nearest",animated=True) 
#    notsave=False
#    #if cfg.__dict__.has_key("test"):
#    #    notsave=cfg.test
#    #f=pyrHOG2.pyrHOG(imname,interv=10,savedir=cfg.auxdir+"/hog/",notsave=not(cfg.savefeat),notload=not(cfg.loadfeat),hallucinate=cfg.hallucinate,cformat=True,flip=imageflip,resize=cfg.resize)
#    f=pyrHOG2.pyrHOG(img,interv=10,savedir=cfg.auxdir+"/hog/",notsave=not(cfg.savefeat),notload=not(cfg.loadfeat),hallucinate=cfg.hallucinate,cformat=True)#,flip=imageflip,resize=cfg.resize)
#    res=[]
#    for clm,m in enumerate(models):
#        if cfg.useRL:
#            res.append(pyrHOG2RL.detectflip(f,m,gtbbox,hallucinate=cfg.hallucinate,initr=cfg.initr,ratio=cfg.ratio,deform=cfg.deform,bottomup=cfg.bottomup,usemrf=cfg.usemrf,numneg=cfg.numneg,thr=cfg.thr,posovr=cfg.posovr,minnegincl=cfg.minnegincl,small=cfg.small,show=cfg.show,cl=clm,mythr=cfg.mythr,mpos=cfg.mpos,usefather=cfg.usefather,useprior=cfg.useprior,K=cfg.k))
#        else:
#            res.append(pyrHOG2.detect(f,m,gtbbox,hallucinate=cfg.hallucinate,initr=cfg.initr,ratio=cfg.ratio,deform=cfg.deform,bottomup=cfg.bottomup,usemrf=cfg.usemrf,numneg=cfg.numneg,thr=cfg.thr,posovr=cfg.posovr,minnegincl=cfg.minnegincl,small=cfg.small,show=cfg.show,cl=clm,mythr=cfg.mythr,mpos=cfg.mpos,usefather=cfg.usefather,useprior=cfg.useprior,emptybb=False,K=cfg.k))
#    if cfg.show:
#        pylab.draw()
#        pylab.show()
#    return res

def rundet(img,cfg,models,gtbbox):
    if cfg.show:
        #img=util.myimread(imname,imageflip,resize=cfg.resize)
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
            res.append(pyrHOG2RL.detectflip(f,m,gtbbox,hallucinate=cfg.hallucinate,initr=cfg.initr,ratio=cfg.ratio,deform=cfg.deform,bottomup=cfg.bottomup,usemrf=cfg.usemrf,numneg=cfg.numneg,thr=cfg.thr,posovr=cfg.posovr,minnegincl=cfg.minnegincl,small=cfg.small,show=cfg.show,cl=clm,mythr=cfg.mythr,mpos=cfg.mpos,usefather=cfg.usefather,useprior=cfg.useprior,K=cfg.k,occl=cfg.occl,fastBU=cfg.fastBU))
        else:
            res.append(pyrHOG2.detect(f,m,gtbbox,hallucinate=cfg.hallucinate,initr=cfg.initr,ratio=cfg.ratio,deform=cfg.deform,bottomup=cfg.bottomup,usemrf=cfg.usemrf,numneg=cfg.numneg,thr=cfg.thr,posovr=cfg.posovr,minnegincl=cfg.minnegincl,small=cfg.small,show=cfg.show,cl=clm,mythr=cfg.mythr,mpos=cfg.mpos,usefather=cfg.usefather,useprior=cfg.useprior,K=cfg.k,occl=cfg.occl,fastBU=cfg.fastBU))
    if cfg.show:
        pylab.draw()
        pylab.show()
    return res

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
        if bbox!=[]:#positive
            cfg.usecrop=False
            gtbbox=[{"bbox":numpy.array(x[:6])*cfg.resize,"img":imname.split("/")[-1]} for x in bbox]   
            if cfg.usecrop:          
                tres=[]
                res=[]
                for x in bbox:
                    margin=0.3
                    dy=x[2]-x[0];dx=x[3]-x[1]
                    dd=max(dy,dx)
                    ny1=max(0,x[0]-margin*dd);ny2=min(x[2]+margin*dd,img[0].shape)
                    nx1=max(0,x[1]-margin*dd);nx2=min(x[3]+margin*dd,img.shape[1])
                    img1=img[ny1:ny2,nx1:nx2]
                    #gt1=[{"bbox":[ny1-x[0]+2*margin*dy,nx1-x[1]+2*margin*dx,ny2-x[0],nx2-x[1],0,0],"img":imname.split("/")[-1]}]
                    gt1=[{"bbox":[x[0]-ny1,x[1]-nx1,x[0]-ny1+dy,x[1]-nx1+dx,0,0],"img":imname.split("/")[-1]}]
                    tres.append(rundet(img1,cfg,models,gt1))
                for cl,mod in enumerate(tres):
                    res.append([[],[],[],[],[]])
                    #tr,best1,worste1,ipos,ineg
                    res[cl][0]=mod[0][0]
                    for el in mod:
                        res[cl][1]+=el[1]
                        res[cl][2]+=el[2]
                        res[cl][3]+=el[3]
                        res[cl][4]+=el[4]
                        if res[cl][1]!=[]:
                            res[cl][1][0]["bbid"]=cl
                        if res[cl][2]!=[]:
                            res[cl][2][0]["bbid"]=cl                
            else:
                res=rundet(img,cfg,models,gtbbox)        
        else:
            res=rundet(img,cfg,models,[])    
    else:
        gtbbox=None
        res=rundet(img,cfg,models,gtbbox)
    return res


def myunique(old,new,oldcl,newcl,numcl):
    if old==[]:
        return new,newcl
    unew=[]
    unewcl=[]
    #mold=numpy.array(old)
    clst=[]
    for c in range(numcl):
        select=numpy.arange(len(oldcl))[numpy.array(oldcl)==c]
        clst.append(numpy.array([old[i] for i in select]))
    for ep,e in enumerate(new):
        #print ep,"/",len(new)
        print ".",
        #check the cluster
        #selec=numpy.arange(len(oldcl))[numpy.array(oldcl)==newcl[ep]]
        apr=numpy.sum(numpy.abs(e[::100]-clst[newcl[ep]][:,::100]),1)
        #print apr
        #raw_input()
        if numpy.all(apr>0.1):
            unew.append(e)
            unewcl.append(newcl[ep])    
        else:
            if numpy.all(numpy.sum(numpy.abs(e-clst[newcl[ep]][apr<=0.1,:]),1)>0.1):
                unew.append(e)
                unewcl.append(newcl[ep]) 
                print ep,"/",len(new)
    print "Doules",len(new)-len(unew)
    #raw_input()
    return [unew,unewcl]

def extract_feat(tr,dtrpos,cumsize,useRL):
    ls=[];lscl=[]
    for el in dtrpos:
        aux=(tr.descr(dtrpos[el],flip=False,usemrf=cfg.usemrf,usefather=cfg.usefather,k=cfg.k))
        ls+=aux
        auxcl=tr.mixture(dtrpos[el])
        lscl+=auxcl
        if not(useRL):
            ls+=(tr.descr(dtrpos[el],flip=True,usemrf=cfg.usemrf,usefather=cfg.usefather,k=cfg.k))
            lscl+=tr.mixture(dtrpos[el])
        if cumsize!=None:
            dns=buildense(aux,auxcl,cumsize)
            #print "Det:",dtrpos[el][0]["img"],[x["scr"] for x in dtrpos[el]],tr.mixture(dtrpos[el])
            print "Det:",dtrpos[el][0]["img"],[numpy.sum(x*w)-r for x in dns],tr.mixture(dtrpos[el])
    #if cumsize!=None:    
    #    raw_input()
    return ls,lscl

def extract_feat2(tr,dtrpos,cumsize,useRL):
    ls=[];lscl=[]
    for el in dtrpos:
        aux=(tr.descr(dtrpos[el],flip=False,usemrf=cfg.usemrf,usefather=cfg.usefather,k=cfg.k))
        #ls+=aux
        auxcl=tr.mixture(dtrpos[el])
        #lscl+=auxcl
        if not(useRL):
            aux2=(tr.descr(dtrpos[el],flip=True,usemrf=cfg.usemrf,usefather=cfg.usefather,k=cfg.k))
            auxcl2=tr.mixture(dtrpos[el])
        for l in range(len(aux)):
            ls.append(aux[l])
            lscl.append(auxcl[l])
            if not(useRL):
                ls.append(aux2[l])
                lscl.append(auxcl2[l])
        if cumsize!=None:
            dns=buildense(aux,auxcl,cumsize)
            #print "Det:",dtrpos[el][0]["img"],[x["scr"] for x in dtrpos[el]],tr.mixture(dtrpos[el])
            print "Det:",dtrpos[el][0]["img"],[numpy.sum(x*w)-r for x in dns],tr.mixture(dtrpos[el])
    #if cumsize!=None:    
    print "Number descr:",len(ls)
    #raw_input()
    return ls,lscl

def loss_pos(trpos,trposcl,cumsize):
    dns=buildense(trpos,trposcl,cumsize)
    loss=0
    lscr=[]
    for l in dns:
        scr=numpy.sum(l*w)
        print "Scr:",scr
        loss+=max(0,1-scr)
        lscr.append(scr)
    return loss,lscr

import pegasos

def train(a):
    #import pegasos
    trpos=a[0]
    trneg=a[1]
    trposcl=a[2]
    trnegcl=a[3]
    w=a[4]
    testname=a[5]
    svmc=a[6]
    #k=a[7]
    #numthr=a[8]
    w,r,prloss=pegasos.trainComp(trpos,trneg,testname+"loss.rpt.txt",trposcl,trnegcl,oldw=w,dir="",pc=svmc,eps=0.01)
    return w,r,prloss

def trainParallel(trpos,trneg,testname,trposcl,trnegcl,w,svmc,multipr,parallel=True,numcore=4):
    
    if not(parallel):
        w,r,prloss=pegasos.trainComp(trpos,trneg,testname+"loss.rpt.txt",trposcl,trnegcl,oldw=w,dir="",pc=svmc)
    else:
        #atrpos=numpy.array(trpos,dtype=object)
        #atrposcl=numpy.array(trposcl,dtype=object)
        #atrneg=numpy.array(trneg,dtype=object)
        #atrnegcl=numpy.array(trnegcl,dtype=object)
        ltrpos=len(trpos);ltrneg=len(trneg)
        reordpos=range(ltrpos);numpy.random.shuffle(reordpos)
        reordneg=range(ltrneg);numpy.random.shuffle(reordneg)
        ltr=[]
        litpos=ltrpos/numcore;litneg=ltrneg/numcore
        atrpos=[];atrposcl=[]
        atrneg=[];atrnegcl=[]
        for ll in range(ltrpos):
            atrpos.append(trpos[reordpos[ll]])                   
            atrposcl.append(trposcl[reordpos[ll]])                   
        for ll in range(ltrneg):
            atrneg.append(trneg[reordneg[ll]])                   
            atrnegcl.append(trnegcl[reordneg[ll]])                   
        for gr in range(numcore-1):
            ltr.append([atrpos[litpos*gr:litpos*(gr+1)],atrneg[litneg*gr:litneg*(gr+1)],atrposcl[litpos*gr:litpos*(gr+1)],atrnegcl[litneg*gr:litneg*(gr+1)],w,testname,svmc])        
        ltr.append([atrpos[litpos*(gr+1):],atrneg[litneg*(gr+1):],atrposcl[litpos*(gr+1):],atrnegcl[litneg*(gr+1):],w,testname,svmc])        
        if not(multipr):
            itr=itertools.imap(train,ltr)        
        else:
            itr=mypool.map(train,ltr)
        waux=numpy.zeros((numcore,len(w)))
        raux=numpy.zeros((numcore))
        #lprloss=[]
        #lenprloss=[]
        for ii,res in enumerate(itr):
            waux[ii]=res[0]
            raux[ii]=res[1]    
        prloss=res[2]#take the last one
        w=numpy.mean(waux,0)
        r=numpy.mean(raux)
    return w,r,prloss

def trainParallel2(trpos,trneg,testname,trposcl,trnegcl,w,svmc,multipr,parallel=True,numcore=4,mypool=None):
    
    if not(parallel):
        w,r,prloss=pegasos.trainComp(trpos,trneg,testname+"loss.rpt.txt",trposcl,trnegcl,oldw=w,dir="",pc=svmc)
    else:
        #atrpos=numpy.array(trpos,dtype=object)
        #atrposcl=numpy.array(trposcl,dtype=object)
        #atrneg=numpy.array(trneg,dtype=object)
        #atrnegcl=numpy.array(trnegcl,dtype=object)
        ltrpos=len(trpos);ltrneg=len(trneg)
        reordpos=range(ltrpos);numpy.random.shuffle(reordpos)
        reordneg=range(ltrneg);numpy.random.shuffle(reordneg)
        ltr=[]
        litpos=ltrpos/numcore;litneg=ltrneg/numcore
        atrpos=[];atrposcl=[]
        atrneg=[];atrnegcl=[]
        for ll in range(ltrpos):
            atrpos.append(trpos[reordpos[ll]])                   
            atrposcl.append(trposcl[reordpos[ll]])                   
        for ll in range(ltrneg):
            atrneg.append(trneg[reordneg[ll]])                   
            atrnegcl.append(trnegcl[reordneg[ll]])                   
        for gr in range(numcore-1):
            ltr.append([atrpos[litpos*gr:litpos*(gr+1)],atrneg[litneg*gr:litneg*(gr+1)],atrposcl[litpos*gr:litpos*(gr+1)],atrnegcl[litneg*gr:litneg*(gr+1)],w,testname,svmc])        
        ltr.append([atrpos[litpos*(gr+1):],atrneg[litneg*(gr+1):],atrposcl[litpos*(gr+1):],atrnegcl[litneg*(gr+1):],w,testname,svmc])        
        if not(multipr):
            itr=itertools.imap(train,ltr)        
        else:
            itr=mypool.map(train,ltr)
        waux=numpy.zeros((numcore,len(w)))
        raux=numpy.zeros((numcore))
        #lprloss=[]
        #lenprloss=[]
        for ii,res in enumerate(itr):
            waux[ii]=res[0]
            raux[ii]=res[1]    
        prloss=res[2]#take the last one
        w=numpy.mean(waux,0)
        r=numpy.mean(raux)
    return w,r,prloss


if __name__=="__main__":

    import sys
    
    batch=""
    if len(sys.argv)>2: #batch configuration
        batch=sys.argv[2]

    #cfg=config()
    try:
        if batch=="batch":
            print "Loading Batch configuration"
            from config_local_batch import * #your own configuration
        else:
            print "Loading Normal configuration"
            from config_local_pascal import * #your own configuration
    except:
        print "config_local.py is not present loading configdef.py"
        from config import * #default configuration  
    
    if len(sys.argv)>1: #class
        cfg.cls=sys.argv[1]
    
    #if cfg.savedir=="":
        #cfg.savedir=InriaPosData(basepath=cfg.dbpath).getStorageDir() #where to save
        #cfg.savedir=VOC07Data(basepath=cfg.dbpath).getStorageDir()

    cfg.testname=cfg.testpath+cfg.cls+("%d"%cfg.numcl)+"_"+cfg.testspec
    cfg.train="keep2"
    util.save(cfg.testname+".cfg",cfg)

    cfg.auxdir=cfg.savedir
    testname=cfg.testname

    if cfg.multipr==1:
        numcore=None
    else:
        numcore=cfg.multipr

    mypool = Pool(numcore)

    stcfg=stats([{"name":"cfg*"}])
    stcfg.report(testname+".rpt.txt","w","Initial Configuration")

    sts=stats(
        [{"name":"it","txt":"Iteration"},
        {"name":"nit","txt":"Negative Iteration"},
        {"name":"trpos","fnc":"len","txt":"Positive Examples"},
        {"name":"trneg","fnc":"len","txt":"Negative Examples"}]
        )

    rpres=stats([{"name":"tinit","txt":"Time from the beginning"},
                {"name":"tpar","txt":"Time last iteration"},
                {"name":"ap","txt":"Average precision: "}])

    clst=stats([{"name":"l","txt":"Cluster "},
                {"name":"npcl","txt":"Positive Examples"},
                {"name":"nncl","txt":"Negative Examples"}])

    stloss=stats([{"name":"output","txt":""},
                {"name":"negratio[-1]","txt":"Ratio Neg loss:"},
                {"name":"nexratio[-1]","txt":"Ratio Examples: "},
                {"name":"posratio[-1]","txt":"Ratio Pos loss: "}])

    stpos=stats([{"name":"cntadded","txt":"Added"},
                {"name":"cntnochange","txt":"No Change"},
                {"name":"cntgoodchnage","txt":"Good Change"},
                {"name":"cntkeepoldscr","txt":"Keep old scr"}])


    #trPosImages=InriaPosData(basepath="/home/databases/")
    #trNegImages=InriaNegData(basepath="/home/databases/")
    #tsImages=InriaTestFullData(basepath="/home/databases/")
    #training
    if cfg.db=="VOC":
        if cfg.year=="2007":
            trPosImages=getRecord(VOC07Data(select="pos",cl="%s_trainval.txt"%cfg.cls,
                            basepath=cfg.dbpath,#"/home/databases/",
                            usetr=True,usedf=False),cfg.maxpos)
            trNegImages=getRecord(VOC07Data(select="neg",cl="%s_trainval.txt"%cfg.cls,
                            basepath=cfg.dbpath,#"/home/databases/",#"/share/ISE/marcopede/database/",
                            usetr=True,usedf=False),cfg.maxneg)
            trNegImagesFull=getRecord(VOC07Data(select="neg",cl="%s_trainval.txt"%cfg.cls,
                            basepath=cfg.dbpath,usetr=True,usedf=False),5000)
            #test
            tsPosImages=getRecord(VOC07Data(select="pos",cl="%s_test.txt"%cfg.cls,
                            basepath=cfg.dbpath,#"/home/databases/",#"/share/ISE/marcopede/database/",
                            usetr=True,usedf=False),cfg.maxtest)
            tsNegImages=getRecord(VOC07Data(select="neg",cl="%s_test.txt"%cfg.cls,
                            basepath=cfg.dbpath,#"/home/databases/",#"/share/ISE/marcopede/database/",
                            usetr=True,usedf=False),cfg.maxneg)
            tsImages=numpy.concatenate((tsPosImages,tsNegImages),0)
            tsImagesFull=getRecord(VOC07Data(select="all",cl="%s_test.txt"%cfg.cls,
                            basepath=cfg.dbpath,
                            usetr=True,usedf=False),5000)
        elif cfg.year=="2011":
            trPosImages=getRecord(VOC11Data(select="pos",cl="%s_train.txt"%cfg.cls,
                            basepath=cfg.dbpath,#"/home/databases/",
                            usetr=True,usedf=False),cfg.maxpos)
            trNegImages=getRecord(VOC11Data(select="neg",cl="%s_train.txt"%cfg.cls,
                            basepath=cfg.dbpath,#"/home/databases/",#"/share/ISE/marcopede/database/",
                            usetr=True,usedf=False),cfg.maxneg)
            trNegImagesFull=getRecord(VOC11Data(select="neg",cl="%s_train.txt"%cfg.cls,
                            basepath=cfg.dbpath,usetr=True,usedf=False),30000)
            #test
            tsPosImages=getRecord(VOC11Data(select="pos",cl="%s_val.txt"%cfg.cls,
                            basepath=cfg.dbpath,#"/home/databases/",#"/share/ISE/marcopede/database/",
                            usetr=True,usedf=False),cfg.maxtest)
            tsNegImages=getRecord(VOC11Data(select="neg",cl="%s_val.txt"%cfg.cls,
                            basepath=cfg.dbpath,#"/home/databases/",#"/share/ISE/marcopede/database/",
                            usetr=True,usedf=False),cfg.maxneg)
            tsImages=numpy.concatenate((tsPosImages,tsNegImages),0)
            tsImagesFull=getRecord(VOC11Data(select="all",cl="%s_val.txt"%cfg.cls,
                            basepath=cfg.dbpath,
                            usetr=True,usedf=False),30000)
            
    elif cfg.db=="ivan":
        trPosImages=getRecord(ImgFile("/media/OS/data/PVTRA101/CLEAR06_PVTRA101a01_502_BboxROI.txt",imgpath="/media/OS/data/PVTRA101/images/",sort=True,amin=400),cfg.maxpos)[:1000:10]
        #trPosImages=getRecord(ImgFile("/media/OS/data/PVTRA101/CLEAR06_PVTRA101a01_PDT_vis1_objid-1_pres-1_occl0_syncat-1_amb0_mob-1.txt",imgpath="/media/OS/data/PVTRA101/images/"),cfg.maxpos)##pedestrian
        #trPosImages=getRecord(ImgFile("/media/OS/data/PVTRA101/CLEAR06_PVTRA101a01_VDT_vis1_objid-1_pres-1_occl0_syncat-1_amb0_mob-1.txt",imgpath="/media/OS/data/PVTRA101/images/"),cfg.maxpos)##cars
        #trPosImages=getRecord(ImgFile("/media/OS/data/PVTRA101/CLEAR06_PVTRA101a01_PVDT_vis1_objid-1_pres-1_occl0_syncat-1_amb0_mob-1.txt",imgpath="/media/OS/data/PVTRA101/images/"),cfg.maxpos)##car+person
        #trNegImages=getRecord(InriaNegData(basepath=cfg.dbpath),cfg.maxneg)
        trNegImages=getRecord(DirImages(imagepath="/media/OS/data/PVTRA101/neg/"),cfg.maxneg)
        trNegImagesFull=trNegImages
        #test
        #tsImages=getRecord(ImgFile("/media/OS/data/PVTRA101/CLEAR06_PVTRA101a01_502_Bbox.txt",imgpath="/media/OS/data/PVTRA101/images/"),10000+cfg.maxtest)[10000:]
        #tsImages=getRecord(ImgFile("/media/OS/data/PVTRA101/GrTr_CLEAR06_PVTRA101a01.txt",imgpath="/media/OS/data/PVTRA101/images/"),10000+cfg.maxtest)[10000:]
        #tsImages=getRecord(ImgFile("/media/OS/data/PVTRA101a19/GrTr_CLEAR06_PVTRA101a19_only12.txt",imgpath="/media/OS/data/PVTRA101a19/CLEAR06_PVTRA101a19/",sort=True,amin=100),cfg.maxtest)[:(1950/12)]#the other frames
        tsImages=getRecord(ImgFile("/media/OS/data/PVTRA101a19/CLEAR06_PVTRA101a19_PV_Celik_allfr.txt",imgpath="/media/OS/data/PVTRA101a19/CLEAR06_PVTRA101a19/",sort=True,amin=100),cfg.maxtest)#pedestrian+vechicles
        tsImagesFull=tsImages
    elif cfg.db=="inria":
        trPosImages=getRecord(InriaPosData(basepath=cfg.dbpath),cfg.maxpos)
        trNegImages=getRecord(InriaNegData(basepath=cfg.dbpath),cfg.maxneg)#check if it works better like this
        trNegImagesFull=getRecord(InriaNegData(basepath=cfg.dbpath),5000)
        #test
        tsImages=getRecord(InriaTestFullData(basepath=cfg.dbpath),cfg.maxtest)
        tsImagesFull=tsImages
    #cluster bounding boxes
    name,bb,r,a=extractInfo(trPosImages)
    trpos={"name":name,"bb":bb,"ratio":r,"area":a}
    import scipy.cluster.vq as vq
    numcl=cfg.numcl
    perc=cfg.perc#10
    minres=10
    minfy=3
    minfx=3
    #maxArea=25*(4-cfg.lev[0])
    maxArea=40*(4-cfg.lev[0])
    usekmeans=False

    #using kmeans
    clc,di=vq.kmeans(r,numcl,3)
    cl=vq.vq(r,clc)[0]
    for l in range(numcl):
        print "Cluster kmeans",l,":"
        print "Samples:",len(a[cl==l])
        print "Mean Area:",numpy.mean(a[cl==l])/16.0
        sa=numpy.sort(a[cl==l])
        print "Min Area:",numpy.mean(sa[int(len(sa)*perc)])/16.0
        print "Aspect:",numpy.mean(r[cl==l])
        print
    #using same number per cluster
    sr=numpy.sort(r)
    spl=[]
    lfy=[];lfx=[]
    cl=numpy.zeros(r.shape)
    for l in range(numcl):
        spl.append(sr[round(l*len(r)/float(numcl))])
    spl.append(sr[-1])
    for l in range(numcl):
        cl[numpy.bitwise_and(r>=spl[l],r<=spl[l+1])]=l
    for l in range(numcl):
        print "Cluster same number",l,":"
        print "Samples:",len(a[cl==l])
        #meanA=numpy.mean(a[cl==l])/16.0/(0.5*4**(cfg.lev[l]-1))#4.0
        meanA=numpy.mean(a[cl==l])/16.0/(4**(cfg.lev[l]-1))#4.0
        print "Mean Area:",meanA
        sa=numpy.sort(a[cl==l])
        #minA=numpy.mean(sa[len(sa)/perc])/16.0/(0.5*4**(cfg.lev[l]-1))#4.0
        minA=numpy.mean(sa[int(len(sa)*perc)])/16.0/(4**(cfg.lev[l]-1))#4.0
        print "Min Area:",minA
        aspt=numpy.mean(r[cl==l])
        print "Aspect:",aspt
        if minA>maxArea:
            minA=maxArea
        #minA=10#for bottle
        if aspt>1:
            fx=(max(minfx,numpy.sqrt(minA/aspt)))
            fy=(fx*aspt)
        else:
            fy=(max(minfy,numpy.sqrt(minA*(aspt))))
            fx=(fy/(aspt))        
        print "Fy:%.2f"%fy,"~",round(fy),"Fx:%.2f"%fx,"~",round(fx)
        lfy.append(round(fy))
        lfx.append(round(fx))
        print

    raw_input()

    cfg.fy=lfy#[7,10]#lfy
    cfg.fx=lfx#[11,7]#lfx

    import time
    initime=time.time()
    #intit model

    #mypool = Pool(numcore)
    
    models=[]
    for c in range(numcl):
        fy=cfg.fy[c]
        fx=cfg.fx[c]
        ww=[]
        dd=[]    
        for l in range(cfg.lev[c]):
            if cfg.useRL:
                lowf=numpy.zeros((fy*2**l,fx*2**l,31)).astype(numpy.float32)
                #lowf[:,:,2]=0.1/31
                #lowf[:,:,18+2]=0.1/31
                lowf[:(fy*2**l)/2,:,2]=0.1/31
                lowf[(fy*2**l)/2:,:,7]=0.1/31
                lowf[:(fy*2**l)/2,:,11]=0.1/31
                lowf[(fy*2**l)/2:,:,16]=0.1/31
                lowf[:(fy*2**l)/2,:,18+2]=0.1/31
                lowf[(fy*2**l)/2:,:,18+7]=0.1/31
                #lowf=lowf/(numpy.sum(lowf**2))
            else:
                lowf1=numpy.ones((fy*2**l,fx*2**l,31)).astype(numpy.float32)
                #lowf=lowf1/(numpy.sum(lowf1**2))
            lowd=-numpy.ones((1*2**l,1*2**l,4)).astype(numpy.float32)
            ww.append(lowf)
            dd.append(lowd)
            rho=0
        mynorm=0
        for wc in ww:
            mynorm+=numpy.sum(wc**2)
        for idw,wc in enumerate(ww):
            ww[idw]=wc*0.1/numpy.sqrt(mynorm)
        models.append({"ww":ww,"rho":rho,"df":dd,"fy":ww[0].shape[0],"fx":ww[0].shape[1]})

    if 0:        
        print "Show model"
        for idm,m in enumerate(models):    
            pylab.figure(100+idm)
            pylab.clf()
            util.drawModel(m["ww"])
            pylab.draw()
            pylab.show()
        raw_input()

    #fulltrpos=[]
    #fulltrposcl=[]
    trneg=[]
    trpos=[]
    trposcl=[]
    dtrpos={}
    trnegcl=[]
    newtrneg=[]
    newtrnegcl=[]
    negratio=[-1]
    posratio=[-1]
    nexratio=[-1]
    fobj=[]
    cumsize=None
    last_round=False
    w=None
    oldprloss=numpy.zeros((0,6))
    totPosEx=0
    for i in range(len(trPosImages)):
            totPosEx += len(trPosImages[i]["bbox"])  
    #totPosEx*=2
    
    pyrHOG2.setK(cfg.k)
    #pyrHOG2.setDENSE(cfg.dense)
    for it in range(cfg.posit):
        if last_round:
            print "Finished!!!!"
            break
        #trpos=[]
        #numoldtrpos=len(trpos)
        #trpos=fulltrpos
        #trposcl=fulltrposcl
        #numoldtrposcl=len(trposcl)
        #trposcl=[]
        #just for test
        newtrneg=[]
        newtrnegcl=[]
        cntnochange=0
        cntgoodchnage=0
        cntkeepoldscr=0
        cntkeepoldbb=0
        cntnotused=0
        cntadded=0
        #clear()
        partime=time.time()
        print "Positive Images:"
        if cfg.bestovr and it==0:#force to take best overlapping
            cfg.mpos=10
        else:
            cfg.mpos=0#0.5
        cfgpos=copy.copy(cfg)
        cfgpos.numneg=cfg.numneginpos
        #arg=[[i,trPosImages[i]["name"],trPosImages[i]["bbox"],models,cfgpos] for i in range(len(trPosImages))]
        if cfg.useRL:      
            arg = []
            for i in range(len(trPosImages)):
                arg.append([i,trPosImages[i]["name"],trPosImages[i]["bbox"],models,cfgpos,False]) 
                arg.append([i,trPosImages[i]["name"],trPosImages[i]["bbox"],models,cfgpos,True])
        else:
            arg=[[i,trPosImages[i]["name"],trPosImages[i]["bbox"],models,cfgpos] for i in range(len(trPosImages))]
        t=time.time()
        #mypool = Pool(numcore)
        if not(cfg.multipr):
            itr=itertools.imap(detectWrap,arg)        
        else:
            #res=mypool.map(detectWrap,arg)
            itr=mypool.imap(detectWrap,arg)
        numbb=0
        for ii,res in enumerate(itr):
            totneg=0
            fuse=[]
            fuseneg=[]
            for mix in res:
                #trpos+=res[3]
                tr=mix[0]
                fuse+=mix[1]
                fuseneg+=mix[2]
                #ineg=tr.descr(mix[2],flip=False)
                #newtrneg+=ineg
                #totneg+=len(ineg)
                #newtrnegcl+=tr.mixture(mix[2])
                #if cfg.useflineg:
                #    inegflip=tr.descr(mix[2],flip=True)
                #    newtrneg+=inegflip
                #    newtrnegcl+=tr.mixture(mix[2])
            #for h in fuse:
            #    h["scr"]+=models[h["cl"]]["ra"]
            rfuse=tr.rank(fuse,maxnum=1000)
            rfuseneg=tr.rank(fuseneg,maxnum=1000)
            nfuse=tr.cluster(rfuse,ovr=cfg.ovrasp)
            nfuseneg=tr.cluster(rfuseneg,ovr=cfg.ovrasp)
            #imname=arg[ii][1].split("/")[-1]
            flipstr=""
            if len(arg[ii])>5:
                if arg[ii][5]:
                    flipstr="_filp"
            imname=arg[ii][1].split("/")[-1]+flipstr
            ineg=tr.descr(nfuseneg,flip=False,usemrf=cfg.usemrf,usefather=cfg.usefather,k=cfg.k)
            newtrneg+=ineg
            if not(cfg.useRL):
                inegflip=tr.descr(nfuseneg,flip=True,usemrf=cfg.usemrf,usefather=cfg.usefather,k=cfg.k)
                newtrneg+=inegflip
                newtrnegcl+=tr.mixture(nfuseneg)     
            newtrnegcl+=tr.mixture(nfuseneg)     
            if it==0:
                #ipos=tr.descr(nfuse,flip=False,usemrf=cfg.usemrf,usefather=cfg.usefather,k=cfg.k)
                #if not(cfg.useRL):
                #    iposflip=tr.descr(nfuse,flip=True,usemrf=cfg.usemrf,usefather=cfg.usefather,k=cfg.k)
                poscl=tr.mixture(nfuse)
                #if trpos.has_key(imname):
                if nfuse!=[]:
                    print "Added %d examples!"%(len(nfuse))
                    dtrpos[imname]=nfuse#[:]
                    cntadded+=len(nfuse)
                else:
                    print "Example not used!"
                #else:
                #    trpos["imname"]=[]
                #trpos+=iposflip
                #trposcl+=poscl
                #trposcl+=poscl
            #datamining positives
            else:
                nb=len(nfuse)
                #dns=buildense(ipos,poscl,cumsize)
                for idel,dt in enumerate(nfuse):
                    #print "BBox:",numbb
                    if dt!=[] and not(dt.has_key("notfound")): #if got a detection for the bbox
                        #if not(dtrpos.has_key(dt["img"])): 
                        if not(dtrpos.has_key(imname)): 
                            print "Added example previuosly empty!"
                            dtrpos[imname]=[dt]#copy.deepcopy(dt)]
                            cntadded+=1
                        else:
                            exmatch=False
                            print "DET BB ID:",dt["bbid"]
                            for idold,dtold in enumerate(dtrpos[imname]):
                                if dt["bbid"]==dtold["bbid"]:
                                    print "OLD BB ID:",dtold["bbid"]
                                    exmatch=True
                                    aux=tr.descr([dtold],flip=False,usemrf=cfg.usemrf,usefather=cfg.usefather,k=cfg.k)[0]
                                    auxcl=tr.mixture([dtold])[0]
                                    dns=buildense([aux],[auxcl],cumsize)
                                    oldscr=numpy.sum(dns*w)#-r
                                    dtaux=tr.descr([dt],flip=False,usemrf=cfg.usemrf,usefather=cfg.usefather,k=cfg.k)[0]
                                    dtauxcl=tr.mixture([dt])[0]
                                    dtdns=buildense([dtaux],[dtauxcl],cumsize)
                                    newscr=numpy.sum(dtdns*w)#-r
                                    if abs(newscr-dt["scr"])>0.0001:
                                        print "Warning dense score and scan score are different!!!!"
                                        raw_input() 
                                    print "New:",dt["scr"],"Old",oldscr
                                    if abs(dt["scr"]-oldscr)<0.0001:
                                        print "No change between old and new labeling! (%f)"%(dt["scr"])
                                        cntnochange+=1
                                        dtrpos[imname][idold]["scr"]=oldscr
                                    elif dt["scr"]-oldscr>0:
                                        print "New score (%f) higher than previous (%f)!"%(dt["scr"],oldscr)
                                        cntgoodchnage+=1
                                        #dtold=dt
                                        dtrpos[imname][idold]=dt#copy.deepcopy(dt)
                                        #dtrpos[imname][idold]["dns"]=dtdns
                                    else:
                                        print "Keep old example (%f) because better score than new (%f)!"%(oldscr,dt["scr"])                      
                                        cntkeepoldscr+=1     
                                        dtrpos[imname][idold]["scr"]=oldscr
                                    
                            if not(exmatch):    
                                print "Added example!!!!"
                                cntadded=0
                                dtrpos[imname].append(dt)#copy.deepcopy(dt))
                    else: #or skip and keep the old
                        if dtrpos.has_key(imname):
                            for ll in dtrpos[imanme]:
                                if ll["bbid"]==dt["bbid"]:
                                    print "***Keep old example because better overlapping!"
                                    cntkeepoldbb+=1
                            else:
                                print "***Example not used!"
                                cntnotused+=1
                        else:       
                            print "****Example not used!"
                            cntnotused+=1
                numbb+=len(nfuse)*2
                #raw_input()    
            print "----Pos Image %d(%s)----"%(ii,imname)
            print "Pos:",len(nfuse),"Neg:",len(nfuseneg)
            print "Tot Pos:",len(dtrpos)," Neg:",len(trneg)+len(newtrneg)
            #check score
            if (nfuse!=[] and not(nfuse[0].has_key("notfound")) and it>0):
                #if cfg.deform:
                aux=tr.descr(nfuse,usemrf=cfg.usemrf,usefather=cfg.usefather,k=cfg.k)[0]
                #else:
                #    aux=tr.descr(nfuse)[0]
                auxcl=tr.mixture(nfuse)[0]
                dns=buildense([aux],[auxcl],cumsize)[0]
                dscr=numpy.sum(dns*w)
                #print "Scr:",nfuse[0]["scr"],"DesneSCR:",dscr,"Diff:",abs(nfuse[0]["scr"]-dscr)
                if abs(nfuse[0]["scr"]-dscr)>0.0001:
                    print "Warning: the two scores must be the same!!!"
                    print "Scr:",nfuse[0]["scr"],"DesneSCR:",dscr,"Diff:",abs(nfuse[0]["scr"]-dscr)
                    raw_input()
            #check score
            if (nfuseneg!=[] and it>0):
                if cfg.deform:
                    aux=tr.descr(nfuseneg,usemrf=cfg.usemrf,usefather=cfg.usefather,k=cfg.k)[0]
                else:
                    aux=tr.descr(nfuseneg)[0]
                auxcl=tr.mixture(nfuseneg)[0]
                dns=buildense([aux],[auxcl],cumsize)[0]
                dscr=numpy.sum(dns*w)
                #print "Scr:",nfuseneg[0]["scr"],"DesneSCR:",dscr,"Diff:",abs(nfuseneg[0]["scr"]-dscr)
                if abs(nfuseneg[0]["scr"]-dscr)>0.0001:
                    print "Warning: the two scores must be the same!!!"
                    print "Scr:",nfuseneg[0]["scr"],"DesneSCR:",dscr,"Diff:",abs(nfuseneg[0]["scr"]-dscr)
                    raw_input()
            if cfg.show and 0:
                pylab.figure(20)
                pylab.ioff()
                pylab.clf()
                pylab.axis("off")
                #pylab.axis("image")
                #img=util.myimread(trPosImages[ii/2]["name"],flip=ii%2)
                img=util.myimread(arg[ii][1],flip=ii%2)
                pylab.imshow(img,interpolation="nearest",animated=True)
                tr.show(nfuse,parts=cfg.show)      
                pylab.show()
                raw_input()
        del itr
    
        #fulltrpos=trpos
        #fulltrposcl=trposcl
        #trpos=remove_empty(fulltrpos)
        #trposcl=remove_empty(fulltrposcl)
        numoldtrpos=len(trpos)
        if it>0:
            moldloss,oldscr=loss_pos(trpos,trposcl,cumsize)
        else:
            moldloss=1
        oldtrpos=trpos;oldtrposcl=trposcl
        trpos,trposcl=extract_feat(tr,dtrpos,cumsize,cfg.useRL)
        if it>0:
            mnewloss,newscr=loss_pos(trpos,trposcl,cumsize)
        else:
            mnewloss=0
        if it>0:
            oldscr=numpy.array(oldscr)
            newscr=numpy.array(newscr)
            if len(oldscr)==len(newscr):
                if numpy.any(newscr-oldscr)<0:
                    print "Error, score is decreasing"
                    print newscr-oldscr

        print "Added",cntadded
        print "No change",cntnochange
        print "Good change",cntgoodchnage
        print"Keep old small scr",cntkeepoldscr
        print "Keep old bbox",cntkeepoldbb
        print "Not used",cntnotused
        print "Total:",cntnochange+cntgoodchnage+cntkeepoldscr+cntkeepoldbb+cntnotused+cntadded
        stpos.report(cfg.testname+".rpt.txt","a","Positive Datamaining")
        #raw_input()

        #if it==0 and cfg.kmeans:#clustering for LR
        if it==0 and cfg.kmeans:#clustering for LR
            trpos=[];trposcl=[]
            trpos2,trposcl2=extract_feat2(tr,dtrpos,cumsize,False)
            for l in range(numcl):
                mytrpos=[]            
                for c in range(len(trpos2)):
                    if trposcl2[c]==l:
                        mytrpos.append(trpos2[c])
                mytrpos=numpy.array(mytrpos)
                cl1=range(0,len(mytrpos),2)
                cl2=range(1,len(mytrpos),2)
                rrnum=3*len(mytrpos)
                if cfg.cls=="person": #speed-up the clustering for person because too many examples
                    rrnum=len(mytrpos)
                for rr in range(rrnum):
                #for rr in range(1000):
                    print "Clustering iteration ",rr
                    oldvar=numpy.sum(numpy.var(mytrpos[cl1],0))+numpy.sum(numpy.var(mytrpos[cl2],0))
                    #print "Variance",oldvar
                    #print "Var1",numpy.sum(numpy.var(mytrpos[cl1],0))
                    #print "Var2",numpy.sum(numpy.var(mytrpos[cl2],0))
                    #c1=numpy.mean(mytrpos[cl1])
                    #c2=numpy.mean(mytrpos[cl1])
                    rel=numpy.random.randint(len(cl1))
                    tmp=cl1[rel]
                    cl1[rel]=cl2[rel]
                    cl2[rel]=tmp
                    newvar=numpy.sum(numpy.var(mytrpos[cl1],0))+numpy.sum(numpy.var(mytrpos[cl2],0))
                    if newvar>oldvar:#go back
                        tmp=cl1[rel]
                        cl1[rel]=cl2[rel]
                        cl2[rel]=tmp
                    else:
                        print "Variance",newvar
                print "Elements Cluster ",l,": ",len(cl1)
                trpos+=(mytrpos[cl1]).tolist()
                trposcl+=([l]*len(cl1))
                    
        if it>0:
            
            #lambd=1.0/(len(trpos)*cfg.svmc)
            #trpos,trneg,trposcl,trnegcl,clsize,w,lamda
            posl,negl,reg,nobj,hpos,hneg=pegasos.objective(trpos,trneg,trposcl,trnegcl,clsize,w,cfg.svmc)
            #oldloss=prloss[-1][0]*numoldtrpos+(totPosEx-numoldtrpos)*(1-cfg.thr)
            #newloss=posl*len(trpos)+(totPosEx-len(trpos))*(1-cfg.thr)
            oldloss=moldloss+(totPosEx-numoldtrpos)*(1-cfg.thr)
            newloss=mnewloss+(totPosEx-len(trpos))*(1-cfg.thr)
            print "IT:",it,"OLDPOSLOSS",oldloss,"NEWPOSLOSS:",newloss
            posratio.append((oldloss-newloss)/oldloss)
            nexratio.append(float(abs(len(trpos)-numoldtrpos))/numoldtrpos)
            print "RATIO: abs(oldpos-newpos)/oldpos:",posratio
            print "N old examples:",numoldtrpos,"N new examples",len(trpos),"ratio",nexratio
            #fobj.append(nobj)
            #print "OBj:",fobj
            #raw_input()
            output="Not converging yet!"
            if newloss>oldloss:
                output+="Warning increasing positive loss\n"
                print output
                raw_input()
            if (posratio[-1]<0.0001) and nexratio[-1]<0.01:
                output+="Very small positive improvement: convergence at iteration %d!"%it
                print output
## for the moment skip positive convergency
##                last_round=True                
            stloss.report(cfg.testname+".rpt.txt","a","Positive Convergency")
            #if (posratio[-1]<0.0001) and nexratio[-1]<0.01:
                #continue

        if it==cfg.posit-1:#last iteration
            last_round=True #use all examples
        #delete doubles
        newtrneg2,newtrnegcl2=myunique(trneg,newtrneg,trnegcl,newtrnegcl,cfg.numcl)
        trneg=trneg+newtrneg2
        trnegcl=trnegcl+newtrnegcl2
        
        #negative retraining
        trneglen=1
        newPositives=True
        if last_round:
            trNegImages=trNegImagesFull
            #cfg.maxneg=5000
        for nit in range(cfg.negit):
            if nit==0 and it>0:
                print "Skipping searching more negatives in the first iteration"
            else:
                newtrneg=[]
                newtrnegcl=[]
                print "Negative Images Iteration %d:"%nit
                #print numpy.who()
                #raw_input()
                limit=False
                cfgneg=copy.copy(cfg)
                cfgneg.numneg=10#cfg.numneginpos
                cfgneg.thr=-1
                nparts=10
                #nparts=cfg.maxneg/16
                t=time.time()
                order=range(nparts)#numpy.random.permutation(range(nparts))
                for rr in order:
                    arg=[[i,trNegImages[i]["name"],trNegImages[i]["bbox"],models,cfgneg] for i in range(rr*len(trNegImages)/nparts,(rr+1)*len(trNegImages)/nparts)]
                    print "PART:%d Elements:%d"%(rr,len(arg))
                    #arg=[[i,trNegImages[i]["name"],trNegImages[i]["bbox"],models,cfgneg] for i in range((it+1)*len(trNegImages)/(10))]
                    t=time.time()
                    if not(cfg.multipr):
                        itr=itertools.imap(detectWrap,arg)        
                    else:
                        itr=mypool.imap(detectWrap,arg)
                    for ii,res in enumerate(itr):
                        totneg=0
                        fuse=[]
                        fuseneg=[]
                        for mix in res:
                            #trpos+=res[3]
                            tr=mix[0]
                            fuse+=mix[1]
                            fuseneg+=mix[2]
                            #ineg=tr.descr(mix[2],flip=False)
                            #newtrneg+=ineg
                            #totneg+=len(ineg)
                            #newtrnegcl+=tr.mixture(mix[2])
                            #if cfg.useflineg:
                            #    inegflip=tr.descr(mix[2],flip=True)
                            #    newtrneg+=inegflip
                            #    newtrnegcl+=tr.mixture(mix[2])
                        rfuse=tr.rank(fuse,maxnum=1000)
                        rfuseneg=tr.rank(fuseneg,maxnum=1000)
                        nfuse=tr.cluster(rfuse,ovr=cfg.ovrasp)
                        nfuseneg=tr.cluster(rfuseneg,ovr=cfg.ovrasp)
                        #if cfg.deform:
                        #trpos+=tr.descr(nfuse,usemrf=cfg.usemrf,usefather=cfg.usefather)         
                        newtrneg+=tr.descr(nfuseneg,usemrf=cfg.usemrf,usefather=cfg.usefather,k=cfg.k)
                        #else:
                        #trpos+=tr.descr(nfuse)         
                        #    newtrneg+=tr.descr(nfuseneg)
                        #trposcl+=tr.mixture(nfuse)
                        newtrnegcl+=tr.mixture(nfuseneg)
                        #if cfg.useflipos:
                        #    if cfg.deform:
                        #        iposflip=tr.descr(nfuse,flip=True,usemrf=cfg.usemrf,usefather=cfg.usefather)
                        #    else:
                        #        iposflip=tr.descr(nfuse,flip=True)
                        #    trpos+=iposflip
                        #    trposcl+=tr.mixture(nfuse)
                        if cfg.useflineg and not(cfg.useRL):
                            if cfg.deform:
                                inegflip=tr.descr(nfuseneg,flip=True,usemrf=cfg.usemrf,usefather=cfg.usefather,k=cfg.k)
                            else:
                                inegflip=tr.descr(nfuseneg,flip=True)
                            newtrneg+=inegflip
                            newtrnegcl+=tr.mixture(nfuseneg)
                        #check score
                        if (nfuseneg!=[] and nit>0):
                            if cfg.deform:
                                aux=tr.descr(nfuseneg,usemrf=cfg.usemrf,usefather=cfg.usefather,k=cfg.k)[0]
                            else:
                                aux=tr.descr(nfuseneg)[0]
                            auxcl=tr.mixture(nfuseneg)[0]
                            dns=buildense([aux],[auxcl],cumsize)[0]
                            dscr=numpy.sum(dns*w)
                            if abs(nfuseneg[0]["scr"]-dscr)>0.0001:
                                print "Warning: the two scores must be the same!!!"
                                print "Scr:",nfuseneg[0]["scr"],"DesneSCR:",dscr,"Diff:",abs(nfuseneg[0]["scr"]-dscr)
                                raw_input()

                        print "----Neg Image %d----"%ii
                        print "Pos:",0,"Neg:",len(nfuseneg)
                        print "Tot Pos:",len(trpos)," Neg:",len(trneg)+len(newtrneg)
                    if len(newtrneg)+len(trneg)+len(trpos)>cfg.maxexamples:
                        print "Cache Limit Reached!"
                        limit=True
                        break
                del itr
            ##print len(trneg),trneglen
            ##if len(trneg)/float(trneglen)<1.2 and not(limit):
            ##    print "Not enough negatives, convergence!"
            ##    break

            #delete doubles
            newtrneg2,newtrnegcl2=myunique(trneg,newtrneg,trnegcl,newtrnegcl,cfg.numcl)
            trneg=trneg+newtrneg2
            trnegcl=trnegcl+newtrnegcl2

            #if len(trneg)/float(trneglen)<1.05 and not(limit):
            if len(newtrneg2)==0 and not(newPositives) and not(limit):
                print "Not enough negatives, convergence!"
                break
            newPositives=False

            print "Building Feature Vector"
            clsize=numpy.zeros(numcl,dtype=numpy.int)#get clusters sizes
            cumsize=numpy.zeros(numcl+1,dtype=numpy.int)
            for l in range(numcl):
                npcl=(l,numpy.sum(numpy.array(trposcl)==l))
                nncl=(l,numpy.sum(numpy.array(trnegcl)==l))
                print "Pos Samples Cluster %d: %d"%npcl
                print "Neg Samples Cluster %d: %d"%nncl
                clst.report(testname+".rpt.txt","a","Cluster Statistics:")
                c=0
                while trnegcl[c]!=l:
				    c+=1
                clsize[l]=len(trneg[c])+1
                cumsize[l+1]=numpy.sum(clsize[:l+1])

            #show pos examples
            if 0:
                pylab.figure(23)
                pylab.clf()
                util.showExamples(ftrpos,fy,fx)
                pylab.draw()
                pylab.show()
                #raw_input()

            #check negative loss
            if nit>0 and not(limit):
                #lambd=1.0/(len(trpos)*cfg.svmc)
                #trpos,trneg,trposcl,trnegcl,clsize,w,lamda
                posl,negl,reg,nobj,hpos,hneg=pegasos.objective(trpos,trneg,trposcl,trnegcl,clsize,w,cfg.svmc)
                print "NIT:",nit,"OLDLOSS",old_nobj,"NEWLOSS:",nobj
                negratio.append(nobj/(old_nobj+0.000001))
                print "RATIO: newobj/oldobj:",negratio
                output="Negative not converging yet!"
                if (negratio[-1]<1.05):
                    output= "Very small negative newloss: convergence at iteration %d!"%nit
                    #raw_input()
                    #break
                stloss.report(cfg.testname+".rpt.txt","a","Negative Convergency")
                if (negratio[-1]<1.05):
                    break
            #else:
            #    negl=1

            print "SVM learning"
            svmname="%s.svm"%testname
            #lib="libsvm"
            lib="linear"
            #lib="linearblock"
            #pc=0.008 #single resolution
            #pc=cfg.svmc #high res
            #util.trainSvmRaw(ftrpos,ftrneg,svmname,dir="",pc=pc,lib=lib)
            #util.trainSvmRaw(ftrneg,ftrpos,svmname,dir="",pc=pc,lib=lib)
            #w,r=util.loadSvm(svmname,dir="",lib=lib)
            #w,r=util.trainSvmRawPeg(ftrpos,ftrneg,testname+".rpt.txt",dir="",pc=pc)

            import pegasos
            if w==None: 
                #w=numpy.zeros(cumsize[-1])
                w=numpy.random.rand(cumsize[-1])
                w=w/numpy.sqrt(numpy.sum(w**2))

            noise=False
            if noise:
                noiselev=0.5*(1-float(it)/(cfg.posit-1))+cfg.noiselev*(float(it)/(cfg.posit-1))
                atrpos=numpy.array(trpos,dtype=object)
                atrposcl=numpy.array(trposcl,dtype=object)
                oldoutlyers=numpy.zeros(len(trpos),dtype=numpy.int)
                newoutlyers=numpy.zeros(len(trpos),dtype=numpy.int)
                for ii in range(10):
                    lscr=[]
                    dns=buildense(trpos,trposcl,cumsize)
                    for f in dns:
                        lscr.append(numpy.sum(f*w))
                    ordered=numpy.argsort(lscr)             
                    ntrpos=atrpos[ordered][len(trpos)*noiselev:len(trpos)]
                    ntrposcl=atrposcl[ordered][len(trposcl)*noiselev:len(trposcl)]
                    #w,r,prloss=pegasos.trainComp(ntrpos,trneg,testname+"loss.rpt.txt",ntrposcl,trnegcl,oldw=w,dir="",pc=cfg.svmc,k=10,numthr=numcore)
                    w,r,prloss=trainParallel(trpos,trneg,testname,trposcl,trnegcl,w,cfg.svmc,cfg.multipr,parallel=True,numcore=numcore)
                    newoutlyers[ordered[:len(trpos)*noiselev]]=1
                    numout=numpy.sum(numpy.bitwise_and(newoutlyers,oldoutlyers))
                    print ordered
                    print ordered[:len(trpos)*noiselev]#ntrpos[:len(trpos)*noiselev]
                    print numout
                    if numout>=int(len(trpos)*noiselev*0.95):
                        print 'Converging because ',numout,'/',int(len(trpos)*noiselev*0.95) 
                        break
                    else:
                        print 'Not congergin yet because',numout,'/',int(len(trpos)*noiselev*0.95) 
                    oldoutlyers=newoutlyers.copy()
                    #raw_input()
            else:
                w,r,prloss=trainParallel(trpos,trneg,testname,trposcl,trnegcl,w,cfg.svmc,cfg.multipr,parallel=True,numcore=numcore)

            old_posl,old_negl,old_reg,old_nobj,old_hpos,old_hneg=pegasos.objective(trpos,trneg,trposcl,trnegcl,clsize,w,cfg.svmc)            
            #pylab.figure(300)
            #pylab.clf()
            #pylab.plot(w)
            pylab.figure(500)
            pylab.clf()
            oldprloss=numpy.concatenate((oldprloss,prloss),0)
            pylab.plot(oldprloss)
            pylab.semilogy()
            pylab.legend(["loss+","loss-","reg","obj","hard+","hard-"],loc='upper left')
            pylab.savefig("%s_loss%d.pdf"%(cfg.testname,it))
            pylab.draw()
            pylab.show()

            bias=100
            for idm,m in enumerate(models):
                models[idm]=tr.model(w[cumsize[idm]:cumsize[idm+1]-1],-w[cumsize[idm+1]-1]*bias,len(m["ww"]),31,m["fy"],m["fx"],usemrf=cfg.usemrf,usefather=cfg.usefather)
                #models[idm]["ra"]=w[cumsize[idm+1]-1]
            util.save("%s%d.model"%(testname,it),models)
            if cfg.deform:
                print m["df"]

            if True:
                print "Show model"
                for idm,m in enumerate(models):    
                    pylab.figure(100+idm)
                    pylab.clf()
                    util.drawModel(m["ww"])
                    pylab.draw()
                    pylab.show()
                    pylab.savefig("%s_hog%d_cl%d.png"%(testname,it,idm))
                    if cfg.deform:
                        pylab.figure(110+idm)
                        pylab.clf()
                        util.drawDeform(m["df"])
                        pylab.draw()
                        pylab.show()
                        pylab.savefig("%s_def%d_cl%d.png"%(testname,it,idm))

            sts.report(testname+".rpt.txt","a","Before Filtering")
            #sort based on score
            #sort=True
            if cfg.sortneg:
                order=[]
                for p,d in enumerate(trneg):
                    order.append(numpy.dot(buildense([d],[trnegcl[p]],cumsize)[0],w))
                order=numpy.array(order)
                sorder=numpy.argsort(order)
                strneg=[]
                strnegcl=[]
                for p in sorder:
                    strneg.append(trneg[p])
                    strnegcl.append(trnegcl[p])
                trneg=strneg
                trnegcl=strnegcl
            #else:
            #    sorder=range(len(trneg))

            print "Filter Data"
            print "Length before:",len(trneg)
            for p,d in enumerate(trneg):
            #for p in sorder:
                aux=buildense([trneg[p]],[trnegcl[p]],cumsize)[0]
                if numpy.sum(aux*w)<-1:
                    if len(trneg)+len(trpos)>(cfg.maxexamples)/2:
                        trneg.pop(p)
                        trnegcl.pop(p)
            print "Length after:",len(trneg)
            trneglen=len(trneg)
            sts.report(testname+".rpt.txt","a","After Filtering")
     
        print "Test"
        if last_round:
            tsImages=tsImagesFull
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
            #for h in fuse:
            #    h["scr"]+=models[h["cl"]]["ra"]
            rfuse=tr.rank(fuse,maxnum=300)
            nfuse=tr.cluster(rfuse,ovr=cfg.ovrasp)
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
                tr.show(nfuse,parts=showlabel,thr=-0.5,maxnum=10)           
                pylab.show()
        del itr
        
        #tp,fp,scr,tot=VOCpr.VOCprlistfastscore(tsImages,detlist,numim=cfg.maxpostest,show=False,ovr=0.5)
        tp,fp,scr,tot=VOCpr.VOCprRecord(tsImages,detlist,show=False,ovr=0.5)
        pylab.figure(15)
        pylab.clf()
        rc,pr,ap=VOCpr.drawPrfast(tp,fp,tot)
        pylab.draw()
        pylab.show()
        pylab.savefig("%s_ap%d.png"%(testname,it))
        #util.savemat("%s_ap%d.mat"%(testname,it),{"tp":tp,"fp":fp,"scr":scr,"tot":tot,"rc":rc,"pr":pr,"ap":ap})
        tinit=((time.time()-initime)/3600.0)
        tpar=((time.time()-partime)/3600.0)
        print "AP(it=",it,")=",ap
        print "Training Time: %.3f h"%(tinit)
        rpres.report(testname+".rpt.txt","a","Results")



