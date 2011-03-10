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


#get image name and bbox
def extractInfo(trPosImages,maxnum=-1,usetr=True,usedf=False):
    bb=numpy.zeros((len(trPosImages)*5,4))#as maximum 5 persons per image in average
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

def detectWrap(a):
    i=a[0]
    imname=a[1]
    bbox=a[2]
    models=a[3]
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
    notsave=False
    #if cfg.__dict__.has_key("test"):
    #    notsave=cfg.test
    f=pyrHOG2.pyrHOG(imname,interv=10,savedir=cfg.auxdir+"/hog/",notsave=not(cfg.savefeat),notload=not(cfg.loadfeat),hallucinate=cfg.hallucinate,cformat=True)
    res=[]
    for clm,m in enumerate(models):
        res.append(pyrHOG2.detect(f,m,gtbbox,hallucinate=cfg.hallucinate,initr=cfg.initr,ratio=cfg.ratio,deform=cfg.deform,bottomup=cfg.bottomup,usemrf=cfg.usemrf,numneg=cfg.numneg,thr=cfg.thr,posovr=0.7,minnegincl=0,small=False,show=cfg.show,cl=clm,mythr=cfg.mythr,mpos=cfg.mpos))
    if cfg.show:
        pylab.show()
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

if __name__=="__main__":

    #cfg=config()
    try:
        from config_local_pascal import * #your own configuration
    except:
        print "config_local.py is not present loading configdef.py"
        from config import * #default configuration  
        
    if cfg.savedir=="":
        #cfg.savedir=InriaPosData(basepath=cfg.dbpath).getStorageDir() #where to save
        cfg.savedir=VOC07Data(basepath=cfg.dbpath).getStorageDir()

    import sys
    if len(sys.argv)>1:
        cfg.cls=sys.argv[1]
 
    cfg.testname=cfg.testpath+cfg.cls+("%d"%cfg.numcl)+"_"+cfg.testspec

    util.save(cfg.testname+".cfg",cfg)

#    cfg.fy=[3,4]#[7,]#[3,4]
#    cfg.fx=[6,5]#[11,]#[6,5]
#    cfg.lev=[3,3,3]
#    cfg.numcl=2
    #cfg.interv=10
    #cfg.ovr=0.45
    #cfg.sbin=8
    #cfg.maxpos=1000#120
    #cfg.maxtest=1000#100
    #cfg.maxneg=1000#120
    #cfg.maxexamples=10000
    #cfg.deform=True
    #cfg.usemrf=True#True
    #cfg.usefather=True#Flase
    #cfg.bottomup=False
    #cfg.initr=1
    #cfg.ratio=1
    #cfg.mpos=1
    #cfg.hallucinate=1
    #cfg.numneginpos=6/cfg.numcl
    #cfg.useflipos=True
    #cfg.useflineg=True
    #cfg.multipr=4
    #cfg.svmc=0.005#0.002#0.004
    #cfg.cls="bicycle"
    #cfg.year="2007"
    #cfg.show=False
    #cfg.thr=-2
    #cfg.mythr=-10
    #cfg.auxdir="/home/databases/VOC2007/VOCdevkit/local/VOC2007/"#"/state/partition1/marcopede/"#InriaPosData(basepath="/home/databases/").getStorageDir()
    cfg.auxdir=cfg.savedir
    #testname="./data/11_03_01/%s_%d_test"%(cfg.cls,cfg.numcl)
    testname=cfg.testname
    #util.save(testname+".cfg",cfg)

    #mydebug=False
    #if mydebug:
    #    cfg.show=False
    #    cfg.multipr=4
    #    cfg.maxpos=10
    #    cfg.maxneg=10
    #    cfg.maxtest=10
    #    cfg.maxexamples=1000
    #    #testname=testname+"_debug"

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

    #trPosImages=InriaPosData(basepath="/home/databases/")
    #trNegImages=InriaNegData(basepath="/home/databases/")
    #tsImages=InriaTestFullData(basepath="/home/databases/")
    trPosImages=getRecord(VOC07Data(select="pos",cl="%s_trainval.txt"%cfg.cls,
                    basepath=cfg.dbpath,#"/home/databases/",
                    trainfile="VOC2007/VOCdevkit/VOC2007/ImageSets/Main/",
                    imagepath="VOC2007/VOCdevkit/VOC2007/JPEGImages/",
                    annpath="VOC2007/VOCdevkit/VOC2007/Annotations/",
                    local="VOC2007/VOCdevkit/local/VOC2007/",
                    usetr=True,usedf=False),cfg.maxpos)
    trNegImages=getRecord(VOC07Data(select="neg",cl="%s_trainval.txt"%cfg.cls,
                    basepath=cfg.dbpath,#"/home/databases/",#"/share/ISE/marcopede/database/",
                    trainfile="VOC2007/VOCdevkit/VOC2007/ImageSets/Main/",
                    imagepath="VOC2007/VOCdevkit/VOC2007/JPEGImages/",
                    annpath="VOC2007/VOCdevkit/VOC2007/Annotations/",
                    local="VOC2007/VOCdevkit/local/VOC2007/",
                    usetr=True,usedf=False),cfg.maxneg)
    tsImages=getRecord(VOC07Data(select="pos",cl="%s_test.txt"%cfg.cls,
                    basepath=cfg.dbpath,#"/home/databases/",#"/share/ISE/marcopede/database/",
                    trainfile="VOC2007/VOCdevkit/VOC2007/ImageSets/Main/",
                    imagepath="VOC2007/VOCdevkit/VOC2007/JPEGImages/",
                    annpath="VOC2007/VOCdevkit/VOC2007/Annotations/",
                    local="VOC2007/VOCdevkit/local/VOC2007/",
                    usetr=True,usedf=False),cfg.maxtest)



    #cluster bounding boxes
    name,bb,r,a=extractInfo(trPosImages)
    trpos={"name":name,"bb":bb,"ratio":r,"area":a}
    import scipy.cluster.vq as vq
    numcl=cfg.numcl
    perc=10
    minres=10
    minfy=3
    minfx=3
    maxArea=30*(4-cfg.lev[0])
    usekmeans=False

    #using kmeans
    clc,di=vq.kmeans(r,numcl,3)
    cl=vq.vq(r,clc)[0]
    for l in range(numcl):
        print "Cluster kmeans",l,":"
        print "Samples:",len(a[cl==l])
        print "Mean Area:",numpy.mean(a[cl==l])/16.0
        sa=numpy.sort(a[cl==l])
        print "Min Area:",numpy.mean(sa[len(sa)/perc])/16.0
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
        meanA=numpy.mean(a[cl==l])/16.0/(0.5*4**(cfg.lev[l]-1))#4.0
        print "Mean Area:",meanA
        sa=numpy.sort(a[cl==l])
        minA=numpy.mean(sa[len(sa)/perc])/16.0/(0.5*4**(cfg.lev[l]-1))#4.0
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

    #raw_input()

    cfg.fy=lfy#[7,10]#lfy
    cfg.fx=lfx#[11,7]#lfx

    import time
    initime=time.time()
    #intit model

    mypool = Pool(numcore)
    
    models=[]
    for c in range(numcl):
        fy=cfg.fy[c]
        fx=cfg.fx[c]
        #lowf1=numpy.random.random((fy,fx,31)).astype(numpy.float32)
        lowf1=numpy.ones((fy,fx,31)).astype(numpy.float32)
        lowf=lowf1/numpy.sqrt(numpy.sum(lowf1**2))/(fx*fy)
        #lowd=numpy.random.random((1,1,4)).astype(numpy.float32)
        lowd=-numpy.ones((1,1,4)).astype(numpy.float32)
        #midf1=numpy.random.random((2*fy,2*fx,31)).astype(numpy.float32)
        midf1=numpy.ones((2*fy,2*fx,31)).astype(numpy.float32)
        midf=midf1/numpy.sqrt(numpy.sum(midf1**2))/(4*fx*fy)
        #midd=numpy.random.random((2,2,4)).astype(numpy.float32)
        midd=-numpy.ones((2,2,4)).astype(numpy.float32)
        #higf1=numpy.random.random((4*fy,4*fx,31)).astype(numpy.float32)
        higf1=numpy.ones((4*fy,4*fx,31)).astype(numpy.float32)
        higf=higf1/numpy.sqrt(numpy.sum(higf1**2))/(16*fx*fy)
        #higd=numpy.random.random((4,4,4)).astype(numpy.float32)
        higd=-numpy.ones((4,4,4)).astype(numpy.float32)
        ww3=[lowf,midf,higf]
        ww2=[midf,higf]
        ww1=[higf]
        ww4=[lowf,midf]
        ww5=[midf]
        ww6=[lowf]
        dd1=[lowd]
        dd2=[lowd,midd]
        dd3=[lowd,midd,higd]
        rho=0
        model1={"ww":ww1,"rho":rho,"ra":0,"df":dd1,"fy":ww1[0].shape[0],"fx":ww1[0].shape[1]}
        model2={"ww":ww2,"rho":rho,"ra":0,"df":dd2,"fy":ww2[0].shape[0],"fx":ww2[0].shape[1]}
        model3={"ww":ww3,"rho":rho,"ra":0,"df":dd3,"fy":ww3[0].shape[0],"fx":ww3[0].shape[1]}
        model4={"ww":ww4,"rho":rho,"ra":0,"df":dd2,"fy":ww4[0].shape[0],"fx":ww4[0].shape[1]}
        model5={"ww":ww5,"rho":rho,"ra":0,"df":dd1,"fy":ww5[0].shape[0],"fx":ww5[0].shape[1]}
        model6={"ww":ww6,"rho":rho,"ra":0,"df":dd1,"fy":ww6[0].shape[0],"fx":ww6[0].shape[1]}
        if cfg.lev[c]==3:
            models.append(model3)
        if cfg.lev[c]==2:
            models.append(model4)
        if cfg.lev[c]==1:
            models.append(model6)
        #lmodels=[model1,model2,model3]
        
    trpos=[]
    trposcl=[]
    trneg=[]
    trnegcl=[]
    newtrneg=[]
    newtrnegcl=[]
    negratio=[]
    posratio=[]
    nexratio=[]
    fobj=[]
    w=None
    oldprloss=numpy.zeros((0,6))
    for it in range(10):
        oldtrpos=trpos[:]
        trpos=[]
        oldtrposcl=trposcl[:]
        trposcl=[]
        #just for test
        newtrneg=[]
        newtrnegcl=[]
        #clear()
        partime=time.time()
        print "Positive Images:"
        if it==0:#force to take best overlapping
            cfg.mpos=10
        else:
            cfg.mpos=0#0.5
        cfgpos=copy.copy(cfg)
        cfgpos.numneg=cfg.numneginpos
        arg=[[i,trPosImages[i]["name"],trPosImages[i]["bbox"],models,cfgpos] for i in range(len(trPosImages))]
        t=time.time()
        #mypool = Pool(numcore)
        if not(cfg.multipr):
            itr=itertools.imap(detectWrap,arg)        
        else:
            #res=mypool.map(detectWrap,arg)
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
            #for h in fuse:
            #    h["scr"]+=models[h["cl"]]["ra"]
            rfuse=tr.rank(fuse,maxnum=1000)
            rfuseneg=tr.rank(fuseneg,maxnum=1000)
            nfuse=tr.cluster(rfuse,ovr=0.5)
            nfuseneg=tr.cluster(rfuseneg,ovr=0.5)
            trpos+=tr.descr(nfuse)         
            newtrneg+=tr.descr(nfuseneg)
            trposcl+=tr.mixture(nfuse)
            newtrnegcl+=tr.mixture(nfuseneg)
            if cfg.useflipos:
                iposflip=tr.descr(nfuse,flip=True)
                trpos+=iposflip
                trposcl+=tr.mixture(nfuse)
            if cfg.useflineg:
                inegflip=tr.descr(nfuseneg,flip=True)
                newtrneg+=inegflip
                newtrnegcl+=tr.mixture(nfuseneg)
            print "----Pos Image %d----"%ii
            print "Pos:",len(nfuse),"Neg:",len(nfuseneg)
            print "Tot Pos:",len(trpos)," Neg:",len(trneg)+len(newtrneg)
            #check score
            if (nfuse!=[] and it>0):
                aux=tr.descr(nfuse)[0]
                auxcl=tr.mixture(nfuse)[0]
                dns=buildense([aux],[auxcl],cumsize)[0]
                dscr=numpy.sum(dns*w)
                #print "Scr:",nfuse[0]["scr"],"DesneSCR:",dscr,"Diff:",abs(nfuse[0]["scr"]-dscr)
                if abs(nfuse[0]["scr"]-dscr)>0.0001:
                    print "Warning: the two scores must be the same!!!"
                    print "Scr:",nfuse[0]["scr"],"DesneSCR:",dscr,"Diff:",abs(nfuse[0]["scr"]-dscr)
                    #raw_input()
            #check score
            if (nfuseneg!=[] and it>0):
                aux=tr.descr(nfuseneg)[0]
                auxcl=tr.mixture(nfuseneg)[0]
                dns=buildense([aux],[auxcl],cumsize)[0]
                dscr=numpy.sum(dns*w)
                #print "Scr:",nfuseneg[0]["scr"],"DesneSCR:",dscr,"Diff:",abs(nfuseneg[0]["scr"]-dscr)
                if abs(nfuseneg[0]["scr"]-dscr)>0.0001:
                    print "Warning: the two scores must be the same!!!"
                    print "Scr:",nfuseneg[0]["scr"],"DesneSCR:",dscr,"Diff:",abs(nfuseneg[0]["scr"]-dscr)
                    #raw_input()
            if cfg.show:
                pylab.figure(20)
                pylab.ioff()
                pylab.clf()
                pylab.axis("off")
                img=util.myimread(trPosImages[ii]["name"])
                pylab.imshow(img,interpolation="nearest",animated=True)
                tr.show(nfuse,parts=cfg.show)      
                pylab.show()
        del itr

        if it>0:
            oldpscr=[]
            for idel,el in enumerate(oldtrpos):
                aux=el
                auxcl=oldtrposcl[idel]
                dns=buildense([aux],[auxcl],cumsize)[0]
                oldpscr.append(numpy.sum(dns*w))
            pscr=[]
            for idel,el in enumerate(trpos):
                aux=el
                auxcl=trposcl[idel]
                dns=buildense([aux],[auxcl],cumsize)[0]
                pscr.append(numpy.sum(dns*w))
            
            pylab.figure(99)
            pylab.clf()
            pylab.plot(oldpscr)
            pylab.plot(pscr)
            #pylab.legend("old","new")
            pylab.show()
            #raw_input()

            lambd=1.0/(len(trpos)*cfg.svmc)
            #trpos,trneg,trposcl,trnegcl,clsize,w,lamda
            posl,negl,reg,nobj,hpos,hneg=pegasos.objective(trpos,trneg,trposcl,trnegcl,clsize,w,lambd)
            print "IT:",it,"OLDPOSLOSS",prloss[-1][0],"NEWPOSLOSS:",posl
            posratio.append(abs(posl-prloss[-1][0])/prloss[-1][0])
            nexratio.append(float(abs(len(trpos)-len(oldtrpos)))/len(oldtrpos)<0.01)
            print "RATIO: abs(oldpos-newpos)/oldpos:",posratio
            print "N old examples:",len(oldtrpos),"N new examples",len(trpos),"ratio",nexratio
            #fobj.append(nobj)
            #print "OBj:",fobj
            #raw_input()
            if posl>prloss[-1][0]:
                print "Warning increasing positive loss"
                #raw_input()
            if (posratio[-1]<0.001) and nexratio[-1]<0.01:
                print "Very small positive improvement: convergence at iteration %d!"%it
                #raw_input()
                break

        #delete doubles
        newtrneg2,newtrnegcl2=myunique(trneg,newtrneg,trnegcl,newtrnegcl,cfg.numcl)
        trneg=trneg+newtrneg2
        trnegcl=trnegcl+newtrnegcl2
        
        #negative retraining
        trneglen=1
        newPositives=True
        for nit in range(10):
            newtrneg=[]
            newtrnegcl=[]
            print "Negative Images Iteration %d:"%nit
            #print numpy.who()
            #raw_input()
            limit=False
            cfgneg=copy.copy(cfg)
            cfgneg.numneg=10#cfg.numneginpos
            cfgneg.thr=-1
            nparts=20
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
                    nfuse=tr.cluster(rfuse,ovr=0.5)
                    nfuseneg=tr.cluster(rfuseneg,ovr=0.5)
                    trpos+=tr.descr(nfuse)         
                    newtrneg+=tr.descr(nfuseneg)
                    trposcl+=tr.mixture(nfuse)
                    newtrnegcl+=tr.mixture(nfuseneg)
                    if cfg.useflipos:
                        iposflip=tr.descr(nfuse,flip=True)
                        trpos+=iposflip
                        trposcl+=tr.mixture(nfuse)
                    if cfg.useflineg:
                        inegflip=tr.descr(nfuseneg,flip=True)
                        newtrneg+=inegflip
                        newtrnegcl+=tr.mixture(nfuseneg)
                    #check score
                    if (nfuseneg!=[] and nit>0):
                        aux=tr.descr(nfuseneg)[0]
                        auxcl=tr.mixture(nfuseneg)[0]
                        dns=buildense([aux],[auxcl],cumsize)[0]
                        dscr=numpy.sum(dns*w)
                        if abs(nfuseneg[0]["scr"]-dscr)>0.0001:
                            print "Warning: the two scores must be the same!!!"
                            print "Scr:",nfuseneg[0]["scr"],"DesneSCR:",dscr,"Diff:",abs(nfuseneg[0]["scr"]-dscr)
                            #raw_input()

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
                lambd=1.0/(len(trpos)*cfg.svmc)
                #trpos,trneg,trposcl,trnegcl,clsize,w,lamda
                posl,negl,reg,nobj,hpos,hneg=pegasos.objective(trpos,trneg,trposcl,trnegcl,clsize,w,lambd)
                print "NIT:",nit,"OLDLOSS",prloss[-1][3],"NEWLOSS:",nobj
                negratio.append(nobj/(prloss[-1][3]+0.000001))
                print "RATIO: newobj/oldobj:",negratio
                #fobj.append(nobj)
                #print "OBj:",fobj
                #raw_input()
                if (negratio[-1]<1.05):
                    print "Very small negative newloss: convergence at iteration %d!"%nit
                    #raw_input()
                    break
            else:
                negl=1

            print "SVM learning"
            svmname="%s.svm"%testname
            #lib="libsvm"
            lib="linear"
            #lib="linearblock"
            #pc=0.008 #single resolution
            pc=cfg.svmc #high res
            #util.trainSvmRaw(ftrpos,ftrneg,svmname,dir="",pc=pc,lib=lib)
            #util.trainSvmRaw(ftrneg,ftrpos,svmname,dir="",pc=pc,lib=lib)
            #w,r=util.loadSvm(svmname,dir="",lib=lib)
            #w,r=util.trainSvmRawPeg(ftrpos,ftrneg,testname+".rpt.txt",dir="",pc=pc)
            import pegasos
            if w==None: 
                w=numpy.zeros(cumsize[-1])
            w,r,prloss=pegasos.trainComp(trpos,trneg,testname+"loss.rpt.txt",trposcl,trnegcl,oldw=w,dir="",pc=pc,k=10,numthr=numcore)
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
            #raw_input()

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
            print "Filter Data"
            print "Length before:",len(trneg)
            for p,d in enumerate(trneg):
                aux=buildense([d],[trnegcl[p]],cumsize)[0]
                if numpy.sum(aux*w)<-1:
                    if len(trneg)+len(trpos)>(cfg.maxexamples)/2:
                        trneg.pop(p)
                        trnegcl.pop(p)
            print "Length after:",len(trneg)
            trneglen=len(trneg)
            sts.report(testname+".rpt.txt","a","After Filtering")
     
        print "Test"
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
            nfuse=tr.cluster(rfuse,ovr=0.5)
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
        print "Training Time: %.3f h"%((tinit)/3600.0)
        rpres.report(testname+".rpt.txt","a","Results")



