import pyrHOG2
import time
import numpy
import pylab

class TreatRL(pyrHOG2.Treat):
    def __init__(self,f,scr,pos,pos1,lr,sample,fy,fx,occl=False,trunc=0):
        pyrHOG2.Treat.__init__(self,f,scr,pos,sample,fy,fx,occl=occl,trunc=trunc)
        self.lr=lr
        self.pose=[pos,pos1]

    def refine(self,ldet):
        """
        refine the localization of the object based on higher resolutions
        """
        rdet=[]
        for item in ldet:
            i=item["i"];cy=item["py"];cx=item["px"];pp=self.lr[i][cy,cx]
            el=item.copy()
            el["ny"]=el["ry"]
            el["nx"]=el["rx"]
            el["rl"]=pp
            mov=numpy.zeros(2)
            el["def"]={"dy":numpy.zeros(self.pose[pp][i].shape[1]),"dx":numpy.zeros(self.pose[pp][i].shape[1])}
            for l in range(self.pos[i].shape[1]):
                aux=self.pose[pp][i][:,l,cy,cx]
                el["def"]["dy"][l]=aux[0]
                el["def"]["dx"][l]=aux[1]
                mov=mov+aux*2**(-l)
            el["ry"]+=mov[0]
            el["rx"]+=mov[1]
            rdet.append(el)
        return rdet

    def show(self,*args,**kargs):#ldet,parts=False,colors=["w","r","g","b"],thr=-numpy.inf,maxnum=numpy.inf):
        """
        show the object detections
        """
        pyrHOG2.Treat.show(self,*args,**kargs)
        ldet=args[0]
        if kargs.has_key("thr"):
            thr=kargs["thr"]
        else:
            thr=-numpy.inf
        for item in ldet:
            if item["scr"]>thr:
                bbox=item["bbox"]
                pylab.text(bbox[3]-10,bbox[2]-10,"%d"%(item["rl"]),bbox=dict(facecolor='w', alpha=0.5),fontsize=10)

    def descr(self,det,flip=False,usemrf=True,usefather=True,k=1.0,usebow=False): 
        """
        convert each detection in a feature descriptor for the SVM
        """  
        ld=[]
        for item in det:
            if item["rl"]==1:
                auxflip=not(flip)
            else:
                auxflip=flip
            dd=pyrHOG2.Treat.descr(self,[item],flip=auxflip,usemrf=usemrf,usefather=usefather,k=k,usebow=usebow)
            ld.append(dd[0])
        return ld


#    def descr(self,det,flip=False,usemrf=True,usefather=True,k=1.0): 
#        """
#        convert each detection in a feature descriptor for the SVM
#        """        
#        ld=[]
#        for item in det:
#            d=numpy.array([])
#            if item["rl"]==1:
#                auxflip=not(flip)
#            else:
#                auxflip=flip
#            for l in range(len(item["feat"])):
#                if not(auxflip):
#                    aux=item["feat"][l]
#                    #print "No flip",aux.shape
#                else:
#                    aux=pyrHOG2.hogflip(item["feat"][l])
#                    #print "Flip",aux.shape
#                d=numpy.concatenate((d,aux.flatten()))
#                if self.occl:
#                    if item["i"]-l*self.interv>=0:
#                        d=nump.concatenate((d,[0.0]))
#                    else:
#                        d=nump.concatenate((d,[1.0]))
#            ld.append(d.astype(numpy.float32))
#        return ld

class TreatDefRL(pyrHOG2.TreatDef):
    def __init__(self,f,scr,pos,pos1,lr,sample,fy,fx,occl=False,trunc=0):
        pyrHOG2.TreatDef.__init__(self,f,scr,pos,sample,fy,fx,occl=occl,trunc=trunc)
        self.lr=lr
        self.pose=[pos,pos1]

    def refine(self,ldet):
        """
            refine the localization of the object based on the position of the parts
        """
        rdet=[]
        for item in ldet:
            i=item["i"];cy=item["py"];cx=item["px"];pp=self.lr[i][cy,cx]
            el=item.copy()
            el["ny"]=el["ry"]
            el["nx"]=el["rx"]
            el["rl"]=pp
            mov=numpy.zeros((1,1,2))
            el["def"]={"dy":[],"dx":[],"ddy":[],"ddx":[],"party":[],"partx":[]}
            for l in range(len(self.pose[pp][i])):
                aux=self.pose[pp][i][l][:,:,:,cy,cx]
                el["def"]["dy"].append(aux[:,:,0])
                el["def"]["dx"].append(aux[:,:,1])
                el["def"]["ddy"].append(aux[:,:,2])
                el["def"]["ddx"].append(aux[:,:,3])
                mov=mov+aux[:,:,:2]*2**(-l)
                el["def"]["party"].append(el["ny"]+mov[:,:,0])
                el["def"]["partx"].append(el["nx"]+mov[:,:,1])
                aux1=numpy.kron(mov.T,[[1,1],[1,1]]).T
                aux2=numpy.zeros((2,2,2))
                aux2[:,:,0]=numpy.array([[0,0],[self.fy*2**-(l+1),self.fy*2**-(l+1)]])
                aux2[:,:,1]=numpy.array([[0,self.fx*2**-(l+1)],[0,self.fx*2**-(l+1)]])
                aux3=numpy.kron(numpy.ones((2**l,2**l)),aux2.T).T
                mov=aux1+aux3
            el["ry"]=numpy.min(el["def"]["party"][-1])
            el["rx"]=numpy.min(el["def"]["partx"][-1])
            el["endy"]=numpy.max(el["def"]["party"][-1])+self.fy*(2**-(l))
            el["endx"]=numpy.max(el["def"]["partx"][-1])+self.fx*(2**-(l))
            rdet.append(el)
        return rdet

    def show(self,*args,**kargs):#ldet,parts=False,colors=["w","r","g","b"],thr=-numpy.inf,maxnum=numpy.inf):
        """
        show the detections in an image
        """        
        pyrHOG2.TreatDef.show(self,*args,**kargs)
        ldet=args[0]
        if kargs.has_key("thr"):
            thr=kargs["thr"]
        else:
            thr=-numpy.inf
        if kargs.has_key("maxnum"):
            maxnum=kargs["maxnum"]
        else:
            maxnum=1000#-numpy.inf	
        for item in ldet[:maxnum+1]:
            if item["scr"]>thr:
                bbox=item["bbox"]
                if item["rl"]==0:
        			rl="R"
                else:
                    rl="L"
                pylab.text(bbox[3]-5,bbox[2]-5,"%s"%(rl),bbox=dict(facecolor='w', alpha=0.5),fontsize=10)
		

    def descr(self,det,flip=False,usemrf=True,usefather=True,k=1.0,usebow=False):   
        """
        convert each detection in a feature descriptor for the SVM
        """  
        ld=[]
        for item in det:
            if item["rl"]==1:
                auxflip=not(flip)
            else:
                auxflip=flip
            dd=pyrHOG2.TreatDef.descr(self,[item],flip=auxflip,usemrf=usemrf,usefather=usefather,k=k,usebow=usebow)
            ld.append(dd[0])
        return ld


def flip(m):
    """
    flip of the object model
    """  
    ww1=[]
    df1=[]
    for l in m["ww"]:
        ww1.append(numpy.ascontiguousarray(pyrHOG2.hogflip(l)))
    m1={"ww":ww1,"rho":m["rho"],"fy":ww1[0].shape[0],"fx":ww1[0].shape[1]}
    if m.has_key("df"):
        for l in m["df"]:
            aux=l.copy()
            aux[:,:,:2]=l[:,::-1,:2]#father flip
            aux[:,:,2]=pyrHOG2.defflip(l[:,:,2])
            aux[:,:,3]=pyrHOG2.defflip(l[:,:,3])
            df1.append(numpy.ascontiguousarray(aux))
        m1["df"]=df1
    if m.has_key("occl"):
        m1["occl"]=m["occl"]
    #for BOW
    hh1=[]
    if m.has_key("hist"):
        for idl,l in enumerate(m["hist"]):
            #hh1.append(numpy.ascontiguousarray(l[::-1]))
            if len(l.shape)>1:
                auxh=numpy.zeros((2**idl,2**idl,l.shape[2]),dtype=numpy.float32)
                for py in range(l.shape[0]):
                    for px in range(l.shape[1]):
                        auxh[py,2**idl-px-1,:]=l[py,px,pyrHOG2.histflip()]
                hh1.append(auxh)
            else:
                hh1.append(l[pyrHOG2.histflip()].copy())
            #hh1.append((l[pyrHOG2.histflip()]).astype(numpy.float32))
        m1["hist"]=hh1
    vv1=[]
    if m.has_key("voc"):
        for l in m["voc"]:
            vv1.append(numpy.ascontiguousarray(l[::-1]))
        m1["voc"]=vv1
    return m1    


def detectflip(f,m,gtbbox=None,auxdir=".",hallucinate=1,initr=1,ratio=1,deform=False,bottomup=False,usemrf=False,numneg=0,thr=-2,posovr=0.7,minnegincl=0.5,small=True,show=False,cl=0,mythr=-10,nms=0.5,inclusion=False,usefather=True,mpos=1,useprior=False,K=1.0,occl=False,trunc=0,useMaxOvr=False,ranktr=1000,fastBU=False,usebow=False):
    """Detect objects with RL flip
        used for both test --> gtbbox=None
        and trainig --> gtbbox = list of bounding boxes
    """
    #build flip
    pyrHOG2.setK(K)
    m1=flip(m)
    t=time.time()        
    if gtbbox!=None and gtbbox!=[] and useprior:
        pr=f.buildPrior(gtbbox,m["fy"],m["fx"])
    else:
        pr=None
    t=time.time()      
    f.resetHOG()
    if deform:
        if bottomup:
            scr,pos=f.scanRCFLDefBU(m,initr=initr,ratio=ratio,small=small,usemrf=usemrf)
            scr1,pos1=f.scanRCFLDefBU(m1,initr=initr,ratio=ratio,small=small,usemrf=usemrf)
        else:
            if usebow:
                scr,pos=f.scanRCFLDefbow(m,initr=initr,ratio=ratio,small=small,usemrf=usemrf,trunc=trunc)
                scr1,pos1=f.scanRCFLDefbow(m1,initr=initr,ratio=ratio,small=small,usemrf=usemrf,trunc=trunc)
            else:
                usethr=True
                if usethr:
                    scr,pos=f.scanRCFLDefThr(m,initr=initr,ratio=ratio,small=small,usemrf=usemrf,mythr=mythr)
                    scr1,pos1=f.scanRCFLDefThr(m1,initr=initr,ratio=ratio,small=small,usemrf=usemrf,mythr=mythr)
                else:
                    scr,pos=f.scanRCFLDef(m,initr=initr,ratio=ratio,small=small,usemrf=usemrf,trunc=trunc)
                    scr1,pos1=f.scanRCFLDef(m1,initr=initr,ratio=ratio,small=small,usemrf=usemrf,trunc=trunc)
    else:
        if usebow:
            scr,pos=f.scanRCFLbow(m,initr=initr,ratio=ratio,small=small,trunc=trunc)
            scr1,pos1=f.scanRCFLbow(m1,initr=initr,ratio=ratio,small=small,trunc=trunc)
        else:
            scr,pos=f.scanRCFL(m,initr=initr,ratio=ratio,small=small,trunc=trunc)
            scr1,pos1=f.scanRCFL(m1,initr=initr,ratio=ratio,small=small,trunc=trunc)
            
    lr=[]
    fscr=[]
    for idl,l in enumerate(scr):
        auxscr=numpy.zeros((l.shape[0],l.shape[1],2),numpy.float32)
        auxscr[:,:,0]=scr[idl]
        auxscr[:,:,1]=scr1[idl]
        fscr.append(numpy.max(auxscr,2))
        lr.append(numpy.argmax(auxscr,2))
        if 0:
            pylab.figure(101)
            pylab.clf()
            print auxscr[:,:,0]
            print auxscr[:,:,1]
            pylab.imshow(lr[-1])
            pylab.show()
            raw_input()
    if deform:
        tr=TreatDefRL(f,fscr,pos,pos1,lr,initr,m["fy"],m["fx"],occl=occl,trunc=trunc)
    else:
        tr=TreatRL(f,fscr,pos,pos1,lr,initr,m["fy"],m["fx"],occl=occl,trunc=trunc)

    numhog=f.getHOG()
    if gtbbox==None:
        if show==True:
            showlabel="Parts"
        else:
            showlabel=False
        if fastBU:#enable TD+BU
            print "Fast BU"
            t1=time.time()
            det=tr.doall(thr=thr,rank=10,refine=True,rawdet=False,cluster=False,show=False,inclusion=inclusion,cl=cl)
            #detR=[];detL=[]
            #for d in det:
            #    if d["rl"]==1:
            #        detR.append(d)
            #    else:
            #        detL.append(d)
            #samplesR=tr.goodsamples(detR,initr=initr,ratio=ratio)
            #samplesL=tr.goodsamples(detL,initr=initr,ratio=ratio)
            #scrR,posR=f.scanRCFLDefBU(m,initr=initr,ratio=ratio,small=small,usemrf=usemrf,mysamples=samplesR)
            #scrL,posL=f.scanRCFLDefBU(m1,initr=initr,ratio=ratio,small=small,usemrf=usemrf,mysamples=samplesL)
            #lr=[];fscr=[]
            #for idl,l in enumerate(scrR):
            #    auxscr=numpy.zeros((l.shape[0],l.shape[1],2),numpy.float32)
            #    auxscr[:,:,0]=scrR[idl]
            #    auxscr[:,:,1]=scrL[idl]
            #    fscr.append(numpy.max(auxscr,2))
            #    lr.append(numpy.argmax(auxscr,2))
            #tr=TreatDefRL(f,fscr,posR,posL,lr,initr,m["fy"],m["fx"],occl=occl)

            samples=tr.goodsamples(det,initr=initr,ratio=ratio)
            scr,pos=f.scanRCFLDefBU(m,initr=initr,ratio=ratio,small=small,usemrf=usemrf,mysamples=samples)
            scr1,pos1=f.scanRCFLDefBU(m1,initr=initr,ratio=ratio,small=small,usemrf=usemrf,mysamples=samples)
            lr=[]
            fscr=[]
            for idl,l in enumerate(scr):
                auxscr=numpy.zeros((l.shape[0],l.shape[1],2),numpy.float32)
                auxscr[:,:,0]=scr[idl]
                auxscr[:,:,1]=scr1[idl]
                fscr.append(numpy.max(auxscr,2))
                lr.append(numpy.argmax(auxscr,2))
                if 0:
                    pylab.figure(101)
                    pylab.clf()
                    print auxscr[:,:,0]
                    print auxscr[:,:,1]
                    pylab.imshow(lr[-1])
                    pylab.show()
                    raw_input()
            tr=TreatDefRL(f,fscr,pos,pos1,lr,initr,m["fy"],m["fx"],occl=occl,trunc=trunc)

            print "Refine Time:",time.time()-t1
            det=tr.doall(thr=thr,rank=100,refine=True,rawdet=False,cluster=nms,show=False,inclusion=inclusion,cl=cl)
        else:
            det=tr.doall(thr=thr,rank=100,refine=True,rawdet=False,cluster=nms,show=False,inclusion=inclusion,cl=cl)
        dettime=time.time()-t
        if show:
            tr.show(det,parts=showlabel,thr=-1.0,maxnum=0)  
        print "Detect: %.3f"%(time.time()-t)
        return tr,det,dettime,numhog
    else:
        best1,worste1=tr.doalltrain(gtbbox,thr=thr,rank=ranktr,show=show,mpos=mpos,numpos=1,posovr=posovr,numneg=numneg,minnegovr=0,minnegincl=minnegincl,cl=cl,useMaxOvr=useMaxOvr)        
        ipos=[];ineg=[]
        if False and show and len(best1)>0:
            import util
            ds=tr.descr(best1,flip=False,usemrf=usemrf,usefather=usefather,k=K)
            pylab.figure(120)
            pylab.clf()
            mm=tr.model(ds[0],0,len(m1["ww"]),31)
            util.drawModel(mm["ww"])
            pylab.show()
            #raw_input()

        print "Detect: %.3f"%(time.time()-t)
        return tr,best1,worste1,ipos,ineg

