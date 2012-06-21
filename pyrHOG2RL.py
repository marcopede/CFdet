import pyrHOG2
import time
import numpy
import pylab
import crf3


class TreatRL(pyrHOG2.Treat):
    def __init__(self,f,scr,pos,pos1,lr,sample,fy,fx,occl=False,trunc=0,small2=False):
        pyrHOG2.Treat.__init__(self,f,scr,pos,sample,fy,fx,occl=occl,trunc=trunc,small2=small2)
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

class TreatCRFRL(pyrHOG2.TreatCRF):
    def __init__(self,f,scr,pos,pos1,lr,sample,fy,fx,m1,m2,pscr,pscr1,ranktr,occl=False,trunc=0):
    #def __init__(self,f,scr,pos,pos1,lr,sample,fy,fx,occl=False,trunc=0):
        pyrHOG2.TreatCRF.__init__(self,f,scr,pos,sample,fy,fx,m1,pscr,ranktr,occl=occl,trunc=trunc)
        self.lr=lr
        self.pose=[pos,pos1]
        self.pscr=[pscr,pscr1]
        self.model=[m1,m2]
        

    def refine(self,ldet):
        """
            refine the localization of the object based on the position of the parts
        """
        ### we are assuming the good model based on rigid also for left right
        ### maybe is not the best choice
        pad=self.pad
        print "Refine CRF",len(ldet)
        #fmodel=flip(self.model)
        rdet=[]
        fy=self.fy
        fx=self.fx
        for idi,item in enumerate(ldet):
            i=item["i"];cy=item["py"];cx=item["px"];pp=self.lr[i][cy,cx]
            el=item.copy()
            el["ny"]=el["ry"]
            el["nx"]=el["rx"]
            el["rl"]=pp
            mov=numpy.zeros(2)
            #el["def"]={"dy":numpy.zeros(self.pos[i].shape[1]),"dx":numpy.zeros(self.pos[i].shape[1])}
            el["def"]={"dy":numpy.zeros(self.pose[pp][i].shape[1]),"dx":numpy.zeros(self.pose[pp][i].shape[1])}
            my=0;mx=0
            for l in range(self.pos[i].shape[1]):
                #aux=self.pos[i][:,l,cy,cx]#[cy,cx,:,l]
                aux=self.pose[pp][i][:,l,cy,cx]
                el["def"]["dy"][l]=aux[0]
                el["def"]["dx"][l]=aux[1]
                my=2*my+el["def"]["dy"][l]
                mx=2*mx+el["def"]["dx"][l]
                mov=mov+aux*2**(-l)
            el["ry"]+=mov[0]
            el["rx"]+=mov[1]
            #el["pscr"]=self.pscr[i][cy,cx]
            el["pscr"]=self.pscr[pp][i][cy,cx]
            rdet.append(el)
            l=len(self.model[pp]["ww"])-1
            if i+self.f.starti-(l)*self.interv>=0:
                feat=pyrHOG2.getfeat(self.f.hog[i+self.f.starti-(l)*self.interv],el["ny"]*2**l+my-1-pad,el["ny"]*2**l+my+fy*2**l-1+pad,el["nx"]*2**l+mx-1-pad,el["nx"]*2**l+mx+fx*2**l-1+pad,self.trunc).astype(numpy.float32)
                m=self.model[pp]["ww"][-1]
                cost=self.model[pp]["cost"]
                #m1=self.model[1]["ww"][-1]
                #cost1=self.model[1]["cost"]
                #nscr1,ndef1=crf3.match(m1,feat,cost1,pad=pad,show=False,feat=False)
                check=0
                if check:
                    nscr,ndef,dfeat,edge=crf3.match(m,feat,cost,pad=pad,show=False)
                    el["efeat"]=feat
                    el["dfeat"]=dfeat
                    el["edge"]=edge
                else:
                    nscr,ndef=crf3.match(m,feat,cost,pad=pad,show=False,feat=False)
#                if nscr1>nscr:
#                    nscr=nscr1
#                    ndef=ndef1
#                    el["rl"]=1
#                else:
#                    el["rl"]=0
#                el["fullpscr"]=[self.pscr[0][i][cy,cx],self.pscr[1][i][cy,cx]]
#                el["pscr"]=self.pscr[el["rl"]][i][cy,cx]
                el["CRF"]=ndef
                #el["edge"]=edge
                el["oldscr"]=item["scr"]
                el["scr"]=nscr+sum(el["pscr"][:-1])-self.model[pp]["rho"]
            else:
                print "small error!"
                raw_input()
        #rerank
        rdet=self.rank(rdet,maxnum=self.ranktr)
        #thresholding
        idel=0 
        for idel,el in enumerate(rdet):
            if el["scr"]<self.thr:
                break    
        rdet=rdet[:idel]
        return rdet


    def show(self,*args,**kargs):#ldet,parts=False,colors=["w","r","g","b"],thr=-numpy.inf,maxnum=numpy.inf):
        """
        show the detections in an image
        """        
        pyrHOG2.TreatCRF.show(self,*args,**kargs)
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
            #print item["CRF"]
            if item["rl"]==1:
                #print "FLIP!!!!!"
                auxflip=not(flip)
            else:
                auxflip=flip
            dd=pyrHOG2.TreatCRF.descr(self,[item],flip=auxflip,usemrf=usemrf,usefather=usefather,k=k,usebow=usebow)
            ld.append(dd[0])
        return ld

#should find a way to make the flip automatic
#otherwise when adding something always problems
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
    #for CRF
    if m.has_key("cost"):
        m1["cost"]=pyrHOG2.crfflip(m["cost"])
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


def detectflip(f,m,gtbbox=None,auxdir=".",hallucinate=1,initr=1,ratio=1,deform=False,bottomup=False,usemrf=False,numneg=0,thr=-2,posovr=0.7,minnegincl=0.5,small=True,show=False,cl=0,mythr=-10,nms=0.5,inclusion=False,usefather=True,mpos=1,useprior=False,K=1.0,occl=False,trunc=0,useMaxOvr=False,ranktr=1000,fastBU=False,usebow=False,CRF=False,small2=False):
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
            if CRF:
                scr,pscr,pos=f.scanRCFL(m,initr=initr,ratio=ratio,small=small,trunc=trunc,partScr=True)
                scr1,pscr1,pos1=f.scanRCFL(m1,initr=initr,ratio=ratio,small=small,trunc=trunc,partScr=True)
            else:
                scr,pos=f.scanRCFL(m,initr=initr,ratio=ratio,small=small,trunc=trunc,partScr=False)
                scr1,pos1=f.scanRCFL(m1,initr=initr,ratio=ratio,small=small,trunc=trunc,partScr=False)
        if small2:
            for l in range(f.interv):
                scr[l]=scr[l]+m["small2"][0]*pyrHOG2.SMALL    
                scr[l+f.interv]=scr[l+f.interv]+m["small2"][1]*pyrHOG2.SMALL          
                scr1[l]=scr1[l]+m["small2"][0]*pyrHOG2.SMALL    
                scr1[l+f.interv]=scr1[l+f.interv]+m["small2"][1]*pyrHOG2.SMALL          
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
        if CRF:
            tr=TreatCRFRL(f,fscr,pos,pos1,lr,initr,m["fy"],m["fx"],m,m1,pscr,pscr1,ranktr,occl=occl,trunc=trunc)
        else:
            tr=TreatRL(f,fscr,pos,pos1,lr,initr,m["fy"],m["fx"],occl=occl,trunc=trunc,small2=small2)

    if gtbbox==None:
        if show==True:
            showlabel="Parts"
        else:
            showlabel=False
        if fastBU:#enable TD+BU
            print "Fast BU"
            t1=time.time()
            det=tr.doall(thr=thr,rank=100,refine=True,rawdet=False,cluster=False,show=False,inclusion=inclusion,cl=cl)
            detR=[];detL=[]
            greedy=False
            if greedy:
                for d in det:
                    if d["rl"]==1:
                        detR.append(d)
                    else:
                        detL.append(d)
                samplesR=tr.goodsamples(detL,initr=initr,ratio=ratio)
                samplesL=tr.goodsamples(detR,initr=initr,ratio=ratio)
                scr,pos=f.scanRCFLDefBU(m,initr=initr,ratio=ratio,small=small,usemrf=usemrf,mysamples=samplesR)
                scr1,pos1=f.scanRCFLDefBU(m1,initr=initr,ratio=ratio,small=small,usemrf=usemrf,mysamples=samplesL)
            else:
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
        numhog=f.getHOG()
        return tr,det,dettime,numhog
    else:#training
        show="Parts"
        best1,worste1=tr.doalltrain(gtbbox,thr=thr,rank=ranktr,show=show,mpos=mpos,numpos=1,posovr=posovr,numneg=numneg,minnegovr=0,minnegincl=minnegincl,cl=cl,useMaxOvr=useMaxOvr)        
        ipos=[];ineg=[]
        if 0 and show and len(best1)>0:
            import util
            ds=tr.descr(best1,flip=False,usemrf=usemrf,usefather=usefather,k=K)
            pylab.figure(120)
            pylab.clf()
            mm=tr.model(ds[0],0,len(m1["ww"]),31)
            util.drawModel(mm["ww"])
            pylab.show()
            raw_input()

        print "Detect: %.3f"%(time.time()-t)
        return tr,best1,worste1,ipos,ineg

