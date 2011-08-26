import pyrHOG2
import time
import numpy
import pylab

def hogflip_old(feat,obin=9):
    """    
    returns the orizontally flipped version of the HOG features
    """
    #feature shape
    #[9 not oriented][18 oriented][4 normalization]
    aux=feat[:,::-1,:]
    last=obin+obin*2
    noriented=numpy.concatenate((aux[:,:,0].reshape(aux.shape[0],aux.shape[1],1),aux[:,:,obin-1:0:-1]),2)
    oriented=numpy.concatenate((aux[:,:,obin].reshape(aux.shape[0],aux.shape[1],1),aux[:,:,last-1:obin:-1]),2)
    norm1=aux[:,:,last+2].reshape(aux.shape[0],aux.shape[1],1)
    norm2=aux[:,:,last+3].reshape(aux.shape[0],aux.shape[1],1)
    norm3=aux[:,:,last].reshape(aux.shape[0],aux.shape[1],1)
    norm4=aux[:,:,last+1].reshape(aux.shape[0],aux.shape[1],1)
    aux=numpy.concatenate((noriented,oriented,norm1,norm2,norm3,norm4),2)
    return aux

def hogflip(feat,obin=9):#pedro
    """    
    returns the orizontally flipped version of the HOG features
    """
    #feature shape
    #[9 not oriented][18 oriented][4 normalization] wrong!!!
    #[18 not oriented][9 oriented][4 normalization]
    p=numpy.array([10,9,8,7,6,5,4,3,2,1,18,17,16,15,14,13,12,11,19,27,26,25,24,23,22,21,20,30,31,28,29])-1
    aux=feat[:,::-1,p]
    return numpy.ascontiguousarray(aux)

class TreatRL(pyrHOG2.Treat):
    def __init__(self,f,scr,pos,pos1,lr,sample,fy,fx,occl=False):
        pyrHOG2.Treat.__init__(self,f,scr,pos,sample,fy,fx,occl)
        self.lr=lr
        self.pose=[pos,pos1]

    def refine(self,ldet):
        """
            refine the localization of the object (py,px) based on higher resolutions
        """
        #rdet=pyrHOG2.Treat.refine(self,ldet)
        #for item in rdet:
        #    i=item["i"];cy=item["py"];cx=item["px"];
        #    item["rl"]=self.lr[i][cy,cx]
        #return rdet
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
                #aux=self.pos[i][:,l,cy,cx]#[cy,cx,:,l]
                aux=self.pose[pp][i][:,l,cy,cx]#[cy,cx,:,l]
                el["def"]["dy"][l]=aux[0]
                el["def"]["dx"][l]=aux[1]
                mov=mov+aux*2**(-l)
            el["ry"]+=mov[0]
            el["rx"]+=mov[1]
            rdet.append(el)
        return rdet

    def show(self,*args,**kargs):#ldet,parts=False,colors=["w","r","g","b"],thr=-numpy.inf,maxnum=numpy.inf):
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

    def descr(self,det,flip=False,usemrf=True,usefather=True,k=1.0):   
        ld=[]
        for item in det:
            d=numpy.array([])
            if item["rl"]==1:
                auxflip=not(flip)
            else:
                auxflip=flip
            for l in range(len(item["feat"])):
                if not(auxflip):
                    aux=item["feat"][l]
                    #print "No flip",aux.shape
                else:
                    aux=hogflip(item["feat"][l])
                    #print "Flip",aux.shape
                d=numpy.concatenate((d,aux.flatten()))
                if self.occl:
                    if item["i"]-l*self.interv>=0:
                        d=nump.concatenate((d,[0.0]))
                    else:
                        d=nump.concatenate((d,[1.0]))
            ld.append(d.astype(numpy.float32))
        return ld

class TreatDefRL(pyrHOG2.TreatDef):#not complete
    def __init__(self,f,scr,pos,pos1,lr,sample,fy,fx,occl=False):
        pyrHOG2.TreatDef.__init__(self,f,scr,pos,sample,fy,fx,occl)
        self.lr=lr
        self.pose=[pos,pos1]

    def refine(self,ldet):
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
                #aux=self.pos[i][l][:,:,:,cy,cx]#[cy,cx,:,l]
                aux=self.pose[pp][i][l][:,:,:,cy,cx]#[cy,cx,:,l]
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
        pyrHOG2.TreatDef.show(self,*args,**kargs)
        ldet=args[0]
        if kargs.has_key("thr"):
            thr=kargs["thr"]
        else:
            thr=-numpy.inf
        if kargs.has_key("maxnum"):
            maxnum=kargs["maxnum"]
        else:
            maxnum=-numpy.inf	
        for item in ldet[:maxnum+1]:
            if item["scr"]>thr:
                bbox=item["bbox"]
                if item["rl"]==0:
        			rl="R"
                else:
                    rl="L"
                pylab.text(bbox[3]-5,bbox[2]-5,"%s"%(rl),bbox=dict(facecolor='w', alpha=0.5),fontsize=10)
		

#    def descr(self,det,flip=False,usemrf=True,usefather=True,k=1.0):   
#        ld=[]
#        for item in det:
#            #if item!=[] and not(item.has_key("notfound")):
#            if item["rl"]==1:
#                auxflip=not(flip)
#            else:
#                auxflip=flip
#            d=numpy.array([])
#            for l in range(len(item["feat"])):
#                if not(auxflip):
#                    d=numpy.concatenate((d,item["feat"][l].flatten()))       
#                    if l>0: #skip deformations level 0
#                        if usefather:
#                            d=numpy.concatenate((d, k*k*(item["def"]["dy"][l].flatten()**2)  ))
#                            d=numpy.concatenate((d, k*k*(item["def"]["dx"][l].flatten()**2)  ))
#                        if usemrf:
#                            d=numpy.concatenate((d,k*k*item["def"]["ddy"][l].flatten()))
#                            d=numpy.concatenate((d,k*k*item["def"]["ddx"][l].flatten()))
#                else:
#                    d=numpy.concatenate((d,hogflip(item["feat"][l]).flatten()))        
#                    if l>0: #skip deformations level 0
#                        if usefather:
#                            aux=(k*k*(item["def"]["dy"][l][:,::-1]**2))#.copy()
#                            d=numpy.concatenate((d,aux.flatten()))
#                            aux=(k*k*(item["def"]["dx"][l][:,::-1]**2))#.copy()
#                            d=numpy.concatenate((d,aux.flatten()))
#                        if usemrf:
#                            aux=pyrHOG2.defflip(k*k*item["def"]["ddy"][l])
#                            d=numpy.concatenate((d,aux.flatten()))
#                            aux=pyrHOG2.defflip(k*k*item["def"]["ddx"][l])
#                            d=numpy.concatenate((d,aux.flatten()))
#            ld.append(d.astype(numpy.float32))
#            #else:
#            #    ld.append([])
#        return ld

    def descr(self,det,flip=False,usemrf=True,usefather=True,k=1.0):   
        ld=[]
        for item in det:
            if item["rl"]==1:
                auxflip=not(flip)
            else:
                auxflip=flip
            #ld.append(
            dd=pyrHOG2.TreatDef.descr(self,[item],flip=auxflip,usemrf=usemrf,usefather=usefather,k=k)
            ld.append(dd[0])
        return ld


def flip(m):
    ww1=[]
    df1=[]
    for l in m["ww"]:
        ww1.append(numpy.ascontiguousarray(hogflip(l)))
    m1={"ww":ww1,"rho":m["rho"],"fy":ww1[0].shape[0],"fx":ww1[0].shape[1]}
    if m.has_key("df"):
        for l in m["df"]:
            aux=l.copy()
            aux[:,:,:2]=l[:,::-1,:2]#father flip
            aux[:,:,2]=pyrHOG2.defflip(l[:,:,2])
            aux[:,:,3]=pyrHOG2.defflip(l[:,:,3])
            df1.append(numpy.ascontiguousarray(aux))
        m1["df"]=df1
    return m1    


def detectflip(f,m,gtbbox=None,auxdir=".",hallucinate=1,initr=1,ratio=1,deform=False,bottomup=False,usemrf=False,numneg=0,thr=-2,posovr=0.7,minnegincl=0.5,small=True,show=False,cl=0,mythr=-10,nms=0.5,inclusion=False,usefather=True,mpos=1,useprior=False,K=1.0,occl=False):
    """Detect objec in images
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
            #scr,pos=f.scanRCFLDefThr(m,initr=initr,ratio=ratio,small=small,usemrf=usemrf,mythr=mythr)
            scr,pos=f.scanRCFLDef(m,initr=initr,ratio=ratio,small=small,usemrf=usemrf)
            #scr1,pos1=f.scanRCFLDefThr(m1,initr=initr,ratio=ratio,small=small,usemrf=usemrf,mythr=mythr)
            scr1,pos1=f.scanRCFLDef(m1,initr=initr,ratio=ratio,small=small,usemrf=usemrf)
        #tr=TreatDef(f,scr,pos,initr,m["fy"],m["fx"])
    else:
        scr,pos=f.scanRCFLpr(m,initr=initr,ratio=ratio,small=small,pr=pr)
        scr1,pos1=f.scanRCFLpr(m1,initr=initr,ratio=ratio,small=small,pr=pr)
        #tr=Treat(f,scr,pos,initr,m["fy"],m["fx"])
    lr=[]
    fscr=[]
    #fpos=[]
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
        tr=TreatDefRL(f,fscr,pos,pos1,lr,initr,m["fy"],m["fx"])
    else:
        tr=TreatRL(f,fscr,pos,pos1,lr,initr,m["fy"],m["fx"])

    numhog=f.getHOG()
    #print "Scan:",time.time()-t    
    #dettime=time.time()-t
    #print "Elapsed Time:",dettime
    #print "Number HOG:",numhog
    #print "Getting Detections"
    #best1,worste1,ipos,ineg=tr.doalltrain(gtbbox,thr=-5,rank=10000,show=show,mpos=10,numpos=1,numneg=5,minnegovr=0.01)        
    if gtbbox==None:
        if show==True:
            showlabel="Parts"
        else:
            showlabel=False
        ref=0#enable TD+BU
        if ref:
            t1=time.time()
            det=tr.doall(thr=thr,rank=100,refine=True,rawdet=False,cluster=False,show=False,inclusion=inclusion,cl=cl)
            detR=[];detL=[]
            for d in det:
                if d["rl"]==1:
                    detR.append(d)
                else:
                    detL.append(d)
            samplesR=tr.goodsamples(detR,initr=initr,ratio=ratio)
            samplesL=tr.goodsamples(detL,initr=initr,ratio=ratio)
            #scr,pos=f.scanRCFLDef(m,initr=initr,ratio=ratio,small=small,usemrf=usemrf)
            #scr,pos=f.scanRCFLDefsamples(m,initr=initr,ratio=ratio,small=small,usemrf=usemrf,mysamples=samples)
            scrR,posR=f.scanRCFLDefBU(m,initr=initr,ratio=ratio,small=small,usemrf=usemrf,mysamples=samplesR)
            scrL,posL=f.scanRCFLDefBU(m1,initr=initr,ratio=ratio,small=small,usemrf=usemrf,mysamples=samplesL)
            #scr,pos=f.scanRCFLDef(m,initr=initr,ratio=ratio,small=small,usemrf=usemrf)
            lr=[];fscr=[]
            for idl,l in enumerate(scrR):
                auxscr=numpy.zeros((l.shape[0],l.shape[1],2),numpy.float32)
                auxscr[:,:,0]=scrR[idl]
                auxscr[:,:,1]=scrL[idl]
                fscr.append(numpy.max(auxscr,2))
                lr.append(numpy.argmax(auxscr,2))
            print "Refine Time:",time.time()-t1
            #tr=TreatDefRL(f,scr,pos,initr,m["fy"],m["fx"])
            tr=TreatDefRL(f,fscr,posR,posL,lr,initr,m["fy"],m["fx"],occl=occl)
            det=tr.doall(thr=thr,rank=100,refine=True,rawdet=False,cluster=nms,show=False,inclusion=inclusion,cl=cl)
        #print "Elapsed Time:",dettime
        #print "Number HOG:",numhog
        else:
            det=tr.doall(thr=thr,rank=100,refine=True,rawdet=False,cluster=nms,show=False,inclusion=inclusion,cl=cl)
        dettime=time.time()-t
#        det=tr.doall(thr=thr,rank=100,refine=True,rawdet=False,cluster=nms,show=False,inclusion=inclusion,cl=cl)#remember to take away inclusion
#        #pylab.gca().set_ylim(0,img.shape[0])
#        #pylab.gca().set_xlim(0,img.shape[1])
#        #pylab.gca().set_ylim(pylab.gca().get_ylim()[::-1])
#        dettime=time.time()-t
#        #print "Elapsed Time:",dettime
#        #print "Number HOG:",numhog
        if show:
            tr.show(det,parts=showlabel,thr=-1.0,maxnum=0)  
        print "Detect:",time.time()-t             
        return tr,det,dettime,numhog
    else:
        best1,worste1=tr.doalltrain(gtbbox,thr=thr,rank=1000,show=show,mpos=mpos,numpos=1,posovr=posovr,numneg=numneg,minnegovr=0,minnegincl=minnegincl,cl=cl)        
        if 0:#True:#remember to use it in INRIA
            if deform:
                #print "deform"
                ipos=tr.descr(best1,flip=False,usemrf=usemrf,usefather=usefather,k=K)
                #iposflip=tr.descr(best1,flip=True,usemrf=usemrf,usefather=usefather)
                #ipos=ipos+iposflip
                ineg=tr.descr(worste1,flip=False,usemrf=usemrf,usefather=usefather,k=K)
                #inegflip=tr.descr(worste1,flip=True,usemrf=usemrf,usefather=usefather)
                #ineg=ineg+inegflip
            else:
                ipos=tr.descr(best1,flip=False)
                #iposflip=tr.descr(best1,flip=True)
                #ipos=ipos+iposflip
                ineg=tr.descr(worste1,flip=False)
                #inegflip=tr.descr(worste1,flip=True)
                #ineg=ineg+inegflip
        else:
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

        print "Detect:",time.time()-t    
        return tr,best1,worste1,ipos,ineg

