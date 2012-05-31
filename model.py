
import numpy
#danger: code dupicated in pyrHOG2.py: find a solution


def initmodel(fy,fx,lev,useRL,deform,numbow=6**4,onlybow=False,CRF=False):
    #fy=cfg.fy[c]
    #fx=cfg.fx[c]
    ww=[]
    hww=[]
    voc=[]
    dd=[]    
    for l in range(lev):
        if useRL:
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
            lowf=numpy.ones((fy*2**l,fx*2**l,31)).astype(numpy.float32)
            #lowf=lowf1/(numpy.sum(lowf1**2))
        if l==0:
            lowd=numpy.zeros((1*2**l,1*2**l,4)).astype(numpy.float32)
            #lowd=-numpy.ones((1*2**l,1*2**l,4)).astype(numpy.float32)
        else:
            lowd=-numpy.ones((1*2**l,1*2**l,4)).astype(numpy.float32)
        ww.append(lowf)
        #hww.append(numpy.ones((2**l,2**l,numbow),dtype=numpy.float32))
        if deform:
            #if l!=3:
            hww.append(0.001*numpy.ones((2**l,2**l,numbow)).astype(numpy.float32))
            #else:
            #hww.append(0.001*numpy.zeros((2**l,2**l,numbow),dtype=numpy.float32))
                #hww[-1][0,0]=0.001*numpy.ones((numbow),dtype=numpy.float32)
            #hww.append(0.001*numpy.random.random((2**l,2**l,numbow)).astype(numpy.float32))
            #else:
            #hww.append(0.001*numpy.zeros((2**l,2**l,numbow),dtype=numpy.float32))
        else:
            hww.append(0.001*numpy.ones((numbow),dtype=numpy.float32))
            #hww.append(1.0*numpy.random.random(numbow).astype(numpy.float32))
        #hww.append(numpy.zeros((2**l,2**l,numbow),dtype=numpy.float32))
        #voc.append(numpy.zeros((2**l,2**l,numbow,siftsize**2*9),dtype=numpy.float32))
        #voc[-1][:,:]=rbook
        dd.append(lowd)
        rho=0
    mynorm=0
    for wc in ww:
        mynorm+=numpy.sum(wc**2)
    for idw,wc in enumerate(ww):
        ww[idw]=wc*0.1/numpy.sqrt(mynorm)
        if onlybow:
            ww[idw][:,:,:]=0.0
            dd[idw][:,:,:]=0.0
    if CRF:
        #cost=0.01*numpy.ones((2,fy*2,fx*2),dtype=numpy.float32)
        cost=0.01*numpy.ones((2,fy*2,fx*2),dtype=numpy.float32)
        #cost[0,-1,:]=0
        #cost[1,:,-1]=0
        #import crf3
        #cache_cost=crf3.cost(fy*2,fx*2,(fy*2*2-1)/2,(fx*2*2-1)/2,c=0.001,ch=cost[0],cv=cost[1])
        #cache_cost=numpy.zeros((2,fy*2*fx*2,(fy*2*2-1)/2*(fx*2*2-1)/2,(fy*2*2-1)/2*(fx*2*2-1)/2),dtype=numpy.float32)
        #crf3.fill_cache(cache_cost,fy*2,fx*2,(fy*2*2-1)/2*(fx*2*2-1)/2,(fy*2*2-1)/2,(fx*2*2-1)/2,cost)
        return {"ww":ww,"hist":hww,"rho":rho,"df":dd,"fy":ww[0].shape[0],"fx":ww[0].shape[1],"cost":cost}        
    return {"ww":ww,"hist":hww,"rho":rho,"df":dd,"fy":ww[0].shape[0],"fx":ww[0].shape[1]}

def model2w(model,deform,usemrf,usefather,k=1,lastlev=0,usebow=False,useCRF=False):
    w=numpy.zeros(0,dtype=numpy.float32)
    for l in range(len(model["ww"])-lastlev):
        #print "here"#,item
        w=numpy.concatenate((w,model["ww"][l].flatten()))
        if deform:
            if usefather:
                w=numpy.concatenate((w,model["df"][l][:,:,0].flatten()))        
                w=numpy.concatenate((w,model["df"][l][:,:,1].flatten()))        
            if usemrf:
                w=numpy.concatenate((w,model["df"][l][:,:,2].flatten()))                
                w=numpy.concatenate((w,model["df"][l][:,:,3].flatten()))   
    if usebow:
        for l in range(len(model["hist"])-lastlev):
            w=numpy.concatenate((w,model["hist"][l].flatten()))
    if useCRF:
        w=numpy.concatenate((w,(model["cost"]*k).flatten()))
    return w

def model2wDef(model,k,deform=False,usemrf=True,usefather=True,lastlev=0,useoccl=False,usebow=False):
        """
        convert each detection in a feature descriptor for the SVM
        """      
        d=numpy.array([],dtype=numpy.float32)
        for l in range(len(model["ww"])-lastlev):
            d=numpy.concatenate((d,model["ww"][l].flatten()))       
            if l>0 : #skip deformations level 0
                if usefather:
                    d=numpy.concatenate((d, (model["df"][l][:,:,0].flatten())  ))
                    d=numpy.concatenate((d, (model["df"][l][:,:,1].flatten())  ))
                if usemrf:
                    d=numpy.concatenate((d,model["df"][l][:,:,2].flatten()))
                    d=numpy.concatenate((d,model["df"][l][:,:,3].flatten()))
            if useoccl:
                d=numpy.concatenate((d,model["occl"]))
        if usebow:
#            for l in range(len(model["hist"])-lastlev):
#                auxd=numpy.zeros((2**l,2**l,6**4),dtype=numpy.float32)
#                for px in range(2**l):
#                    for py in range(2**l):
#                        auxd[py,px,:]=model["hist"][l][py:(py+1),px:(px+1)]
#pyrHOG2.hog2bow((item["feat"][l][py*fy:(py+1)*fy,px*fx:(px+1)*fx]).copy())
            #d=numpy.concatenate((d,auxd.flatten()))
            for l in range(len(model["hist"])-lastlev):
                d=numpy.concatenate((d,model["hist"][l].flatten()))
        return d

def w2model(descr,rho,lev,fsz,fy=[],fx=[],bin=5,siftsize=2,deform=False,usemrf=False,usefather=False,k=1,mindef=0.0,useoccl=False,usebow=False,useCRF=False):
        #does not work with occlusions
        """
        build a new model from the weights of the SVM
        """     
        ww=[]  
        p=0
        occl=[0]*lev
#        if fy==[]:
#            fy=self.fy
#        if fx==[]:
#            fx=self.fx
        d=descr
        for l in range(lev):
            dp=(fy*fx)*4**l*fsz
            ww.append((d[p:p+dp].reshape((fy*2**l,fx*2**l,fsz))).astype(numpy.float32))
            p+=dp
            if useoccl:
                occl[l]=d[p]
                p+=1
        hist=[]
        if usebow:
            for l in range(lev):
                hist.append(d[p:p+bin**(siftsize**2)].astype(numpy.float32))
                #hist.append(numpy.zeros(625,dtype=numpy.float32))
                p=p+bin**(siftsize**2)
        m={"ww":ww,"rho":rho,"fy":fy,"fx":fx,"occl":occl,"hist":hist,"voc":hist}
        if useCRF:
            m["cost"]=((d[p:].reshape((2,2*fy,2*fx))/float(k)).clip(mindef,10))
        return m

def w2modelDef(descr,rho,lev,fsz,fy=[],fx=[],bin=5,siftsize=2,mindef=0.001,usemrf=True,usefather=True,useoccl=False,usebow=False): 
    """
    build a new model from the weights of the SVM
    """     
    ww=[]  
    df=[]
    occl=[0]*lev
    p=0
    d=descr
    for l in range(lev):
        dp=(fy*fx)*4**l*fsz
        ww.append((d[p:p+dp].reshape((fy*2**l,fx*2**l,fsz))).astype(numpy.float32))
        p+=dp
        if l>0: #skip level 0
            ddp=4**l
            aux=numpy.zeros((2**l,2**l,4))
            if usefather:
                aux[:,:,0]=d[p:p+ddp].reshape((2**l,2**l))
                p+=ddp
                aux[:,:,1]=d[p:p+ddp].reshape((2**l,2**l))
                p+=ddp
            if usemrf:
                aux[:,:,2]=d[p:p+ddp].reshape((2**l,2**l))
                p+=ddp
                aux[:,:,3]=d[p:p+ddp].reshape((2**l,2**l))
                p+=ddp
            df.append(aux.astype(numpy.float32))
        else:
            df.append(numpy.zeros((2**l,2**l,4),dtype=numpy.float32))
        if useoccl:
            occl[l]=d[p]
            p+=1
    hist=[]
    if usebow:
        for l in range(lev):
            hist.append(d[p:p+(4**l)*bin**(siftsize**2)].astype(numpy.float32).reshape((2**l,2**l,bin**(siftsize**2))))
            #hist.append(numpy.zeros(625,dtype=numpy.float32))
            p=p+(4**l)*bin**(siftsize**2)
    m={"ww":ww,"rho":rho,"fy":fy,"fx":fx,"df":df,"occl":occl,"hist":hist}
    return m

