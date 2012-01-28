
import numpy
#danger: code dupicated in pyrHOG2.py: find a solution

def model2w(model,deform,usemrf,usefather,k,lastlev=0,usebow=False):
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
    return w

def model2wDef(model,k,flip=False,usemrf=True,usefather=True,lastlev=0,useoccl=False,usebow=False):
        """
        convert each detection in a feature descriptor for the SVM
        """      
        d=numpy.array([],dtype=numpy.float32)
        for l in range(len(model["ww"])-lastlev):
            d=numpy.concatenate((d,model["ww"][l].flatten()))       
            if l>0 : #skip deformations level 0
                if usefather:
                    d=numpy.concatenate((d, k*k*(model["df"][l][:,:,0].flatten()**2)  ))
                    d=numpy.concatenate((d, k*k*(model["df"][l][:,:,1].flatten()**2)  ))
                if usemrf:
                    d=numpy.concatenate((d,k*k*model["df"][l][:,:,2].flatten()))
                    d=numpy.concatenate((d,k*k*model["df"][l][:,:,3].flatten()))
            if useoccl:
                d=numpy.concatenate((d,model["occl"]))
        if usebow:
            for l in range(len(model["hist"])-lastlev):
                d=numpy.concatenate((d,model["hist"][l].flatten()))
        return d

def w2model(descr,rho,lev,fsz,fy=[],fx=[],bin=5,siftsize=2,deform=False,usemrf=False,usefather=False,useoccl=False,usebow=False):
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

