
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

def w2model(descr,rho,lev,fsz,fy=[],fx=[],bin=5,siftsize=2,usemrf=False,usefather=False,useoccl=False,usebow=False):
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
