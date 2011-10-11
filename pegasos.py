import numpy
import ctypes
from ctypes import c_float,c_int,c_void_p,POINTER,pointer
import pylab

ctypes.cdll.LoadLibrary("./libfastpegasos.so")
lpeg= ctypes.CDLL("libfastpegasos.so")
#void fast_pegasos(ftype *w,int wx,ftype *ex,int exy,ftype *label,ftype lambda,int iter,int part)
lpeg.fast_pegasos.argtypes=[
    numpy.ctypeslib.ndpointer(dtype=c_float,ndim=1,flags="C_CONTIGUOUS")#w
    ,c_int #sizew
    ,numpy.ctypeslib.ndpointer(dtype=c_float,ndim=2,flags="C_CONTIGUOUS")#samples
    ,c_int #numsamples
    ,numpy.ctypeslib.ndpointer(dtype=c_float,ndim=1,flags="C_CONTIGUOUS")#labels
    ,c_float #lambda
    ,c_int #iter
    ,c_int #part
    ]

lpeg.fast_pegasos_noproj.argtypes=[
    numpy.ctypeslib.ndpointer(dtype=c_float,ndim=1,flags="C_CONTIGUOUS")#w
    ,c_int #sizew
    ,numpy.ctypeslib.ndpointer(dtype=c_float,ndim=2,flags="C_CONTIGUOUS")#samples
    ,c_int #numsamples
    ,numpy.ctypeslib.ndpointer(dtype=c_float,ndim=1,flags="C_CONTIGUOUS")#labels
    ,c_float #lambda
    ,c_int #iter
    ,c_int #part
    ]

#lpeg.fast_pegasos2.argtypes=[
#    numpy.ctypeslib.ndpointer(dtype=c_float,ndim=1,flags="C_CONTIGUOUS")#w
#    ,c_int #sizew
#    ,numpy.ctypeslib.ndpointer(dtype=c_float,ndim=2,flags="C_CONTIGUOUS")#samples
#    ,c_int #numsamples
#    ,numpy.ctypeslib.ndpointer(dtype=c_float,ndim=1,flags="C_CONTIGUOUS")#labels
#    ,numpy.ctypeslib.ndpointer(dtype=c_float,ndim=1,flags="C_CONTIGUOUS")#regmul
#    ,numpy.ctypeslib.ndpointer(dtype=c_float,ndim=1,flags="C_CONTIGUOUS")#lowbound
#    ,c_float #lambda
#    ,c_int #iter
#    ,c_int #part
#    ]


#ftype objective(ftype *w,int wx,ftype *ex, int exy,ftype *label,ftype lambda)
lpeg.objective.argtypes=[
    numpy.ctypeslib.ndpointer(dtype=c_float,ndim=1,flags="C_CONTIGUOUS")#w
    ,c_int #sizew
    ,numpy.ctypeslib.ndpointer(dtype=c_float,ndim=2,flags="C_CONTIGUOUS")#samples
    ,c_int #numsamples
    ,numpy.ctypeslib.ndpointer(dtype=c_float,ndim=1,flags="C_CONTIGUOUS")#labels
    ,c_float #lambda
    ]
lpeg.objective.restype=ctypes.c_float

##void fast_pegasos_comp(ftype *w,int numcomp,int *compx,int *compy,ftype **ptrsamplescomp,int totsamples,int *label,int *comp,ftype lambda,int iter,int part)
#lpeg.fast_pegasos_comp.argtypes=[
#    numpy.ctypeslib.ndpointer(dtype=c_float,ndim=1,flags="C_CONTIGUOUS")#w
#    ,c_int #numcomp
#    ,POINTER(c_int) #compx
#    ,POINTER(c_int) #compy
#    ,POINTER(c_void_p) #ptrsamples
#    ,numpy.ctypeslib.ndpointer(dtype=c_int,ndim=1,flags="C_CONTIGUOUS") #label
#    ,numpy.ctypeslib.ndpointer(dtype=c_int,ndim=1,flags="C_CONTIGUOUS") #components
#    ,c_float #lambda
#    ,c_int #iter
#    ,c_int #parts
#    ]

#void fast_pegasos_comp(ftype *w,int numcomp,int *compx,int *compy,ftype **ptrsamplescomp,int totsamples,int *label,int *comp,ftype lambda,int iter,int part)
lpeg.fast_pegasos_comp.argtypes=[
    numpy.ctypeslib.ndpointer(dtype=c_float,ndim=1,flags="C_CONTIGUOUS")#w
    ,c_int #num comp
    ,POINTER(c_int) #compx
    ,POINTER(c_int) #compy 
    ,POINTER(c_void_p)#samples
    ,c_int #numsamples
    ,numpy.ctypeslib.ndpointer(dtype=c_int,ndim=1,flags="C_CONTIGUOUS")#labels
    ,numpy.ctypeslib.ndpointer(dtype=c_int,ndim=1,flags="C_CONTIGUOUS")#comp number
    ,c_float #lambda
    ,c_int #iter
    ,c_int #part
    ]

#void fast_pegasos_comp_parall(ftype *w,int numcomp,int *compx,int *compy,ftype **ptrsamplescomp,int totsamples,int *label,int *comp,ftype lambda,int iter,int part,int k,int numthr)
lpeg.fast_pegasos_comp_parall.argtypes=[
    numpy.ctypeslib.ndpointer(dtype=c_float,ndim=1,flags="C_CONTIGUOUS")#w
    ,c_int #numcomp
    ,POINTER(c_int) #compx
    ,POINTER(c_int) #compy
    ,POINTER(c_void_p) #ptrsamples
    ,c_int #numsamples
    ,numpy.ctypeslib.ndpointer(dtype=c_int,ndim=1,flags="C_CONTIGUOUS") #label
    ,numpy.ctypeslib.ndpointer(dtype=c_int,ndim=1,flags="C_CONTIGUOUS") #components
    ,c_float #lambda
    ,c_int #iter
    ,c_int #parts
    ,c_int #k
    ,c_int #numthr
    ]


#ftype objective_comp(ftype *w,int numcomp,int *compx,int *compy,ftype **ptrsamplescomp,int *label,int* comp,ftype lambda,ftype *errpos,ftype *errneg)
#lpeg.objective_comp.argtypes=[
#    numpy.ctypeslib.ndpointer(dtype=c_float,ndim=1,flags="C_CONTIGUOUS")#w
#    ,c_int #numcomp
#    ,POINTER(c_int) #compx
#    ,POINTER(c_int) #compy
#    ,POINTER(c_void_p) #ptrsamples
#    ,numpy.ctypeslib.ndpointer(dtype=c_int,ndim=1,flags="C_CONTIGUOUS") #label
#    ,numpy.ctypeslib.ndpointer(dtype=c_int,ndim=1,flags="C_CONTIGUOUS") #components
#    ,c_float #lambda
#    ,POINTER(c_float) #posloss
#    ,POINTER(c_float) #negloss
#    ,POINTER(c_float) #reg
#    ]
#lpeg.objective_comp.restype=ctypes.c_float

#lpeg.objective2.argtypes=[
#    numpy.ctypeslib.ndpointer(dtype=c_float,ndim=1,flags="C_CONTIGUOUS")#w
#    ,c_int #sizew
#    ,numpy.ctypeslib.ndpointer(dtype=c_float,ndim=2,flags="C_CONTIGUOUS")#samples
#    ,c_int #numsamples
#    ,numpy.ctypeslib.ndpointer(dtype=c_float,ndim=1,flags="C_CONTIGUOUS")#labels
#    ,c_float #lambda
#    ,c_float #pos loss
#    ,c_float #neg loss 
#    ,c_float #reg
#    ]
#lpeg.objective2.restype=ctypes.c_float


#def train(posnfeat,negnfeat,fname,oldw=None,dir="./save/",pc=0.017,path="/home/marcopede/code/c/liblinear-1.7",maxtimes=100,eps=0.001,bias=100):
#    """
#        The same as trainSVMRaw but it does use files instad of lists:
#        it is slower but it needs less memory.
#    """
#    ff=open(fname,"a")
#    posntimes=len(posnfeat)
#    negntimes=len(negnfeat)
#    ntimes=posntimes+negntimes
#    fdim=len(posnfeat[0])+1
#    bigm=numpy.zeros((ntimes,fdim),dtype=numpy.float32)
#    w=numpy.zeros(fdim,dtype=numpy.float32)
#    if oldw!=None:
#        w[:-1]=oldw
#    for l in range(posntimes):
#        bigm[l,:-1]=posnfeat[l]
#        bigm[l,-1]=bias
#    for l in range(negntimes):
#        bigm[posntimes+l,:-1]=negnfeat[l]
#        bigm[posntimes+l,-1]=bias
#    labels=numpy.concatenate((numpy.ones(posntimes),-numpy.ones(negntimes))).astype(numpy.float32)
#    print "Starting Pegasos SVM training"
#    ff.write("Starting Pegasos SVM training\n")
#    lamd=1/(pc*ntimes)
#    obj=0.0
#    nobj=lpeg.objective(w,fdim,bigm,ntimes,labels,lamd)
#    print "Initial Objective Function:",nobj
#    for tt in range(maxtimes):
#        lpeg.fast_pegasos(w,fdim,bigm,ntimes,labels,lamd,ntimes*10,tt)
#        #lpeg.fast_pegasos(w,fdim,bigm,ntimes,labels,lamd,int(100/lamd),tt)
#        nobj=lpeg.objective(w,fdim,bigm,ntimes,labels,lamd)
#        print "Objective Function:",nobj
#        ff.write("Objective Function:%f\n"%nobj)
#        ratio=abs(abs(obj/nobj)-1)
#        print "Ratio:",ratio
#        ff.write("Ratio:%f\n"%ratio)
#        if ratio<eps:
#            print "Converging after %d iterations"%tt
#            break
#        obj=nobj
#        #sts.report(fname,"a","Training")
#    b=-w[-1]*float(bias)
#    #fast_pegasos(ftype *w,int wx,ftype *ex,int exy,ftype *label,ftype lambda,int iter,int part)
#    return w[:-1],b

#def trainkeep(posnfeat,negnfeat,fname,oldw=None,dir="./save/",pc=0.017,path="/home/marcopede/code/c/liblinear-1.7",mintimes=50,maxtimes=200,eps=0.001,bias=100,lowbound=None,regmul=None,mul=None):
#    """
#        The same as trainSVMRaw but it does use files instad of lists:
#        it is slower but it needs less memory.
#    """
#    posloss=0
#    negloss=0
#    reg=0
#    ff=open(fname,"a")
#    posntimes=len(posnfeat)
#    negntimes=len(negnfeat)
#    ntimes=posntimes+negntimes
#    fdim=len(posnfeat[0])+1
#    bigm=numpy.zeros((ntimes,fdim),dtype=numpy.float32)
#    w=numpy.zeros(fdim,dtype=numpy.float32)
#    bestw=w.copy()
#    if oldw!=None:
#        w[:-1]=oldw
#    for l in range(posntimes):
#        bigm[l,:-1]=posnfeat[l]
#        bigm[l,-1]=bias
#    for l in range(negntimes):
#        bigm[posntimes+l,:-1]=negnfeat[l]
#        bigm[posntimes+l,-1]=bias
#    if lowbound==None:
#        lowbound=-100*numpy.ones(fdim,dtype=numpy.float32)
#    if regmul==None:
#        regmul=numpy.ones(fdim,dtype=numpy.float32)
#        #regmul[-1]=0
#    if mul==None:
#        mul=numpy.ones(fdim,dtype=numpy.float32)
#    bigm=bigm*mul
#    labels=numpy.concatenate((numpy.ones(posntimes),-numpy.ones(negntimes))).astype(numpy.float32)
#    print "Starting Pegasos SVM training"
#    ff.write("Starting Pegasos SVM training\n")
#    lamd=1/(pc*ntimes)
#    obj=0.0
#    nobj=lpeg.objective(w,fdim,bigm,ntimes,labels,lamd)
#    bestobj=1000
#    print "Initial Objective Function:",nobj
#    for tt in range(maxtimes):
#        #lpeg.fast_pegasos_noproj(w,fdim,bigm,ntimes,labels,lamd,ntimes*10,tt)
#        lpeg.fast_pegasos2(w,fdim,bigm,ntimes,labels,regmul,lowbound,lamd,ntimes*10,tt)
#        #lpeg.fast_pegasos(w,fdim,bigm,ntimes,labels,lamd,int(100/lamd),tt)
#        nobj=lpeg.objective2(w,fdim,bigm,ntimes,labels,lamd,posloss,negloss,reg)
#        if nobj<bestobj:
#            bestw=w.copy()
#            bestobj=nobj
#        print "Objective Function:",nobj
#        print "Best Obejective:",bestobj
#        ff.write("Objective Function:%f\n"%nobj)
#        ff.write("Pos Loss:%f Neg Loss:%f\n"%(posloss,negloss))
#        ratio=abs(abs(obj/nobj)-1)
#        print "Ratio:",ratio
#        ff.write("Ratio:%f\n"%ratio)
#        if ratio<eps and tt>mintimes:
#            print "Converging after %d iterations"%tt
#            break
#        obj=nobj
#        #sts.report(fname,"a","Training")
#    w=bestw.copy()
#    b=-w[-1]*float(bias)
#    w=w*mul
#    #fast_pegasos(ftype *w,int wx,ftype *ex,int exy,ftype *label,ftype lambda,int iter,int part)
#    return w[:-1],b


#def train_new(posnfeat,negnfeat,fname,oldw=None,dir="./save/",pc=0.017,path="/home/marcopede/code/c/liblinear-1.7",mintimes=50,maxtimes=200,eps=0.001,bias=100,lowbound=None,regmul=None,mul=None):
#    """
#        The same as trainSVMRaw but it does use files instad of lists:
#        it is slower but it needs less memory.
#    """
#    posloss=0
#    negloss=0
#    reg=0
#    ff=open(fname,"a")
#    posntimes=len(posnfeat)
#    negntimes=len(negnfeat)
#    ntimes=posntimes+negntimes
#    fdim=len(posnfeat[0])+1
#    bigm=numpy.zeros((ntimes,fdim),dtype=numpy.float32)
#    w=numpy.zeros(fdim,dtype=numpy.float32)
#    if oldw!=None:
#        w[:-1]=oldw
#    for l in range(posntimes):
#        bigm[l,:-1]=posnfeat[l]
#        bigm[l,-1]=bias
#    for l in range(negntimes):
#        bigm[posntimes+l,:-1]=negnfeat[l]
#        bigm[posntimes+l,-1]=bias
#    if lowbound==None:
#        lowbound=-100*numpy.ones(fdim,dtype=numpy.float32)
#    if regmul==None:
#        regmul=numpy.ones(fdim,dtype=numpy.float32)
#        #regmul[-1]=0
#    if mul==None:
#        mul=numpy.ones(fdim,dtype=numpy.float32)
#    bigm=bigm*mul
#    labels=numpy.concatenate((numpy.ones(posntimes),-numpy.ones(negntimes))).astype(numpy.float32)
#    print "Starting Pegasos SVM training"
#    ff.write("Starting Pegasos SVM training\n")
#    lamd=1/(pc*ntimes)
#    obj=0.0
#    nobj=lpeg.objective(w,fdim,bigm,ntimes,labels,lamd)
#    print "Initial Objective Function:",nobj
#    for tt in range(maxtimes):
#        #lpeg.fast_pegasos_noproj(w,fdim,bigm,ntimes,labels,lamd,ntimes*10,tt)
#        lpeg.fast_pegasos2(w,fdim,bigm,ntimes,labels,regmul,lowbound,lamd,ntimes*10,tt)
#        #lpeg.fast_pegasos(w,fdim,bigm,ntimes,labels,lamd,int(100/lamd),tt)
#        nobj=lpeg.objective2(w,fdim,bigm,ntimes,labels,lamd,posloss,negloss,reg)
#        print "Objective Function:",nobj
#        ff.write("Objective Function:%f\n"%nobj)
#        ff.write("Pos Loss:%f Neg Loss:%f\n"%(posloss,negloss))
#        ratio=abs(abs(obj/nobj)-1)
#        print "Ratio:",ratio
#        ff.write("Ratio:%f\n"%ratio)
#        if ratio<eps and tt>mintimes:
#            print "Converging after %d iterations"%tt
#            break
#        obj=nobj
#        #sts.report(fname,"a","Training")
#    b=-w[-1]*float(bias)
#    w=w*mul
#    #fast_pegasos(ftype *w,int wx,ftype *ex,int exy,ftype *label,ftype lambda,int iter,int part)
#    return w[:-1],b


def objective(trpos,trneg,trposcl,trnegcl,clsize,w,C):
    posloss=0.0
    total=float(len(trpos))
    clsum=numpy.concatenate(([0],numpy.cumsum(clsize)))
    hardpos=0.0
    for idl,l in enumerate(trpos):
        c=int(trposcl[idl])
        pstart=clsum[c]
        pend=pstart+clsize[c]
        scr=numpy.sum(w[pstart:pend-1]*l)+100*w[pend-1]
        posloss+=max(0,1-scr)
        if scr<0:
            hardpos+=1
        #print "hinge",max(0,1-scr),"scr",scr
        #raw_input()
    negloss=0
    hardneg=0
    for idl,l in enumerate(trneg):
        c=int(trnegcl[idl])
        pstart=clsum[c]
        pend=pstart+clsize[c]
        scr=(numpy.sum(w[pstart:pend-1]*l)+100*w[pend-1])
        negloss+=max(0,1+scr)
        if scr>0:
            hardneg+=1
        #print "hinge",max(0,1+scr),"scr",scr
        #raw_input()
    scr=[]
    for idc in range(len(clsize)):
        pstart=clsum[idc]
        pend=pstart+clsize[idc]
        scr.append(numpy.sum(w[pstart:pend]**2))    
    #reg=lamda*max(scr)*0.5
    #print "C in OBJECTIVE",C
    reg=(max(scr))*0.5/total
    posloss=C*posloss/total
    negloss=C*negloss/total
    hardpos=C*float(hardpos)/total
    hardneg=C*float(hardneg)/total
    return posloss,negloss,reg,(posloss+negloss)+reg,hardpos,hardneg

def trainComp(trpos,trneg,fname="",trposcl=None,trnegcl=None,oldw=None,dir="./save/",pc=0.017,path="/home/marcopede/code/c/liblinear-1.7",mintimes=30,maxtimes=200,eps=0.001,bias=100,num_stop_count=5,numthr=1,k=1):
    """
        The same as trainSVMRaw but it does use files instad of lists:
        it is slower but it needs less memory.
    """
    #ff=open(fname,"a")
    if trposcl==None:
        trposcl=numpy.zeros(len(trpos),dtype=numpy.int)
    if trnegcl==None:
        trnegcl=numpy.zeros(len(trneg),dtype=numpy.int)
    numcomp=numpy.array(trposcl).max()+1
    trcomp=[]
    newtrcomp=[]
    trcompcl=[]
    alabel=[]
    label=[]
    for l in range(numcomp):
        trcomp.append([])#*numcomp
        label.append([])
    #trcomp=[trcomp]
    #trcompcl=[]
    #label=[[]]*numcomp
    compx=[0]*numcomp
    compy=[0]*numcomp
    for l in range(numcomp):
        compx[l]=len(trpos[numpy.where(numpy.array(trposcl)==l)[0][0]])+1
    for p,val in enumerate(trposcl):
        trcomp[val].append(numpy.concatenate((trpos[p],[bias])).astype(c_float))
        #trcompcl.append(val)
        label[val].append(1)
        compy[val]+=1
    for p,val in enumerate(trnegcl):
        trcomp[val].append(numpy.concatenate((trneg[p],[bias])).astype(c_float))        
        #trcompcl.append(val)
        label[val].append(-1)
        compy[val]+=1
    ntimes=len(trpos)+len(trneg)
    fdim=numpy.sum(compx)#len(posnfeat[0])+1
    #bigm=numpy.zeros((ntimes,fdim),dtype=numpy.float32)
    w=numpy.zeros(fdim,dtype=numpy.float32)
    if oldw!=None:
        w=oldw.astype(numpy.float32)
        #w[:-1]=oldw
    #for l in range(posntimes):
    #    bigm[l,:-1]=posnfeat[l]
    #    bigm[l,-1]=bias
    #for l in range(negntimes):
    #    bigm[posntimes+l,:-1]=negnfeat[l]
    #    bigm[posntimes+l,-1]=bias
    #labels=numpy.concatenate((numpy.ones(posntimes),-numpy.ones(negntimes))).astype(numpy.float32)
    for l in range(numcomp):
        trcompcl=numpy.concatenate((trcompcl,numpy.ones(compy[l],dtype=numpy.int32)*l))
        alabel=numpy.concatenate((alabel,numpy.array(label[l])))
    trcompcl=trcompcl.astype(numpy.int32)
    alabel=alabel.astype(numpy.int32)
    arrint=(c_int*numcomp)
    arrfloat=(c_void_p*numcomp)
    #trcomp1=[list()]*numcomp
    for l in range(numcomp):#convert to array
        trcomp[l]=numpy.array(trcomp[l])
        newtrcomp.append(trcomp[l].ctypes.data_as(c_void_p))
    print "Clusters size:",compx
    print "Clusters elements:",compy
    print "Starting Pegasos SVM training"
    #ff.write("Starting Pegasos SVM training\n")
    #lamd=1/(pc*ntimes)
    #lamd=1/(pc*len(trpos))
    #lamd=0.5#1/(pc*len(trpos))
    #print "Lambda",lamd
    obj=0.0
    ncomp=c_int(numcomp)
    
    #print "check"
    #pylab.figure(340);pylab.plot(alabel);pylab.show()
    #pylab.figure(350);pylab.plot(trcomp[0].sum(0));pylab.show()
    #pylab.figure(350);pylab.plot(trcomp[1].sum(0));pylab.show()
    #print "X0:",trcomp[0][0,:7],newtrcomp[0]
    #print "X1:",trcomp[1][0,:7],newtrcomp[1]
    #raw_input()
    loss=[]
    posl,negl,reg,nobj,hpos,hneg=objective(trpos,trneg,trposcl,trnegcl,compx,w,pc)
    loss.append([posl,negl,reg,nobj,hpos,hneg])
    for tt in range(maxtimes):
        lpeg.fast_pegasos_comp(w,ncomp,arrint(*compx),arrint(*compy),arrfloat(*newtrcomp),ntimes,alabel,trcompcl,pc,ntimes*10,tt)
        #lpeg.fast_pegasos_comp_parall(w,ncomp,arrint(*compx),arrint(*compy),arrfloat(*newtrcomp),ntimes,alabel,trcompcl,pc,ntimes*10,tt,numthr*4,numthr)
        #lpeg.fast_pegasos_comp(w,ncomp,arrint(*compx),arrint(*compy),arrfloat(*newtrcomp),ntimes,alabel,trcompcl,lamd,ntimes*10*numcomp/k,tt,k,numthr)
        #nobj=lpeg.objective(w,fdim,bigm,ntimes,labels,lamd)
        #nobj=1
        posl,negl,reg,nobj,hpos,hneg=objective(trpos,trneg,trposcl,trnegcl,compx,w,pc)
        loss.append([posl,negl,reg,nobj,hpos,hneg])
        print "Objective Function:",nobj
        print "PosLoss:%.6f NegLoss:%.6f Reg:%.6f"%(posl,negl,reg)
        #ff.write("Objective Function:%f\n"%nobj)
        ratio=abs(abs(obj/nobj)-1)
        print "Ratio:",ratio
        #ff.write("Ratio:%f\n"%ratio)
        if ratio<eps and tt>mintimes:
            if stop_count==0:
                print "Converging after %d iterations"%tt
                break
            else:
                print "Missing ",stop_count," iterations to converge" 
                stop_count-=1
        else:
            stop_count=num_stop_count
        obj=nobj
        #sts.report(fname,"a","Training")
    #b=-w[-1]*float(bias)
    #fast_pegasos(ftype *w,int wx,ftype *ex,int exy,ftype *label,ftype lambda,int iter,int part)
    return w,0,loss


#def trainComp_wrong(trpos,trneg,fname,trposcl=None,trnegcl=None,oldw=None,dir="./save/",pc=0.017,path="/home/marcopede/code/c/liblinear-1.7",maxtimes=100,eps=0.01,bias=100):
#    """
#        The same as trainSVMRaw but it does use files instad of lists:
#        it is slower but it needs less memory.
#    """
#    ff=open(fname,"a")
#    if trposcl==None:
#        trposcl=numpy.zeros(len(trpos),dtype=numpy.int)
#    if trnegcl==None:
#        trnegcl=numpy.zeros(len(trneg),dtype=numpy.int)
#    numcomp=numpy.array(trposcl).max()+1
#    trcomp=[]
#    newtrcomp=[]
#    trcompcl=[]
#    alabel=[]
#    label=[]
#    for l in range(numcomp):
#        trcomp.append([])#*numcomp
#        label.append([])
#    #trcomp=[trcomp]
#    #trcompcl=[]
#    #label=[[]]*numcomp
#    compx=[0]*numcomp
#    compy=[0]*numcomp
#    for l in range(numcomp):
#        compx[l]=len(trpos[numpy.where(numpy.array(trposcl)==l)[0][0]])+1
#    for p,val in enumerate(trposcl):
#        trcomp[val].append(numpy.concatenate((trpos[p],[bias])).astype(c_float))
#        #trcompcl.append(val)
#        label[val].append(1)
#        compy[val]+=1
#    for p,val in enumerate(trnegcl):
#        trcomp[val].append(numpy.concatenate((trneg[p],[bias])).astype(c_float))        
#        #trcompcl.append(val)
#        label[val].append(-1)
#        compy[val]+=1
#    ntimes=len(trpos)+len(trneg)
#    fdim=numpy.sum(compx)#len(posnfeat[0])+1
#    #bigm=numpy.zeros((ntimes,fdim),dtype=numpy.float32)
#    w=numpy.zeros(fdim,dtype=numpy.float32)
#    #if oldw!=None:
#    #    w[:-1]=oldw
#    #for l in range(posntimes):
#    #    bigm[l,:-1]=posnfeat[l]
#    #    bigm[l,-1]=bias
#    #for l in range(negntimes):
#    #    bigm[posntimes+l,:-1]=negnfeat[l]
#    #    bigm[posntimes+l,-1]=bias
#    #labels=numpy.concatenate((numpy.ones(posntimes),-numpy.ones(negntimes))).astype(numpy.float32)
#    for l in range(numcomp):
#        trcompcl=numpy.concatenate((trcompcl,numpy.ones(compy[l],dtype=numpy.int32)*l))
#        alabel=numpy.concatenate((alabel,numpy.array(label[l])))
#    trcompcl=trcompcl.astype(numpy.int32)
#    alabel=alabel.astype(numpy.int32)
#    arrint=(c_int*numcomp)
#    arrfloat=(c_void_p*numcomp)
#    #trcomp1=[list()]*numcomp
#    for l in range(numcomp):#convert to array
#        trcomp[l]=numpy.array(trcomp[l])
#        newtrcomp.append(trcomp[l].ctypes.data_as(c_void_p))
#    print "Clusters size:",compx
#    print "Clusters elements:",compy
#    print "Starting Pegasos SVM training"
#    ff.write("Starting Pegasos SVM training\n")
#    lamd=1/(pc*ntimes)
#    obj=0.0
#    ncomp=c_int(numcomp)
#    
#    #print "check"
#    #pylab.figure(340);pylab.plot(alabel);pylab.show()
#    #pylab.figure(350);pylab.plot(trcomp[0].sum(0));pylab.show()
#    #pylab.figure(350);pylab.plot(trcomp[1].sum(0));pylab.show()
#    #print "X0:",trcomp[0][0,:7],newtrcomp[0]
#    #print "X1:",trcomp[1][0,:7],newtrcomp[1]
#    #raw_input()

#    posloss=c_float(0.0)
#    negloss=c_float(0.0)
#    regul=c_float(0.0)
#    for tt in range(maxtimes):
#        print "ITERATION"
#        #nobj=lpeg.objective_comp(w,ncomp,arrint(*compx),arrint(*compy),arrfloat(*newtrcomp),alabel,trcompcl,lamd,posloss,negloss,reg)
#        lpeg.fast_pegasos_comp(w,ncomp,arrint(*compx),arrint(*compy),arrfloat(*newtrcomp),ntimes,alabel,trcompcl,lamd,ntimes*10*numcomp,tt)
#        #ftype objective_comp(ftype *w,int numcomp,int *compx,int *compy,ftype **ptrsamplescomp,int *label,ftype lambda,int *comp,ftype *errpos,ftype *errneg)
#        nobj=lpeg.objective_comp(w,ncomp,arrint(*compx),arrint(*compy),arrfloat(*newtrcomp),alabel,trcompcl,lamd,posloss,negloss,regul)
#        nobj=1
#        print "Objective Function:",nobj
#        ff.write("Objective Function:%f\n"%nobj)
#        ratio=abs(abs(obj/nobj)-1)
#        print "Ratio:",ratio
#        ff.write("Ratio:%f\n"%ratio)
#        #if ratio<eps:
#        #    print "Converging after %d iterations"%tt
#        #    break
#        obj=nobj
#        #sts.report(fname,"a","Training")
#    #b=-w[-1]*float(bias)
#    #fast_pegasos(ftype *w,int wx,ftype *ex,int exy,ftype *label,ftype lambda,int iter,int part)
#    return w,0


