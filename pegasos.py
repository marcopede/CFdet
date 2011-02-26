import numpy
import ctypes
from ctypes import c_float,c_int

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


def train(posnfeat,negnfeat,fname,oldw=None,dir="./save/",pc=0.017,path="/home/marcopede/code/c/liblinear-1.7",maxtimes=100,eps=0.001,bias=100):
    """
        The same as trainSVMRaw but it does use files instad of lists:
        it is slower but it needs less memory.
    """
    ff=open(fname,"a")
    posntimes=len(posnfeat)
    negntimes=len(negnfeat)
    ntimes=posntimes+negntimes
    fdim=len(posnfeat[0])+1
    bigm=numpy.zeros((ntimes,fdim),dtype=numpy.float32)
    w=numpy.zeros(fdim,dtype=numpy.float32)
    if oldw!=None:
        w[:-1]=oldw
    for l in range(posntimes):
        bigm[l,:-1]=posnfeat[l]
        bigm[l,-1]=bias
    for l in range(negntimes):
        bigm[posntimes+l,:-1]=negnfeat[l]
        bigm[posntimes+l,-1]=bias
    labels=numpy.concatenate((numpy.ones(posntimes),-numpy.ones(negntimes))).astype(numpy.float32)
    print "Starting Pegasos SVM training"
    ff.write("Starting Pegasos SVM training\n")
    lamd=1/(pc*ntimes)
    obj=0.0
    nobj=lpeg.objective(w,fdim,bigm,ntimes,labels,lamd)
    print "Initial Objective Function:",nobj
    for tt in range(maxtimes):
        lpeg.fast_pegasos(w,fdim,bigm,ntimes,labels,lamd,ntimes*10,tt)
        #lpeg.fast_pegasos(w,fdim,bigm,ntimes,labels,lamd,int(100/lamd),tt)
        nobj=lpeg.objective(w,fdim,bigm,ntimes,labels,lamd)
        print "Objective Function:",nobj
        ff.write("Objective Function:%f\n"%nobj)
        ratio=abs(abs(obj/nobj)-1)
        print "Ratio:",ratio
        ff.write("Ratio:%f\n"%ratio)
        if ratio<eps:
            print "Converging after %d iterations"%tt
            break
        obj=nobj
        #sts.report(fname,"a","Training")
    b=-w[-1]*float(bias)
    #fast_pegasos(ftype *w,int wx,ftype *ex,int exy,ftype *label,ftype lambda,int iter,int part)
    return w[:-1],b

def trainkeep(posnfeat,negnfeat,fname,oldw=None,dir="./save/",pc=0.017,path="/home/marcopede/code/c/liblinear-1.7",mintimes=50,maxtimes=200,eps=0.001,bias=100,lowbound=None,regmul=None,mul=None):
    """
        The same as trainSVMRaw but it does use files instad of lists:
        it is slower but it needs less memory.
    """
    posloss=0
    negloss=0
    reg=0
    ff=open(fname,"a")
    posntimes=len(posnfeat)
    negntimes=len(negnfeat)
    ntimes=posntimes+negntimes
    fdim=len(posnfeat[0])+1
    bigm=numpy.zeros((ntimes,fdim),dtype=numpy.float32)
    w=numpy.zeros(fdim,dtype=numpy.float32)
    bestw=w.copy()
    if oldw!=None:
        w[:-1]=oldw
    for l in range(posntimes):
        bigm[l,:-1]=posnfeat[l]
        bigm[l,-1]=bias
    for l in range(negntimes):
        bigm[posntimes+l,:-1]=negnfeat[l]
        bigm[posntimes+l,-1]=bias
    if lowbound==None:
        lowbound=-100*numpy.ones(fdim,dtype=numpy.float32)
    if regmul==None:
        regmul=numpy.ones(fdim,dtype=numpy.float32)
        #regmul[-1]=0
    if mul==None:
        mul=numpy.ones(fdim,dtype=numpy.float32)
    bigm=bigm*mul
    labels=numpy.concatenate((numpy.ones(posntimes),-numpy.ones(negntimes))).astype(numpy.float32)
    print "Starting Pegasos SVM training"
    ff.write("Starting Pegasos SVM training\n")
    lamd=1/(pc*ntimes)
    obj=0.0
    nobj=lpeg.objective(w,fdim,bigm,ntimes,labels,lamd)
    bestobj=1000
    print "Initial Objective Function:",nobj
    for tt in range(maxtimes):
        #lpeg.fast_pegasos_noproj(w,fdim,bigm,ntimes,labels,lamd,ntimes*10,tt)
        lpeg.fast_pegasos2(w,fdim,bigm,ntimes,labels,regmul,lowbound,lamd,ntimes*10,tt)
        #lpeg.fast_pegasos(w,fdim,bigm,ntimes,labels,lamd,int(100/lamd),tt)
        nobj=lpeg.objective2(w,fdim,bigm,ntimes,labels,lamd,posloss,negloss,reg)
        if nobj<bestobj:
            bestw=w.copy()
            bestobj=nobj
        print "Objective Function:",nobj
        print "Best Obejective:",bestobj
        ff.write("Objective Function:%f\n"%nobj)
        ff.write("Pos Loss:%f Neg Loss:%f\n"%(posloss,negloss))
        ratio=abs(abs(obj/nobj)-1)
        print "Ratio:",ratio
        ff.write("Ratio:%f\n"%ratio)
        if ratio<eps and tt>mintimes:
            print "Converging after %d iterations"%tt
            break
        obj=nobj
        #sts.report(fname,"a","Training")
    w=bestw.copy()
    b=-w[-1]*float(bias)
    w=w*mul
    #fast_pegasos(ftype *w,int wx,ftype *ex,int exy,ftype *label,ftype lambda,int iter,int part)
    return w[:-1],b


def train_new(posnfeat,negnfeat,fname,oldw=None,dir="./save/",pc=0.017,path="/home/marcopede/code/c/liblinear-1.7",mintimes=50,maxtimes=200,eps=0.001,bias=100,lowbound=None,regmul=None,mul=None):
    """
        The same as trainSVMRaw but it does use files instad of lists:
        it is slower but it needs less memory.
    """
    posloss=0
    negloss=0
    reg=0
    ff=open(fname,"a")
    posntimes=len(posnfeat)
    negntimes=len(negnfeat)
    ntimes=posntimes+negntimes
    fdim=len(posnfeat[0])+1
    bigm=numpy.zeros((ntimes,fdim),dtype=numpy.float32)
    w=numpy.zeros(fdim,dtype=numpy.float32)
    if oldw!=None:
        w[:-1]=oldw
    for l in range(posntimes):
        bigm[l,:-1]=posnfeat[l]
        bigm[l,-1]=bias
    for l in range(negntimes):
        bigm[posntimes+l,:-1]=negnfeat[l]
        bigm[posntimes+l,-1]=bias
    if lowbound==None:
        lowbound=-100*numpy.ones(fdim,dtype=numpy.float32)
    if regmul==None:
        regmul=numpy.ones(fdim,dtype=numpy.float32)
        #regmul[-1]=0
    if mul==None:
        mul=numpy.ones(fdim,dtype=numpy.float32)
    bigm=bigm*mul
    labels=numpy.concatenate((numpy.ones(posntimes),-numpy.ones(negntimes))).astype(numpy.float32)
    print "Starting Pegasos SVM training"
    ff.write("Starting Pegasos SVM training\n")
    lamd=1/(pc*ntimes)
    obj=0.0
    nobj=lpeg.objective(w,fdim,bigm,ntimes,labels,lamd)
    print "Initial Objective Function:",nobj
    for tt in range(maxtimes):
        #lpeg.fast_pegasos_noproj(w,fdim,bigm,ntimes,labels,lamd,ntimes*10,tt)
        lpeg.fast_pegasos2(w,fdim,bigm,ntimes,labels,regmul,lowbound,lamd,ntimes*10,tt)
        #lpeg.fast_pegasos(w,fdim,bigm,ntimes,labels,lamd,int(100/lamd),tt)
        nobj=lpeg.objective2(w,fdim,bigm,ntimes,labels,lamd,posloss,negloss,reg)
        print "Objective Function:",nobj
        ff.write("Objective Function:%f\n"%nobj)
        ff.write("Pos Loss:%f Neg Loss:%f\n"%(posloss,negloss))
        ratio=abs(abs(obj/nobj)-1)
        print "Ratio:",ratio
        ff.write("Ratio:%f\n"%ratio)
        if ratio<eps and tt>mintimes:
            print "Converging after %d iterations"%tt
            break
        obj=nobj
        #sts.report(fname,"a","Training")
    b=-w[-1]*float(bias)
    w=w*mul
    #fast_pegasos(ftype *w,int wx,ftype *ex,int exy,ftype *label,ftype lambda,int iter,int part)
    return w[:-1],b

