import numpy
from database import *
import pyrHOG2
import scipy.cluster.vq as vq
import util
import pylab
import pegasos
import drawHOG

def showHOG(feat,siftsize):
    #util.drawHOG9(feat.reshape((siftsize,siftsize,31)))
    img=drawHOG.drawHOG(feat.reshape((siftsize,siftsize,31)))
    pylab.axis("off")
    pylab.imshow(img,cmap=pylab.cm.gray,interpolation="nearest")

def showHOGflat(feat,siftsize):
    #util.drawHOG9(feat.reshape((siftsize,siftsize,31)))
    img=drawHOG.drawHOG(feat)
    pylab.axis("off")
    pylab.imshow(img,cmap=pylab.cm.gray,interpolation="nearest")

def showBook(book,siftsize):
    pylab.figure()
    pylab.clf()
    num=round(numpy.sqrt(book.shape[0]))+1
    for l in range(book.shape[0]):
        print l
        pylab.subplot(num,num,l+1)
        showHOG(book[l],siftsize)
    pylab.show()

def showImage(sim):
    pylab.figure(20)
    pylab.clf()
    num=round(numpy.sqrt(book.shape[0]))+1
    l=1
    sample=2
    sim1=sim#.T.reshape(sim.shape[2],sim.shape[0],sim.shape[1]).T
    #sim2=sim1.copy().reshape(sim1.shape[0],sim1.shape[1],sim1.shape[2])
    print "y",sim1.shape[0],"x",sim1.shape[1]
    for y in range(sim1.shape[0]/sample):
        for x in range(sim1.shape[1]/sample):
            print y,x
            pylab.subplot(sim1.shape[0]/sample,sim1.shape[1]/sample,l)#y*(sim.shape[1]/sample)+x+1)
            showHOG(sim1[y*sample,x*sample,:].reshape(siftsize,siftsize,31))
            #showHOG(sim2[y*sample,x*sample,:].reshape(siftsize,siftsize,31))
            pylab.show()
            l=l+1
            #raw_input()
    pylab.show()

def showBuild(words,book):
    pylab.figure(21)
    pylab.clf()
    num=round(numpy.sqrt(book.shape[0]))+1
    l=1
    sample=2
    sim1=sim#.T.reshape(sim.shape[2],sim.shape[0],sim.shape[1]).T
    #sim2=sim1.copy().reshape(sim1.shape[0],sim1.shape[1],sim1.shape[2])
    print "y",sim1.shape[0],"x",sim1.shape[1]
    for y in range(sim1.shape[0]/sample):
        for x in range(sim1.shape[1]/sample):
            print y,x
            pylab.subplot(sim1.shape[0]/sample,sim1.shape[1]/sample,l)#y*(sim.shape[1]/sample)+x+1)
            showHOG(book[words[y*sample,x*sample],:].reshape(siftsize,siftsize,31))
            #showHOG(sim1[y*sample,x*sample,:].reshape(siftsize,siftsize,31))
            #showHOG(sim2[y*sample,x*sample,:].reshape(siftsize,siftsize,31))
            pylab.show()
            l=l+1
            #raw_input()
    pylab.show()

def match(feat,book):
    #ff=numpy.tile(feat.T,(book.shape[0],1,1)).T
    #res=numpy.argmax(numpy.sum(ff*book.T,1),1)
    res=numpy.zeros((feat.shape[0],book.shape[0]))
    res1=numpy.zeros((feat.shape[0],book.shape[0]))
    for cl in range(book.shape[0]):
        res[:,cl]=(numpy.sum(feat*book[cl,:],1))
        #res1[:,cl]=numpy.sum((feat-book[cl,:])**2,1)
    #print res.argmax(1)
    #print res1.argmin(1)
    #sdd
    #val=res.max(1)
    #clu=res.argmax(1)
    #clu[val<0]=-1
    return res.argmax(1)

def matchthr(feat,book,thr=0,dist="",bias=False):
    #print dist
    #ff=numpy.tile(feat.T,(book.shape[0],1,1)).T
    #res=numpy.argmax(numpy.sum(ff*book.T,1),1)
    res=numpy.zeros((feat.shape[0],book.shape[0]))
    #res1=numpy.zeros((feat.shape[0],book.shape[0]))
    for cl in range(book.shape[0]):
        #res[:,cl]=(numpy.sum(feat*book[cl,:],1))
        if dist=="eucl":
            #print "EUCL"
            res[:,cl]=(numpy.sum((feat-book[cl,:])**2,1))
        else:
            #print "SCALAR"
            #feat1=(feat.T/(numpy.sqrt(numpy.sum(feat**2,1)).T)).T
            if bias:
                res[:,cl]=(numpy.sum((feat*book[cl,:-1]),1))+bias*book[cl,-1]        
            else:
                res[:,cl]=(numpy.sum((feat*book[cl,:]),1))        
        #res[:,cl]=numpy.sum((feat-book[cl,:])**2,1)
    #print res.argmax(1)
    #print res1.argmin(1)
    #sdd
    #val=res.max(1)
    if dist=="eucl":
        clu=res.argmin(1)
    else:
        clu=res.argmax(1)
    #clu[val<=0]=-1
    #print clu
    #raw_input()
    return clu,res

def matchind(feat,book,thr=False,dist="",bias=False):
    res=numpy.zeros((feat.shape[0],book.shape[0]))
    for cl in range(book.shape[0]):
        if dist=="eucl":
            #print "EUCL"
            res[:,cl]=(numpy.sum((feat-book[cl,:])**2,1))
        else:
            #print "SCALAR"
            #feat1=(feat.T/(numpy.sqrt(numpy.sum(feat**2,1)).T)).T
            if bias:
                res[:,cl]=(numpy.sum((feat*book[cl,:-1]),1))+bias*book[cl,-1]        
            else:
                res[:,cl]=(numpy.sum((feat*book[cl,:]),1))        
    if dist=="eucl":
        clu=res.argmin(0)
    else:
        clu=res.argmax(0)
    if thr!=False:
        for cl in range(book.shape[0]):
            if res[clu[cl],cl]<thr:
                clu[cl]=-1            
    return clu,res

def hogtosift(hog,siftsize,geom=False):
    him=hog#numpy.ascontiguousarray(hog)
    #print hog.flags
    #raw_input()
    himy=him.shape[0]-siftsize+1
    himx=him.shape[1]-siftsize+1
    sift=numpy.zeros((himy,himx,siftsize,siftsize,hog.shape[2]))
    for sy in range(siftsize):
        for sx in range(siftsize):
            sift[:,:,sy,sx]=him[sy:sy+himy,sx:sx+himx]		
    sim=sift.reshape((himy,himx,siftsize**2*hog.shape[2]))	
    #sim=sift.reshape((himx,himy,siftsize**2*hog.shape[2]))
    #showImage(sim)
    feat=sim.T.reshape((sim.shape[2],himy*himx)).T
    if 0:
        pylab.figure()
        showHOGflat(hog,siftsize)
        pylab.figure()
        showBook(feat[:100],siftsize)
        pylab.draw()
        pylab.show()
        raw_input()
    if geom:
        return feat,sim
    return feat

def hogtosift2(hog,siftsize,geom=False):
    him=hog#numpy.ascontiguousarray(hog)
    #print hog.flags
    #raw_input()
    himy=him.shape[0]-siftsize
    himx=him.shape[1]-siftsize
    sift=numpy.zeros((himy,himx,siftsize,siftsize,hog.shape[2]))
    for sy in range(siftsize):
        for sx in range(siftsize):
            sift[:,:,sy,sx]=him[sy:sy+himy,sx:sx+himx]		
    sim=sift.reshape((himy,himx,siftsize**2*hog.shape[2]))	
    #sim=sift.reshape((himx,himy,siftsize**2*hog.shape[2]))
    #showImage(sim)
    feat=sim.T.reshape((sim.shape[2],himy*himx)).T
    if 0:
        pylab.figure()
        showHOGflat(hog,siftsize)
        pylab.figure()
        showBook(feat[:100],siftsize)
        pylab.draw()
        pylab.show()
        raw_input()
    if geom:
        return feat,sim
    return feat,himy,himx

#def hist(trImages,rbook,numim,siftsize,dist=""):
#    hist=numpy.zeros((numim,numcl))
#    afeat=numpy.zeros((numim,numcl,siftsize**2*31))
#    for i in range(min(numim,trImages.getTotal())):
#        print "Image %d/%d"%(i,min(numim,trImages.getTotal()))
#        img=trImages.getImageName(i)
#        feat=pyrHOG2.pyrHOG(img,interv=1)
#        feat=hogtosift(feat.hog[1])
#        #showBook(feat)
#        #raw_input()
#        #him=feat.hog[0]
#        #himy=him.shape[0]-siftsize
#        #himx=him.shape[1]-siftsize
#        #sift=numpy.zeros((himy,himx,siftsize,siftsize,feat.hog[0].shape[2]))
#        #for sy in range(siftsize):
#        #    for sx in range(siftsize):
#        #        sift[:,:,sy,sx]=him[sy:sy+himy,sx:sx+himx]			
#        #sim=sift.reshape((himx,himy,siftsize**2*feat.hog[0].shape[2]))
#        #feat=sim.T.reshape((sim.shape[2],himy*himx)).T
#        if dist=="":#euclidean distance
#            words=vq.vq(feat,rbook)[0]
#        else:#scalr product
#            words,res=matchthr(feat,rbook)
#        for cl in range(numcl):
#            if numpy.sum(words==cl)>0:
#                #afeat[i,cl,:]=feat[numpy.argmax(res[:,cl]*(words==cl)),:]
#                afeat[i,cl,:]=numpy.mean(feat[words==cl,:],0)
#        #showBook(afeat[i,:,:])
#        hist[i,:]=numpy.histogram(words,numcl,(-0.5,numcl-0.5))[0]
#        print numpy.sum(rbook**2,1)
#        print "Hist:",hist[i,:]
#        #raw_input()
#    return hist[:i+1,:],afeat[:i+1,:,:]

if False:

    #compute hog from a sequence of images
    VOCbase="/home/owner/databases/"
    cls="bicycle"
    trImages=VOC07Data(cl="%s_train.txt"%cls,select="all",basepath=VOCbase,usetr=True,usedf=False)
    trPosImages=VOC07Data(cl="%s_train.txt"%cls,select="pos",basepath=VOCbase,usetr=True,usedf=False)
    trNegImages=VOC07Data(cl="%s_train.txt"%cls,select="neg",basepath=VOCbase,usetr=True,usedf=False)
    testPosImages=VOC07Data(select="pos",cl="%s_val.txt"%cls,basepath=VOCbase,usetr=True,usedf=False)
    testNegImages=VOC07Data(select="neg",cl="%s_val.txt"%cls,basepath=VOCbase,usetr=True,usedf=False)
    testImages=VOC07Data(select="pos",cl="%s_test.txt"%cls,basepath=VOCbase,usetr=True,usedf=False)
    #trImages=VOC06Data(cl="%s_trainval.txt"%cls,select="all",basepath=VOCbase,usetr=True,usedf=False)
    #trPosImages=VOC06Data(cl="%s_trainval.txt"%cls,select="pos",basepath=VOCbase,usetr=True,usedf=False)
    #trNegImages=VOC06Data(cl="%s_trainval.txt"%cls,select="neg",basepath=VOCbase,usetr=True,usedf=False)
    #testPosImages=VOC06Data(select="pos",cl="%s_test.txt"%cls,basepath=VOCbase,usetr=True,usedf=False)
    #testNegImages=VOC06Data(select="neg",cl="%s_test.txt"%cls,basepath=VOCbase,usetr=True,usedf=False)
    #testImages=VOC06Data(select="pos",cl="%s_test.txt"%cls,basepath=VOCbase,usetr=True,usedf=False)

    siftsize=4
    numcl=20
    numim=32
    maxsift=10000
    svect=numpy.zeros((2*maxsift,siftsize**2*31))
    pos=0
    lb=1.0

    #compute vocabulary
    recVOC=False
    name="book%d_%d_%s_"%(siftsize,numcl,cls)
    try:
	    rbook=numpy.load('%s.npz'%(name))["arr_0"]
    except:
        print "Computing Vocabulary"
        recVOC=True
    if recVOC:
        for i in range(min(numim,trImages.getTotal())):
            img=trImages.getImageName(i)
            feat=pyrHOG2.pyrHOG(img,interv=1)
            him=feat.hog[1]
            himy=him.shape[0]-siftsize
            himx=him.shape[1]-siftsize
            feat=hogtosift(him)
		    #sift=numpy.zeros((himy,himx,siftsize,siftsize,feat.hog[0].shape[2]))
		    #for sy in range(siftsize):
		    #	for sx in range(siftsize):
		    #		sift[:,:,sy,sx]=him[sy:sy+himy,sx:sx+himx]			
		    #sim=sift.reshape((himx,himy,siftsize**2*feat.hog[0].shape[2]))
            svect[pos:pos+himy*himx]=feat#sim.T.reshape((sim.shape[2],himy*himx)).T
            pos+=himy*himx
            print pos
            if pos>maxsift:
                break
        rbook,di=vq.kmeans(svect[:10000],numcl,3)
        numpy.savez('%s.npz'%(name), rbook)

    print "Vocabulary computed"
    #showBook(rbook)
    raw_input()


    #show segmentation
    name="mytrain%d"%12
    if False:
        w1=util.load("w10.pz")
        b1=0
        #book=rbook
        book=w1.reshape((numcl,siftsize**2*31))
        showBook(book)
        #pylab.figure()
        #pylab.imshow(numpy.arange(numcl).reshape((numcl,1)),vmax=numcl,interpolation="nearest")
        #w1,b1=util.loadSvmLib("%s.svm"%name,dir="./")   
        for i in range(testImages.getTotal()):
            img=testImages.getImageName(i)
            #img="a.png"
            feat=pyrHOG.pyrHOG(img,interv=1)
            him=feat.hog[1]
            himy=him.shape[0]-siftsize
            himx=him.shape[1]-siftsize
            pylab.figure(12)
            pylab.clf()
            pylab.imshow(testImages.getImage(i))
            feat,sim=hogtosift(him,True)
            pylab.figure(13)
            pylab.clf()
            #book=w1[:-numcl].reshape((numcl,siftsize**2*31))
            #words,res=matchthr(feat,book)
            words=vq.vq(feat,book)[0]
            words1=words.reshape(sim.shape[1],sim.shape[0]).T
            scr1=numpy.sum(book[words]**2,1).reshape(sim.shape[1],sim.shape[0]).T
            pylab.imshow(words1,vmax=numcl)#cmap=pylab.get_cmap(pylab.cm.flag_r))
            #showBuild(words1,book)
            pylab.figure(14)
            pylab.clf()
            pylab.imshow(scr1)
            pylab.draw()
            pylab.show()
            print "Done"
            raw_input()

    #compute histograms
    recSVM=False
    name="hist%d_%d_%s_"%(siftsize,numcl,cls)
    import util
    try:
        hpos,hneg,fpos,fneg=util.load("%s.p"%name)
    except:
        recSVM=True
    if recSVM:
        hpos,fpos=hist(trPosImages,rbook,numim,siftsize)
        hneg,fneg=hist(trNegImages,rbook,numim,siftsize)
        util.save("%s.p"%(name),(hpos,hneg,fpos,fneg))

    pylab.figure()
    pylab.clf()
    pylab.plot(hpos.mean(0))
    pylab.plot(hneg.mean(0))
    pylab.show()

    #cumpute SVM
    name="ntrain%d_%d_%s_"%(siftsize,numcl,cls)
    Force=False
    #try:
    #    w,b=util.loadSvmLib("%s.svm"%name,dir="./")
    #except: 
    #    Force=True      
    #    #normal histogram svm training 
    #if Force:
    hpos=(hpos.T/numpy.sum(hpos,1).T).T.astype(numpy.float32)		
    hneg=(hneg.T/numpy.sum(hneg,1).T).T.astype(numpy.float32)	
        #w=numpy.zeros(cumsize[-1])
        #w,r,prloss=pegasos.trainComp(trpos,trneg,testname+"loss.rpt.txt",trposcl,trnegcl,oldw=w,dir="",pc=pc,k=10,numthr=numcore)
    import pegasos
    w=pegasos.train(hpos,hneg,C=0.001,maxtimes=200)   
    #util.trainSvmRawLib(hpos,hneg,"%s.svm"%name,dir="./",pc=0.001)
    #w,b=util.loadSvmLib("%s.svm"%name,dir="./")

    #my svm training
    #name="mytrain"
    #try:
    #    w1,b1=util.loadSvmLib("%s.svm"%name,dir="./")
    #except:       
    #    fpos=fpos.reshape((fpos.shape[0],siftsize**2*31*numcl))
    #    fneg=fpos.reshape((fneg.shape[0],siftsize**2*31*numcl))
    #    hpos=numpy.concatenate((fpos,hpos),1).astype(numpy.float32)		
    #    hneg=numpy.concatenate((fneg,hneg),1).astype(numpy.float32)		
    #    util.trainSvmRawLib(hpos,hneg,"%s.svm"%name,dir="./",pc=1)
    #    w1,b1=util.loadSvmLib("%s.svm"%name,dir="./")
        
    #nbook=w1[:-100].reshape((numcl,siftsize**2*31))
    #nbook=nbook/numpy.sum(abs(nbook)) 

    #classify in validation set
    recVAL=False
    name="histval%d_%d_%s_"%(siftsize,numcl,cls)
    try:
        #pass
        hpos,hneg,fpos,fneg=util.load("%s.p"%name)
    except:
        recVAL=True
    if recVAL:
        hpos,fpos=hist(testPosImages,rbook,numim,siftsize)
        hneg,fneg=hist(testNegImages,rbook,numim,siftsize)
        util.save("%s.p"%name,(hpos,hneg,fpos,fneg))

    hpos=(hpos.T/numpy.sum(hpos,1).T).T.astype(numpy.float32)		
    hneg=(hneg.T/numpy.sum(hneg,1).T).T.astype(numpy.float32)	
    b=0
    p=numpy.sum(numpy.sum(hpos*w,1)-b>0)
    n=numpy.sum(numpy.sum(hneg*w,1)-b>0)

    print "Pos:",numpy.sum(hpos*w,1)-b
    print "Neg:",numpy.sum(hneg*w,1)-b
    print p,n
    raw_input()

    nbook=numpy.random.random((numcl,siftsize**2*31))-0.5
    #nbook=rbook
    nbook=(nbook.T/numpy.sqrt((nbook**2).sum(1)+0.000001).T).T
    pylab.figure(2)
    #showBook(nbook)
    #raw_input()
    #foldneg=numpy.zeros((0,siftsize**2*31*numcl+numcl))
    foldneg=numpy.zeros((0,siftsize**2*31*numcl))
    #foldpos=numpy.zeros((0,siftsize**2*31*numcl+numcl))
    foldpos=numpy.zeros((0,siftsize**2*31*numcl))
    #w1=numpy.zeros(siftsize**2*31*numcl)
    w1=numpy.random.random(siftsize**2*31*numcl)-0.5
    w1=w1/numpy.sqrt(numpy.sum(w1**2))
    count=0
    oldf=0

    for l in range(100):
        #our method retraining
        recVAL=True
        name="myfeat%d_"%l
        try:
            hpos,hneg,fpos,fneg=util.load("%s%d.p"%(name,numcl))
        except:
            recVAL=True
        if recVAL:
            hpos,fpos=hist(trPosImages,nbook,numim,siftsize,"scalar")
            hneg,fneg=hist(trNegImages,nbook,numim,siftsize,"scalar")
            util.save("%s%d.p"%(name,numcl),(hpos,hneg,fpos,fneg))

        #showBook(fpos[0,:,:])
        #raw_input()
        fpos=fpos.reshape((fpos.shape[0],siftsize**2*31*numcl))
        fneg=fneg.reshape((fneg.shape[0],siftsize**2*31*numcl))
        #fpos=(fpos.T/numpy.sum(fpos,1).T).T.astype(numpy.float32)/numcl		
        #fneg=(fneg.T/numpy.sum(fneg,1).T).T.astype(numpy.float32)/numcl
        hpos=(hpos.T/numpy.sum(hpos,1).T).T.astype(numpy.float32)		
        hneg=(hneg.T/numpy.sum(hneg,1).T).T.astype(numpy.float32)
        hpos=fpos.astype(numpy.float32)
        #hpos=numpy.concatenate((fpos,hpos),1).astype(numpy.float32)		
        hneg=fneg.astype(numpy.float32)
        #hneg=numpy.concatenate((fneg,hneg),1).astype(numpy.float32)			

        #hpos=numpy.concatenate((hpos,foldpos),0).astype(numpy.float32)
        #foldpos=hpos.copy()
        #hneg=numpy.concatenate((hneg,foldneg),0).astype(numpy.float32)
        #foldneg=hneg.copy()

        recSVM=True
        name="mytrain%d"%l
        try:
            pass
            #w1,b1=util.loadSvmLib("%s.svm"%name,dir="./")
        except:  
            resSVM=True
        if recSVM:     
            if False:
                util.trainSvmRawLib(hpos,hneg,"%s.svm"%name,dir="./",pc=0.001)
                w1,b1=util.loadSvmLib("%s.svm"%name,dir="./")
                #w1,b1=pegasos.trainsvm(hpos,hneg,lb=0.01,it=10000,k=1,eps=0.0001,b=True)
            else:
                eps=0.00001
                inner=1000
                withb=False
                if withb:
                    w1=numpy.concatenate((w1,[0]))
                    hpos1=numpy.concatenate((hpos,numpy.ones((hpos.shape[0],1))),1)
                    hneg1=numpy.concatenate((hneg,numpy.ones((hneg.shape[0],1))),1)
                else:
                    hpos1=hpos
                    hneg1=hneg
                for c in range(10):
                    print "Iteration:",c*inner
                    #w1=pegasos.pegasosOne(w1,hpos1,hneg1,lb=lb,it=inner,k=1,eps=eps,fact=c)
                    #w1=pegasos.pegasosMax(w1,hpos1,hneg1,lb=lb,it=inner,k=1,eps=eps,fact=c,smax=siftsize**2*31)
                    #w1=pegasos.pegasosMax(w1,hpos1,hneg1,lb=lb,it=inner,k=1,eps=eps,fact=c,smax=siftsize**2*31)
                    w1=pegasos.train(hpos1,hneg1,C=0.001)#,smax=siftsize**2*31)
                    #examples=numpy.concatenate((hpos1,hneg1),0)
                    #labels=numpy.ones(hpos1.shape[0]+hneg1.shape[0])
                    #labels[hpos1.shape[0]:]=-labels[hpos1.shape[0]:]
                    #print "W before",w1
                    #pegasos.ff.fast_pegasos(w1,len(w1),examples,examples.shape[0],labels,lb,inner,c)
                    #print "W after",w1
                    #raw_input()
                    #f=pegasos.objf(w1,hpos1,hneg1,lb)
                    #print f
                    #pylab.figure(10)
                    #pylab.plot([count-1,count],[oldf,f],"bo-")
                    #pylab.show()
                    #pylab.draw()
                    count=count+1
                    #if abs((oldf-f)/f)<eps:
                    #    print "Convergence!"
                    break
                    #oldf=f
                if withb:
                    b1=w1[-1]
                    w1=w1[:-1]
                else:
                    b1=0
        util.save("w10.pz",w1)
        #nbook=w1[:-numcl].reshape((numcl,siftsize**2*31))
        nbook=w1.reshape((numcl,siftsize**2*31))
        #    nbook=nbook/numpy.sum(abs(nbook)) 
    #l2 norm
        nbook=(nbook.T/numpy.sqrt((nbook**2).sum(1)+0.00001).T).T
        
        #for cl in range(numcl):
        #    if numpy.all(nbook[cl,:]==0):
        #        nbook[cl,:]=numpy.random.random((1,siftsize**2*31))-0.5
        #        nbook[cl,:]=nbook[cl,:]/numpy.sqrt(numpy.sum(nbook[cl,:]**2))#/numcl
        pylab.figure(1)
        pylab.clf()
        #pylab.plot(w1)
        pylab.figure(2)
        pylab.clf()
        pylab.plot(w1)
        pylab.draw()
        pylab.show()

        p=numpy.sum(numpy.sum(hpos*w1,1)-b1>0)
        n=numpy.sum(numpy.sum(hneg*w1,1)-b1>0)

        print "Pos:",numpy.sum(hpos*w1,1)-b1
        print "Neg:",numpy.sum(hneg*w1,1)-b1
        
        print p,n
        showBook(nbook)
        #raw_input()

        #our method
        recVAL=True
        name="myval_"
        try:
            hpos,hneg,fpos,fneg=util.load("%s%d.p"%(name,numcl))
        except:
            recVAL=True
        if recVAL:
            hpos,fpos=hist(testPosImages,nbook,numim,siftsize,"scalar")
            hneg,fneg=hist(testNegImages,nbook,numim,siftsize,"scalar")
            util.save("%s%d.p"%(name,numcl),(hpos,hneg,fpos,fneg))

        ac=numpy.sum(hneg,0)+numpy.sum(hpos,0)
        fpos=fpos.reshape((fpos.shape[0],siftsize**2*31*numcl))
        fneg=fneg.reshape((fneg.shape[0],siftsize**2*31*numcl))
        #fpos=(fpos.T/numpy.sum(fpos,1).T).T.astype(numpy.float32)/numcl		
        #fneg=(fneg.T/(numpy.sum(fneg,1)+0.00001).T).T.astype(numpy.float32)/numcl
        hpos=(hpos.T/numpy.sum(hpos,1).T).T.astype(numpy.float32)		
        hneg=(hneg.T/numpy.sum(hneg,1).T).T.astype(numpy.float32)
        hpos=fpos.astype(numpy.float32)
        #hpos=numpy.concatenate((fpos,hpos),1).astype(numpy.float32)		
        hneg=fneg.astype(numpy.float32)
        #hneg=numpy.concatenate((fneg,hneg),1).astype(numpy.float32)			
        t=0.0
        p=numpy.sum(numpy.sum(hpos*w1,1)-b1>t)
        n=numpy.sum(numpy.sum(hneg*w1,1)-b1>t)

        print "TEST:"
        print "Pos:",numpy.mean(numpy.sum(hpos*w1,1)-b1),numpy.sum(hpos*w1,1)-b1
        print "Neg:",numpy.mean(numpy.sum(hneg*w1,1)-b1),numpy.sum(hneg*w1,1)-b1

        print "Active Clusters:",ac
        print p,n
        cmd=raw_input()
        if cmd=="s":
            #show segmentation
            name="mytrain%d"%12
            if True:
                #w1=util.load("w10.pz")
                #b1=0
                #book=rbook
                book=w1.reshape((numcl,siftsize**2*31))
                #showBook(book)
                #pylab.figure()
                #pylab.imshow(numpy.arange(numcl).reshape((numcl,1)),vmax=numcl,interpolation="nearest")
                #w1,b1=util.loadSvmLib("%s.svm"%name,dir="./")   
                for i in range(10):#testImages.getTotal()):
                    img=testImages.getImageName(i)
                    #img="a.png"
                    feat=pyrHOG2.pyrHOG(img,interv=1)
                    him=feat.hog[1]
                    himy=him.shape[0]-siftsize
                    himx=him.shape[1]-siftsize
                    pylab.figure(12)
                    pylab.clf()
                    pylab.imshow(testImages.getImage(i))
                    feat,sim=hogtosift(him,True)
                    pylab.figure(13)
                    pylab.clf()
                    #book=w1[:-numcl].reshape((numcl,siftsize**2*31))
                    #words,res=matchthr(feat,book)
                    words=vq.vq(feat,book)[0]
                    words1=words.reshape(sim.shape[1],sim.shape[0]).T
                    scr1=numpy.sum(book[words]**2,1).reshape(sim.shape[1],sim.shape[0]).T
                    pylab.imshow(words1,vmax=numcl)#cmap=pylab.get_cmap(pylab.cm.flag_r))
                    #showBuild(words1,book)
                    pylab.figure(14)
                    pylab.clf()
                    pylab.imshow(scr1)
                    pylab.draw()
                    pylab.show()
                    print "Done"
                    #raw_input()



