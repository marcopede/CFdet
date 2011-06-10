import matplotlib
matplotlib.use("Agg") #if run out of ipython do not show any graph
#from procedures import *
from database import *
from multiprocessing import Pool
import util
import pyrHOG2
import VOCpr
import time
import trainINRIA
import itertools

class config(object):
    pass

#cfg.fy=7
#cfg.fx=3
#cfg.lev=3
#cfg.interv=10
#cfg.ovr=0.5
#cfg.sbin=8
#cfg.maxpos=1000#120
#cfg.maxneg=1000#120
#deform=True
#usemrf=True
#initr=1
#ratio=2
#cfg.hallucinate=1
#show=True
#it=6
#sx=0.05 #reduce x dimension of
#testname="./data/test28_10/simpletest"
#testname="./data/test28_10/withoutneg"
#testname="./data/test28_10/hall1"
#testname="./data/test29_10/defnoFatherCache"
#testname="./data/test30_10/highres"
#testname="./data/test30_10/4partsfull"
#testname="./data/test1_8/testfull1000"
#testname="./data/test1_8/fullnodef";it=7
#testname="./data/test2_8/nopneginpos"
#testname="./data/test3_8/Full";it=6
#testname="./data/test11_10/test";it=7
#testname="./data/test3_11/CVPR11";it=5
#testname="./data/test3_11/CVPR11_fullneg";it=7
#testname="./data/test3_11/CVPR11_usefahter";it=9# 93!!!
#testname="./data/test3_11/CVPR11_usefahter";it=6
#testname="./data/INRIA/testTDnoMRFright";it=9
#testname="./data/CVPR/bottomup";it=9
#testname="./data/11_02_26/inria_both";it=10
#testname="./data/11_03_28/inria1_1aspect_last_sort";it=6
testname="./data/INRIA/inria_bothfull";it=6
import sys
if len(sys.argv)>1:
    it=int(sys.argv[1])
#testname="./data/INRIA/testTDnoMRF";it=6
cfg=util.load(testname+".cfg")
cfg.maxtest=1000#100
cfg.numneg=1000
cfg.ovr=0.5
cfg.thr=-2
cfg.k=1.0
#cfg.deform=True
#cfg.bottomup=True
cfg.loadfeat=False
cfg.savefeat=False
#cfg.usemrf=True#False#True#da togliere
#cfg.ratio=1#datogliere
cfg.multipr=8
cfg.inclusion=False
#cfg.nms=0.5
cfg.posovr=0
cfg.dbpath="/home/marcopede/databases/"
#cfg.auxdir="/state/partition1/marcopede/INRIA/"#InriaPosData(basepath="/home/databases/").getStorageDir()
cfg.auxdir=InriaPosData(basepath=cfg.dbpath).getStorageDir()
cfg.show=True
cfg.mythr=-10
cfg.useprior=False
cfg.small=1
cfg.dense=5
pyrHOG2.DENSE=0
#cfg.usemrf=False
#cfg.maxpostest=1000

#cfg.initr=0
#cfg.ratio=0
#cfg.hallucinate=
#cfg.thr=-2

#tsImages=getRecord(InriaTestFullData(basepath=cfg.dbpath),cfg.maxtest)
tsImages=getRecord(InriaTestData(basepath=cfg.dbpath),cfg.maxtest)

m=util.load("%s%d.model"%(testname,it))

if 0:
    print "Show model"
    pylab.figure(100)
    pylab.clf()
    util.drawModel(m["ww"])
    pylab.figure(101)
    pylab.clf()
    util.drawDeform(m["df"])
    pylab.draw()
    pylab.show()

if cfg.multipr==1:
    numcore=None
else:
    numcore=cfg.multipr

initime=time.time()
print "Test"
detlist=[]
dettime=[]
numhog=0
print "---Test Images----"
arg=[[i,tsImages[i]["name"],None,m,cfg] for i in range(len(tsImages))]
mypool = Pool(numcore)
if not(cfg.multipr):
    itr=itertools.imap(trainINRIA.detectWrap,arg)        
else:
    itr=mypool.imap(trainINRIA.detectWrap,arg)
for i,im in enumerate(itr):
    print "---- Image %d----"%i
    print "Detections:", len(im[1]) 
    dettime.append(im[2])
    numhog+=im[3]
    #raw_input()   
    for l in im[1]:
        detlist.append([tsImages[i]["name"].split("/")[-1].split(".")[0],l["scr"],l["bbox"][1],l["bbox"][0],l["bbox"][3],l["bbox"][2]])
del itr
mypool.close()
tp,fp,scr,tot=VOCpr.VOCprRecord(tsImages,detlist,show=False,ovr=0.5)#solved a problem that was giving around 1-2 point less
pylab.figure(16)
pylab.clf()
fppi,miss,ap=VOCpr.drawMissRatePerImage(tp,fp,tot,len(tsImages))
pylab.draw()
pylab.show()
pylab.figure(15)
pylab.clf()
rc,pr,ap=VOCpr.drawPrfast(tp,fp,tot)
pylab.draw()
pylab.show()
#pylab.savefig("%s_ap%d_test.png"%(testname,it))
#pylab.savefig("%s_miss%d_test.png"%(testname,it))
tinit=((time.time()-initime))#/3600.0)
print "Average Detection Time:",numpy.mean(dettime)
print "Number of Computed HOGS",numhog
print "AP(it=",it,")=",ap
print "Test Time: %.3fs"%tinit
#rpres.report(testname+".rpt.txt","a","Results")



