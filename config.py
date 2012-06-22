#default configuration file

class config(object):
    pass
cfg=config()

cfg.dbpath="/home/databases/"
cfg.fy=8#remember small
cfg.fx=3
cfg.lev=3
cfg.interv=10
cfg.ovr=0.45
cfg.sbin=8
cfg.maxpos=2000#120
cfg.maxtest=2000#100
cfg.maxneg=2000#120
cfg.maxexamples=10000
cfg.deform=True
cfg.usemrf=False
cfg.usefather=True
cfg.bottomup=False
cfg.inclusion=False
cfg.initr=1
cfg.ratio=1
cfg.mpos=0.5
cfg.posovr=0.7
cfg.hallucinate=1
cfg.numneginpos=5
cfg.useflipos=True
cfg.useflineg=True
cfg.svmc=0.001#0.001#0.002#0.004
cfg.convPos=0.005
cfg.convNeg=1.05
cfg.show=False
cfg.thr=-2
cfg.multipr=4
cfg.negit=10#10
cfg.posit=8
cfg.perc=0.15
cfg.comment="I shuld get more than 84... hopefully"
cfg.numneg=0#not used but necessary
cfg.testname="./data/11_02_28/inria_rightpedro"
cfg.savefeat=False #save precomputed features 
cfg.loadfeat=True
cfg.useprior=False
cfg.ovr=0.5
cfg.k=1.0
cfg.small=False
cfg.dense=0
cfg.ovrasp=0.3
cfg.minnegincl=0
cfg.sortneg=False
cfg.bestovr=False
cfg.useRL=False
cfg.kmeans=False
cfg.resize=None
cfg.ranktr=500
cfg.occl=False
cfg.small2=False
#cfg.savedir=InriaPosData(basepath=dbpath).getStorageDir() #where to save
#    mydebug=False
#    if mydebug:
#        cfg.multipr=False
#        cfg.maxpos=10
#        cfg.maxneg=10
#        cfg.maxtest=10
#        cfg.maxexamples=1000
 

