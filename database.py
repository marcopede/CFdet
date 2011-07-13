import numpy
import pylab
import scipy.misc.pilutil as pil
import string
import pickle

def myimread(imgname):
    img=None
    if imgname.split(".")[-1]=="png":
        img=pylab.imread(imgname)
    else:
        img=pil.imread(imgname)        
    if img.ndim<3:
        aux=numpy.zeros((img.shape[0],img.shape[1],3))
        aux[:,:,0]=img
        aux[:,:,1]=img
        aux[:,:,2]=img
        img=aux
    return img


def getbboxINRIA(filename):
    """
    get the ground truth bbox from the INRIA database at filename
    """
    fd=open(filename,"r")
    lines=fd.readlines()
    rect=[]
    for idx,item in enumerate(lines):
        p=item.find("Bounding box")
        if p!=-1:
            p=item.find("PASperson")
            if p!=-1:
                p=item.find(":")
                item=item[p:]
                #print item[p:]
                p=item.find("(")
                pXmin=int(item[p+1:].split(" ")[0][:-1])
                pYmin=int(item[p+1:].split(" ")[1][:-1])
                p=item[p:].find("-")
                item=item[p:]
                p=item.find("(")
                pXmax=int(item[p+1:].split(" ")[0][:-1])
                pYmax=int(item[p+1:].split(" ")[1][:-2])
                rect.append((pYmin,pXmin,pYmax,pXmax,0,0))
    return rect

def getbboxVOC06(filename,cl="person",usetr=False,usedf=False):
    """
    get the ground truth bbox from the PASCAL VOC 2006 database at filename
    """
    fd=open(filename,"r")
    lines=fd.readlines()
    rect=[]
    cl="PAS"+cl
    for idx,item in enumerate(lines):
        p=item.find("Bounding box")#look for the bounding box
        if p!=-1:
            p=item.find(cl)#check if it is a person
            if p!=-1:
                p=item.find("Difficult")#check that it is not truncated
                if p==-1 or usedf:
                    p=item.find("Trunc")#check that it is not truncated
                    if p==-1 or usetr:
                        p=item.find(":")
                        item=item[p:]
                        #print item[p:]
                        p=item.find("(")
                        pXmin=int(item[p+1:].split(" ")[0][:-1])
                        pYmin=int(item[p+1:].split(" ")[1][:-1])
                        p=item[p:].find("-")
                        item=item[p:]
                        p=item.find("(")
                        pXmax=int(item[p+1:].split(" ")[0][:-1])
                        pYmax=int(item[p+1:].split(" ")[1][:-3])
                        rect.append((pYmin,pXmin,pYmax,pXmax,0,0))
    return rect

import xml.dom.minidom
from xml.dom.minidom import Node

def getbboxVOC07(filename,cl="person",usetr=False,usedf=False):
    """
    get the ground truth bbox from the PASCAL VOC 2007 database at filename
    """
    rect=[]
    doc = xml.dom.minidom.parse(filename)
    for node in doc.getElementsByTagName("object"):
        #print node
        tr=0
        df=0
        if node.getElementsByTagName("name")[0].childNodes[0].data==cl:
            pose=node.getElementsByTagName("pose")[0].childNodes[0].data#last
            if node.getElementsByTagName("difficult")[0].childNodes[0].data=="0" or usedf:
                if node.getElementsByTagName("truncated")[0].childNodes[0].data=="0" or usetr:
                    if node.getElementsByTagName("difficult")[0].childNodes[0].data=="1":
                        df=1
                    if node.getElementsByTagName("truncated")[0].childNodes[0].data=="1":
                        tr=1
                    l=node.getElementsByTagName("bndbox")
                    #print l
                    for el in l:
                        #print el.parentNode.nodeName
                        if el.parentNode.nodeName=="object":
                            xmin=int(el.getElementsByTagName("xmin")[0].childNodes[0].data)
                            ymin=int(el.getElementsByTagName("ymin")[0].childNodes[0].data)
                            xmax=int(el.getElementsByTagName("xmax")[0].childNodes[0].data)
                            ymax=int(el.getElementsByTagName("ymax")[0].childNodes[0].data)
                            #rect.append((ymin,xmin,ymax,xmax,tr,df))
                            rect.append((ymin,xmin,ymax,xmax,tr,df,pose))#last
    return rect

class imageData:
    """
    interface call to handle a database
    """
    def __init__():
        print "Not implemented"
        
    def getDBname():
        return "Not implemented"
        
    def getImage(i):
        """
        gives the ith image from the database
        """
        print "Not implemented"
    
    def getImageName(i): 
        """
        gives the ith image name from the database
        """
        print "Not implemented"
        
    def getBBox(self,i):
        """
        retrun a list of ground truth bboxs from the ith image
        """
        #print "Not implemented"
        return []
        
    def getTotal():
        """
         return the total number of images in the db
        """
        print "Not implemented"
    
def getRecord(data,total=-1,pos=True):
    """return all the gt data in a record"""
    if total==-1:
        total=data.getTotal()
    else:
        total=min(data.getTotal(),total)
    arrPos=numpy.zeros(total,dtype=[("id",numpy.int32),("name",object),("bbox",list)])
    for i in range(total):
        arrPos[i]["id"]=i
        arrPos[i]["name"]=data.getImageName(i)
        arrPos[i]["bbox"]=data.getBBox(i)
    return arrPos


class InriaPosData(imageData):
    """
    INRIA database for positive examples
    """
    def __init__(self,basepath="/home/databases/",
                        trainfile="INRIAPerson/Train/pos.lst",
                        imagepath="INRIAPerson/Train/pos/",
                        local="INRIAPerson/",
                        annpath="INRIAPerson/Train/annotations/"):
        self.basepath=basepath        
        self.trainfile=basepath+trainfile
        self.imagepath=basepath+imagepath
        self.annpath=basepath+annpath
        self.local=basepath+local
        fd=open(self.trainfile,"r")
        self.trlines=fd.readlines()
        fd.close()
        
    def getStorageDir(self):
        return self.local

    def getDBname(self):
        return "INRIA POS"

    def getImageByName(self,name):
        return myimread(name)
        
    def getImage(self,i):
        item=self.trlines[i]
        return myimread((self.imagepath+item.split("/")[-1])[:-1])
    
    def getImageName(self,i):
        item=self.trlines[i]
        return (self.imagepath+item.split("/")[-1])[:-1]
    
    def getBBox(self,i,cl="",usetr="",usedf=""):
        item=self.trlines[i]
        filename=self.annpath+item.split("/")[-1].split(".")[0]+".txt"
        return getbboxINRIA(filename)
    
    def getTotal(self):
        return len(self.trlines)

class InriaParts(imageData):
    """
    INRIA database for positive examples
    """
    def __init__(self,numparts,part,select="pos",basepath="/home/databases/",
                        trainfile="INRIAPerson/Train/pos.lst",
                        imagepath="INRIAPerson/Train/pos/",
                        local="INRIAPerson/",
                        annpath="INRIAPerson/Train/annotations/"):
        self.basepath=basepath
        self.select=select
        if select=="pos":
            self.trainfile=basepath+trainfile
            self.imagepath=basepath+imagepath
        else:
            self.trainfile=basepath+"INRIAPerson/Train/neg.lst"
            self.imagepath=basepath+"INRIAPerson/Train/neg/"
        self.annpath=basepath+annpath
        self.local=basepath+local
        fd=open(self.trainfile,"r")
        self.trlines=fd.readlines()
        self.part=part
        self.numparts=numparts
        fd.close()
        
    def getStorageDir(self):
        return self.local

    def getDBname(self):
        return "INRIA POS"

    def getImageByName(self,name):
        return myimread(name)
        
    def getImage(self,i):
        item=self.trlines[i*self.numparts+self.part]
        return myimread((self.imagepath+item.split("/")[-1])[:-1])
    
    def getImageName(self,i):
        item=self.trlines[i*self.numparts+self.part]
        return (self.imagepath+item.split("/")[-1])[:-1]
    
    def getBBox(self,i,cl="",usetr="",usedf=""):
        if self.select=="neg":
            return []
        item=self.trlines[i*self.numparts+self.part]
        filename=self.annpath+item.split("/")[-1].split(".")[0]+".txt"
        return getbboxINRIA(filename)
    
    def getTotal(self):
        return len(self.trlines[self.part::self.numparts])

class InriaNegData(imageData):
    """
        INRIA database for negative examples (no bbox method)
    """
    def __init__(self,basepath="/home/databases/",
                        trainfile="INRIAPerson/Train/neg.lst",
                        imagepath="INRIAPerson/Train/neg/",
                        local="INRIAPerson/",
                        ):
        self.basepath=basepath
        self.trainfile=basepath+trainfile
        self.imagepath=basepath+imagepath
        self.local=basepath+local
        fd=open(self.trainfile,"r")
        self.trlines=fd.readlines()
        fd.close()
        
    def getDBname():
        return "INRIA NEG"

    def getStorageDir(self):
        return self.local#basepath+"INRIAPerson/"
        
    def getImage(self,i):
        item=self.trlines[i]
        return myimread((self.imagepath+item.split("/")[-1])[:-1])
    
    def getImageName(self,i):
        item=self.trlines[i]
        return (self.imagepath+item.split("/")[-1])[:-1]
    
    def getTotal(self):
        return len(self.trlines)
    
class InriaTestData(imageData):#not done yet
    """
    INRIA database for positive examples
    """
    def __init__(self,basepath="/home/databases/",
                        trainfile="INRIAPerson/Test/pos.lst",
                        imagepath="INRIAPerson/Test/pos/",
                        local="INRIAPerson/",
                        annpath="INRIAPerson/Test/annotations/"):
        self.basepath=basepath
        self.trainfile=basepath+trainfile
        self.imagepath=basepath+imagepath
        self.annpath=basepath+annpath
        self.local=basepath+local
        fd=open(self.trainfile,"r")
        self.trlines=fd.readlines()
        fd.close()
        
    def getDBname():
        return "INRIA POS"

    def getStorageDir(self):
        return self.local
        
    def getImage(self,i):
        item=self.trlines[i]
        return myimread((self.imagepath+item.split("/")[-1])[:-1])
    
    def getImageName(self,i):
        item=self.trlines[i]
        return (self.imagepath+item.split("/")[-1])[:-1]
    
    def getBBox(self,i):
        item=self.trlines[i]
        filename=self.annpath+item.split("/")[-1].split(".")[0]+".txt"
        return getbboxINRIA(filename)
    
    def getTotal(self):
        return len(self.trlines)

class InriaTestFullData(imageData):
    """
    INRIA database for positive examples
    """
    def __init__(self,basepath="/home/databases/",
                        trainfile="INRIAPerson/Test/pos.lst",
                        imagepath="INRIAPerson/Test/pos/",
                        imagepath2="INRIAPerson/Test/neg/",
                        local="INRIAPerson/",
                        annpath="INRIAPerson/Test/annotations/"):
        self.basepath=basepath
        self.trainfile=basepath+trainfile
        self.imagepath=basepath+imagepath
        self.imagepath2=basepath+imagepath2
        self.annpath=basepath+annpath
        self.local=basepath+local
        fd=open(self.trainfile,"r")
        self.trlines=fd.readlines()
        self.numpos=len(self.trlines)
        fd=open(basepath+"INRIAPerson/Test/neg.lst","r")
        self.trlines=self.trlines+fd.readlines()
        fd.close()
        
    def getDBname():
        return "INRIA POS"

    def getStorageDir(self):
        return self.local
        
    def getImage(self,i):
        item=self.trlines[i]
        impath=self.imagepath
        if i>=self.numpos:
            impath=self.imagepath2    
        return myimread((impath+item.split("/")[-1])[:-1])
    
    def getImageName(self,i):
        item=self.trlines[i]
        impath=self.imagepath
        if i>=self.numpos:
            impath=self.imagepath2 
        return (impath+item.split("/")[-1])[:-1]

    def getImageByName2(self,name):
        #item=self.trlines[i]
        impath=self.imagepath
        #if i>=self.numpos:
        #    impath=self.imagepath2 
        try:
            img=myimread(self.imagepath+name+".png")
        except:
            try:
                img=myimread(self.imagepath2+name+".png")
            except:
                img=myimread(self.imagepath2+name+".jpg")
        return img
    
    def getBBox(self,i):
        if i>=self.numpos:
            return []
        item=self.trlines[i]
        filename=self.annpath+item.split("/")[-1].split(".")[0]+".txt"
        return getbboxINRIA(filename)
    
    def getTotal(self):
        return len(self.trlines)

class InriaTestFullParts(imageData):
    """
    INRIA database for positive examples
    """
    def __init__(self,numparts,part,basepath="/home/databases/",
                        trainfile="INRIAPerson/Test/pos.lst",
                        imagepath="INRIAPerson/Test/pos/",
                        imagepath2="INRIAPerson/Test/neg/",
                        local="INRIAPerson/",
                        annpath="INRIAPerson/Test/annotations/"):
        self.basepath=basepath
        self.trainfile=basepath+trainfile
        self.imagepath=basepath+imagepath
        self.imagepath2=basepath+imagepath2
        self.annpath=basepath+annpath
        self.local=basepath+local
        fd=open(self.trainfile,"r")
        self.trlines=fd.readlines()
        self.numpos=len(self.trlines)
        self.numparts=numparts
        self.part=part
        fd=open(basepath+"INRIAPerson/Test/neg.lst","r")
        self.trlines=self.trlines+fd.readlines()
        fd.close()
        
    def getDBname():
        return "INRIA POS"

    def getStorageDir(self):
        return self.local
        
    def getImage(self,i):
        item=self.trlines[i*self.numparts+self.part]
        impath=self.imagepath
        if i*self.numparts+self.part>=self.numpos:
            impath=self.imagepath2    
        return myimread((impath+item.split("/")[-1])[:-1])
    
    def getImageName(self,i):
        item=self.trlines[i*self.numparts+self.part]
        impath=self.imagepath
        if i*self.numparts+self.part>=self.numpos:
            impath=self.imagepath2 
        return (impath+item.split("/")[-1])[:-1]
    
    def getBBox(self,i):
        if i*self.numparts+self.part>=self.numpos:
            return []
        item=self.trlines[i*self.numparts+self.part]
        filename=self.annpath+item.split("/")[-1].split(".")[0]+".txt"
        return getbboxINRIA(filename)
    
    def getTotal(self):
        return len(self.trlines[self.part::self.numparts])

    
class VOC06Data(imageData):
    """
    VOC06 instance (you can choose positive or negative images with the option select)
    """
    def __init__(self,select="all",cl="person_train.txt",
                        basepath="meadi/DADES-2/",
                        trainfile="VOC2006/VOCdevkit/VOC2006/ImageSets/",
                        imagepath="VOC2006/VOCdevkit/VOC2006/PNGImages/",
                        annpath="VOC2006/VOCdevkit/VOC2006/Annotations/",
                        local="VOC2006/VOCdevkit/local/VOC2006/",
                        usetr=False,usedf=False,precompute=True):
        self.cl=cl
        self.usetr=usetr
        self.usedf=usedf
        self.local=basepath+local
        self.trainfile=basepath+trainfile+cl
        self.imagepath=basepath+imagepath
        self.annpath=basepath+annpath
        self.prec=precompute
        fd=open(self.trainfile,"r")
        self.trlines=fd.readlines()
        fd.close()
        if select=="all":#All images
            self.str=""
        if select=="pos":#Positives images
            self.str="1\n"
        if select=="neg":#Negatives images
            self.str="-1\n"
        self.selines=self.__selected()
        if self.prec:
            self.selbbox=self.__precompute()
        #sdf
    
    def __selected(self):
        lst=[]
        for id,it in enumerate(self.trlines):
            if self.str=="" or it.split(" ")[-1]==self.str:
                lst.append(it)
        return lst
    
    def __precompute(self):
        lst=[]
        tot=len(self.selines)
        cl=self.cl.split("_")[0]
        for id,it in enumerate(self.selines):
            print id,"/",tot
            filename=self.annpath+it.split(" ")[0]+".txt"
            #print filename
            #print getbboxVOC06(filename,cl,self.usetr,self.usedf)
            lst.append(getbboxVOC06(filename,cl,self.usetr,self.usedf))
        #raw_input()
        return lst

    def getDBname(self):
        return "VOC06"
        
    def getImage(self,i):
        item=self.selines[i]
        return myimread((self.imagepath+item.split(" ")[0])+".png")
    
    def getImageByName2(self,name):
        return myimread(self.imagepath+name+".png")

    def getImageByName(self,name):
        return myimread(name)
    
    def getImageName(self,i):
        item=self.selines[i]
        return (self.imagepath+item.split(" ")[0]+".png")

    def getImageRaw(self,i):
        item=self.selines[i]
        return im.open((self.imagepath+item.split(" ")[0])+".png")#pil.imread((self.imagepath+item.split(" ")[0])+".jpg")    
    
    def getStorageDir(self):
        return self.local
    
    def getBBox(self,i,cl=None,usetr=None,usedf=None):
        if usetr==None:
            usetr=self.usetr
        if usedf==None:
            usedf=self.usedf
        if cl==None:#use the right class
            cl=self.cl.split("_")[0]
        bb=[]
        if self.prec:
            bb=self.selbbox[i][:]
            #print self.selbbox
        else:
            item=self.selines[i]
            filename=self.annpath+item.split(" ")[0]+".txt"
            bb=getbboxVOC06(filename,cl,usetr,usedf)
        return bb

    def getBBoxByName(self,name,cl=None,usetr=None,usedf=None):
        if usetr==None:
            usetr=self.usetr
        if usedf==None:
            usedf=self.usedf
        if cl==None:#use the right class
            cl=self.cl.split("_")[0]
        filename=self.annpath+name+".txt"
        return getbboxVOC06(filename,cl,usetr,usedf)
            
    def getTotal(self):
        return len(self.selines)
    
import Image as im

#VOCbase="/share/pascal2007/"

class VOC07Data(VOC06Data):
    """
    VOC07 instance (you can choose positive or negative images with the option select)
    """
    def __init__(self,select="all",cl="person_train.txt",
                basepath="media/DADES-2/",
                trainfile="VOC2007/VOCdevkit/VOC2007/ImageSets/Main/",
                imagepath="VOC2007/VOCdevkit/VOC2007/JPEGImages/",
                annpath="VOC2007/VOCdevkit/VOC2007/Annotations/",
                local="VOC2007/VOCdevkit/local/VOC2007/",
                usetr=False,usedf=False):
        self.cl=cl
        self.usetr=usetr
        self.usedf=usedf
        self.local=basepath+local
        self.trainfile=basepath+trainfile+cl
        self.imagepath=basepath+imagepath
        self.annpath=basepath+annpath
        fd=open(self.trainfile,"r")
        self.trlines=fd.readlines()
        fd.close()
        if select=="all":#All images
            self.str=""
        if select=="pos":#Positives images
            self.str="1\n"
        if select=="neg":#Negatives images
            self.str="-1\n"
        self.selines=self.__selected()
        
    def __selected(self):
        lst=[]
        for id,it in enumerate(self.trlines):
            if self.str=="" or it.split(" ")[-1]==self.str:
                lst.append(it)
        return lst

    def getDBname(self):
        return "VOC07"
    
    def getStorageDir(self):
        return self.local#"/media/DADES-2/VOC2007/VOCdevkit/local/VOC2007/"
        
    def getImage(self,i):
        item=self.selines[i]
        return myimread((self.imagepath+item.split(" ")[0])+".jpg")
    
    def getImageRaw(self,i):
        item=self.selines[i]
        return im.open((self.imagepath+item.split(" ")[0])+".jpg")#pil.imread((self.imagepath+item.split(" ")[0])+".jpg")    
    
    def getImageByName(self,name):
        return myimread(name)
    
    def getImageByName2(self,name):
        return myimread(self.imagepath+name+".jpg")

    def getImageName(self,i):
        item=self.selines[i]
        return (self.imagepath+item.split(" ")[0]+".jpg")
    
    def getTotal(self):
        return len(self.selines)
    
    def getBBox(self,i,cl=None,usetr=None,usedf=None):
        if usetr==None:
            usetr=self.usetr
        if usedf==None:
            usedf=self.usedf
        if cl==None:#use the right class
            cl=self.cl.split("_")[0]
        item=self.selines[i]
        filename=self.annpath+item.split(" ")[0]+".xml"
        return getbboxVOC07(filename,cl,usetr,usedf)

class VOC07Parts(VOC06Data):
    """
    VOC07 instance (you can choose positive or negative images with the option select)
    """
    def __init__(self,numparts,select="all",cl="person_train.txt",
                basepath="media/DADES-2/",
                trainfile="VOC2007/VOCdevkit/VOC2007/ImageSets/Main/",
                imagepath="VOC2007/VOCdevkit/VOC2007/JPEGImages/",
                annpath="VOC2007/VOCdevkit/VOC2007/Annotations/",
                local="VOC2007/VOCdevkit/local/VOC2007/",
                usetr=False,usedf=False):
        self.cl=cl
        self.usetr=usetr
        self.usedf=usedf
        self.local=basepath+local
        self.trainfile=basepath+trainfile+cl
        self.imagepath=basepath+imagepath
        self.annpath=basepath+annpath
        fd=open(self.trainfile,"r")
        self.trlines=fd.readlines()
        fd.close()
        if select=="all":#All images
            self.str=""
        if select=="pos":#Positives images
            self.str="1\n"
        if select=="neg":#Negatives images
            self.str="-1\n"
        self.selines=self.__selected()
        self.listparts=[]
        dl=len(self.selines)/numparts
        for l in range(numparts):
            self.listparts.append(self.selines[l*dl:(l+1)*dl])
        self.listparts[-1]=self.selines[l*dl:]        
        self.part=0

    def __selected(self):
        lst=[]
        for id,it in enumerate(self.trlines):
            if self.str=="" or it.split(" ")[-1]==self.str:
                lst.append(it)
        return lst
        
    def setPart(self,n):
        self.part=n

    def getParts(self):
        return len(self.listparts)

    def getDBname(self):
        return "VOC07"
    
    def getStorageDir(self):
        return self.local#"/media/DADES-2/VOC2007/VOCdevkit/local/VOC2007/"
        
    def getImage(self,i,part=-1):
        #item=self.selines[i]
        if part==-1:
            part=self.part
        item=self.listparts[part][i]
        return myimread((self.imagepath+item.split(" ")[0])+".jpg")
    
    def getImageRaw(self,i,part=-1):
        #item=self.selines[i]
        if part==-1:
            part=self.part
        item=self.listparts[part][i]
        return im.open((self.imagepath+item.split(" ")[0])+".jpg")#pil.imread((self.imagepath+item.split(" ")[0])+".jpg")    
    
    def getImageByName(self,name):
        return myimread(name)
    
    def getImageName(self,i,part=-1):
        #item=self.selines[i]
        if part==-1:
            part=self.part
        item=self.listparts[part][i]
        return (self.imagepath+item.split(" ")[0]+".jpg")
    
    def getTotal(self,part=-1):
        #return len(self.selines)
        if part==-1:
            part=self.part
        return len(self.listparts[part])
    
    def getBBox(self,i,part=-1,cl=None,usetr=None,usedf=None):
        if usetr==None:
            usetr=self.usetr
        if usedf==None:
            usedf=self.usedf
        if cl==None:#use the right class
            cl=self.cl.split("_")[0]
        #item=self.selines[i]
        if part==-1:
            part=self.part
        item=self.listparts[part][i]
        filename=self.annpath+item.split(" ")[0]+".xml"
        return getbboxVOC07(filename,cl,usetr,usedf)

class VOC07Parts2(VOC06Data):
    """
    VOC07 instance (you can choose positive or negative images with the option select)
    """
    def __init__(self,numparts,part,select="all",cl="person_train.txt",
                basepath="media/DADES-2/",
                trainfile="VOC2007/VOCdevkit/VOC2007/ImageSets/Main/",
                imagepath="VOC2007/VOCdevkit/VOC2007/JPEGImages/",
                annpath="VOC2007/VOCdevkit/VOC2007/Annotations/",
                local="VOC2007/VOCdevkit/local/VOC2007/",
                usetr=False,usedf=False):
        self.cl=cl
        self.usetr=usetr
        self.usedf=usedf
        self.local=basepath+local
        self.trainfile=basepath+trainfile+cl
        self.imagepath=basepath+imagepath
        self.annpath=basepath+annpath
        fd=open(self.trainfile,"r")
        self.trlines=fd.readlines()
        fd.close()
        if select=="all":#All images
            self.str=""
        if select=="pos":#Positives images
            self.str="1\n"
        if select=="neg":#Negatives images
            self.str="-1\n"
        self.selines=self.__selected()
        self.listparts=[]
        dl=len(self.selines)/numparts
        l=part
        self.listparts=self.selines[l*dl:(l+1)*dl]
        if l==numparts-1:
            self.listparts=self.selines[l*dl:]        
        self.part=part

    def __selected(self):
        lst=[]
        for id,it in enumerate(self.trlines):
            if self.str=="" or it.split(" ")[-1]==self.str:
                lst.append(it)
        return lst
        
    def getDBname(self):
        return "VOC07"
    
    def getStorageDir(self):
        return self.local#"/media/DADES-2/VOC2007/VOCdevkit/local/VOC2007/"
        
    def getImage(self,i):
        #item=self.selines[i]
        item=self.listparts[i]
        return myimread((self.imagepath+item.split(" ")[0])+".jpg")
    
    def getImageRaw(self,i):
        #item=self.selines[i]
        item=self.listparts[i]
        return im.open((self.imagepath+item.split(" ")[0])+".jpg")#pil.imread((self.imagepath+item.split(" ")[0])+".jpg")    
    
    def getImageByName(self,name):
        return myimread(name)
    
    def getImageName(self,i):
        #item=self.selines[i]
        item=self.listparts[i]
        return (self.imagepath+item.split(" ")[0]+".jpg")
    
    def getTotal(self):
        #return len(self.selines)
        return len(self.listparts)
    
    def getBBox(self,i,part=-1,cl=None,usetr=None,usedf=None):
        if usetr==None:
            usetr=self.usetr
        if usedf==None:
            usedf=self.usedf
        if cl==None:#use the right class
            cl=self.cl.split("_")[0]
        #item=self.selines[i]
        item=self.listparts[i]
        filename=self.annpath+item.split(" ")[0]+".xml"
        return getbboxVOC07(filename,cl,usetr,usedf)

class VOC07Parts3(VOC06Data):
    """
    VOC07 instance (you can choose positive or negative images with the option select)
    """
    def __init__(self,numparts,part,select="all",cl="person_train.txt",
                basepath="media/DADES-2/",
                trainfile="%s/VOCdevkit/%s/ImageSets/",
                imagepath="%s/VOCdevkit/%s/",
                annpath="%s/VOCdevkit/%s/Annotations/",
                local="%s/VOCdevkit/local/%s/",
                usetr=False,usedf=False,db="VOC2007"):
        self.cl=cl
        self.usetr=usetr
        self.usedf=usedf
        self.local=basepath+(local%(db,db))
        if db=="VOC2006":
            self.trainfile=basepath+(trainfile%(db,db))+cl
            self.imagepath=basepath+(imagepath%(db,db))+"PNGImages/"
            self.ext=".png"        
        if db=="VOC2007":
            self.trainfile=basepath+(trainfile%(db,db))+"Main/"+cl
            self.imagepath=basepath+(imagepath%(db,db))+"JPEGImages/"        
            self.ext=".jpg"        
        self.annpath=basepath+(annpath%(db,db))
        fd=open(self.trainfile,"r")
        self.trlines=fd.readlines()
        fd.close()
        if select=="all":#All images
            self.str=""
        if select=="pos":#Positives images
            self.str="1\n"
        if select=="neg":#Negatives images
            self.str="-1\n"
        self.selines=self.__selected()
        self.listparts=[]
        l=part
        self.listparts=self.selines[l::numparts]
        self.part=part
        self.db=db
        
    def __selected(self):
        lst=[]
        for id,it in enumerate(self.trlines):
            if self.str=="" or it.split(" ")[-1]==self.str:
                lst.append(it)
        return lst

    def getDBname(self):
        return "VOC07"
    
    def getStorageDir(self):
        return self.local#"/media/DADES-2/VOC2007/VOCdevkit/local/VOC2007/"
        
    def getImage(self,i):
        #item=self.selines[i]
        item=self.listparts[i]
        return myimread((self.imagepath+item.split(" ")[0])+self.ext)
    
    def getImageRaw(self,i):
        #item=self.selines[i]
        item=self.listparts[i]
        return im.open((self.imagepath+item.split(" ")[0])+self.ext)#pil.imread((self.imagepath+item.split(" ")[0])+".jpg")    
    
    def getImageByName(self,name):
        return myimread(name)
    
    def getImageName(self,i):
        #item=self.selines[i]
        item=self.listparts[i]
        return (self.imagepath+item.split(" ")[0]+self.ext)
    
    def getTotal(self):
        #return len(self.selines)
        return len(self.listparts)
    
    def getBBox(self,i,part=-1,cl=None,usetr=None,usedf=None):
        if usetr==None:
            usetr=self.usetr
        if usedf==None:
            usedf=self.usedf
        if cl==None:#use the right class
            cl=self.cl.split("_")[0]
        #item=self.selines[i]
        item=self.listparts[i]
        if self.db=="VOC2007":
            filename=self.annpath+item.split(" ")[0]+".xml"
            bbx=getbboxVOC07(filename,cl,usetr,usedf)
        if self.db=="VOC2006":
            filename=self.annpath+item.split(" ")[0]+".txt"
            bbx=getbboxVOC06(filename,cl,usetr,usedf)
        return bbx


class VOC07Joinold(VOC06Data):
    """
    VOC07 instance (you can choose positive or negative images with the option select)
    """
    def __init__(self,numparts,part,select="all",cl="person_train.txt",
                basepath="media/DADES-2/",
                trainfile="VOCdevkit/VOC2007/ImageSets/Main/",
                imagepath="VOCdevkit/VOC2007/JPEGImages/",
                annpath="VOCdevkit/VOC2007/Annotations/",
                local="VOCdevkit/local/VOC2007/",
                usetr=False,usedf=False):
        self.cl=cl
        self.usetr=usetr
        self.usedf=usedf
        self.local=basepath+local
        self.trainfile=basepath+trainfile+cl
        self.imagepath=basepath+imagepath
        self.annpath=basepath+annpath
        fd=open(self.trainfile,"r")
        self.trlines=fd.readlines()
        fd.close()
        if select=="all":#All images
            self.str=""
        if select=="pos":#Positives images
            self.str="1\n"
        if select=="neg":#Negatives images
            self.str="-1\n"
        self.selines=self.__selected()
        self.listparts=[]
        dl=len(self.selines)/numparts
        l=part
        self.listparts=self.selines[:(l+1)*dl]
        if l==numparts-1:
            self.listparts=self.selines[:]        
        self.part=part

    def __selected(self):
        lst=[]
        for id,it in enumerate(self.trlines):
            if self.str=="" or it.split(" ")[-1]==self.str:
                lst.append(it)
        return lst
        
    def getDBname(self):
        return "VOC07"
    
    def getStorageDir(self):
        return self.local#"/media/DADES-2/VOC2007/VOCdevkit/local/VOC2007/"
        
    def getImage(self,i):
        #item=self.selines[i]
        item=self.listparts[i]
        return myimread((self.imagepath+item.split(" ")[0])+".jpg")
    
    def getImageRaw(self,i):
        #item=self.selines[i]
        item=self.listparts[i]
        return im.open((self.imagepath+item.split(" ")[0])+".jpg")#pil.imread((self.imagepath+item.split(" ")[0])+".jpg")    
    
    def getImageByName(self,name):
        return myimread(name)

    def getImageByName2(self,name):
        return myimread(self.imagepath+name+".jpg")
    
    def getImageName(self,i):
        #item=self.selines[i]
        item=self.listparts[i]
        return (self.imagepath+item.split(" ")[0]+".jpg")
    
    def getTotal(self):
        #return len(self.selines)
        return len(self.listparts)
    
    def getBBox(self,i,part=-1,cl=None,usetr=None,usedf=None):
        if usetr==None:
            usetr=self.usetr
        if usedf==None:
            usedf=self.usedf
        if cl==None:#use the right class
            cl=self.cl.split("_")[0]
        #item=self.selines[i]
        item=self.listparts[i]
        filename=self.annpath+item.split(" ")[0]+".xml"
        return getbboxVOC07(filename,cl,usetr,usedf)

class VOC07Join(VOC06Data):
    """
    VOC07 instance (you can choose positive or negative images with the option select)
    """
    def __init__(self,numparts,part1,part2,select="all",cl="person_train.txt",
                basepath="media/DADES-2/",
                trainfile="VOCdevkit/VOC2007/ImageSets/Main/",
                imagepath="VOCdevkit/VOC2007/JPEGImages/",
                annpath="VOCdevkit/VOC2007/Annotations/",
                local="VOCdevkit/local/VOC2007/",
                usetr=False,usedf=False):
        self.cl=cl
        self.usetr=usetr
        self.usedf=usedf
        self.local=basepath+local
        self.trainfile=basepath+trainfile+cl
        self.imagepath=basepath+imagepath
        self.annpath=basepath+annpath
        fd=open(self.trainfile,"r")
        self.trlines=fd.readlines()
        fd.close()
        if select=="all":#All images
            self.str=""
        if select=="pos":#Positives images
            self.str="1\n"
        if select=="neg":#Negatives images
            self.str="-1\n"
        self.selines=self.__selected()
        self.listparts=[]
        dl=len(self.selines)/numparts
        lstart=part1
        lend=part2
        self.listparts=self.selines[(lstart)*dl:(lend+1)*dl]
        if lend==numparts:
            self.listparts=self.selines[(lstart)*dl:]        
        self.part=part2-part1

    def __selected(self):
        lst=[]
        for id,it in enumerate(self.trlines):
            if self.str=="" or it.split(" ")[-1]==self.str:
                lst.append(it)
        return lst
        
    def getDBname(self):
        return "VOC07"
    
    def getStorageDir(self):
        return self.local#"/media/DADES-2/VOC2007/VOCdevkit/local/VOC2007/"
        
    def getImage(self,i):
        #item=self.selines[i]
        item=self.listparts[i]
        return myimread((self.imagepath+item.split(" ")[0])+".jpg")
    
    def getImageRaw(self,i):
        #item=self.selines[i]
        item=self.listparts[i]
        return im.open((self.imagepath+item.split(" ")[0])+".jpg")#pil.imread((self.imagepath+item.split(" ")[0])+".jpg")    
    
    def getImageByName(self,name):
        return myimread(name)
    
    def getImageByName2(self,name):
        return myimread(self.imagepath+name+".jpg")

    def getImageName(self,i):
        #item=self.selines[i]
        item=self.listparts[i]
        return (self.imagepath+item.split(" ")[0]+".jpg")
    
    def getTotal(self):
        #return len(self.selines)
        return len(self.listparts)
    
    def getBBox(self,i,part=-1,cl=None,usetr=None,usedf=None):
        if usetr==None:
            usetr=self.usetr
        if usedf==None:
            usedf=self.usedf
        if cl==None:#use the right class
            cl=self.cl.split("_")[0]
        #item=self.selines[i]
        item=self.listparts[i]
        filename=self.annpath+item.split(" ")[0]+".xml"
        return getbboxVOC07(filename,cl,usetr,usedf)


class DirImages(VOC06Data):
    """
    VOC07 instance (you can choose positive or negative images with the option select)
    """
    def __init__(self,select="all",cl="person_train.txt",
                basepath="media/DADES-2/",
                trainfile="VOCdevkit/VOC2007/ImageSets/Main/",
                imagepath="VOCdevkit/VOC2007/JPEGImages/",
                annpath="",#VOCbase+"VOCdevkit/VOC2007/Annotations/",
                local="/tmp/",#VOCbase+"VOCdevkit/local/VOC2007/",
                usetr=False,usedf=False,ext=".png"):
        import glob
        self.cl=cl
        self.usetr=usetr
        self.usedf=usedf
        self.local=basepath+local
        self.trainfile=basepath+trainfile+cl
        self.imagepath=imagepath
        self.annpath=basepath+annpath
        self.ext=ext
        self.selines=glob.glob(self.imagepath+"/*"+ext)
        self.selines.sort()
        
##    def __selected(self):
##        lst=[]
##        for id,it in enumerate(self.trlines):
##            if self.str=="" or it.split(" ")[-1]==self.str:
##                lst.append(it)
##        return lst

    def getDBname(self):
        return "Images"
    
    def getStorageDir(self):
        return self.local
        
    def getImage(self,i):
        item=self.selines[i]
        return myimread(item)
    
##    def getImageRaw(self,i):
##        item=self.selines[i]
##        return im.open((self.imagepath+item.split(" ")[0])+".jpg")#pil.imread((self.imagepath+item.split(" ")[0])+".jpg")    
    
    def getImageByName(self,name):
        return myimread(name)
    
    def getImageName(self,i):
        item=self.selines[i]
        return (item)
    
    def getTotal(self):
        return len(self.selines)
    
##    def getBBox(self,i,cl=None,usetr=None,usedf=None):
##        if usetr==None:
##            usetr=self.usetr
##        if usedf==None:
##            usedf=self.usedf
##        if cl==None:#use the right class
##            cl=self.cl.split("_")[0]
##        item=self.selines[i]
##        filename=self.annpath+item.split(" ")[0]+".xml"
##        return getbboxVOC07(filename,cl,usetr,usedf)

import glob
import util

class CaltechData(imageData):
    """
    Caltech instance (you can choose positive or negative images with the option select)
    """
    def __init__(self,select="all",cl="airplanes",num=10,nset=0,
                        basepath="/home/databases/101_ObjectCategories/",
                        #trainfile="VOC2006/VOCdevkit/VOC2006/ImageSets/",
                        #imagepath="VOC2006/VOCdevkit/VOC2006/PNGImages/",
                        #annpath="VOC2006/VOCdevkit/VOC2006/Annotations/",
                        local="local/"):
        self.cl=cl
        self.num=num
        self.path=basepath
        self.local=basepath+local
        self.select=select
        #import glob
        self.classes=glob.glob(basepath+"*")
        self.clidx=self.classes[0]
        self.clpos=0
        lf={}
        self.lts=[]
        if nset==-1:#random
            lf=glob.glob(basepath+cl+"/*.jpg")
            numpy.random.shuffle(lf)
        else:
            for acl in self.classes:
                acl1=acl.split("/")[-1]
                try:
                    
                    print "Loading class ",acl," set ",nset
                    lf[acl1]=util.load(basepath+acl1+"/sample%d"%nset)            
                except:
                    print "Loading failed",acl
                    lf[acl1]=glob.glob(basepath+acl1+"/*.jpg")
                    numpy.random.shuffle(lf[acl1])
                    util.save(basepath+acl1+"/sample%d"%nset,lf[acl1],prt=0)            
                    print "Shuffling and saving class ",acl," set ",nset
                self.lts=self.lts+lf[acl1][self.num:self.num+5]
        self.lf=lf
    
    def getDBname(self):
        return "Caltech101"
        
    def getImage(self,i):
        if self.select=="trpos":
        #item=self.selines[i]
            #print self.cl;raw_input()
            return myimread(self.lf[self.cl][i])
        if self.select=="trneg":
            acl=self.cl
            while (acl==self.cl):
                val=numpy.random.random_integers(len(self.classes))-1
                acl=self.classes[val].split("/")[-1]
            #alf=glob.glob(self.path+acl+"/*.jpg")
            #numpy.random.shuffle(alf)
            #print alf[i]
            return myimread(self.lf[acl][i%self.num])
        if self.select=="tspos":
        #item=self.selines[i]
            return myimread(self.lf[self.cl][i+self.num])
        if self.select=="tsall":
        #item=self.selines[i]
            #acl=self.cl
            #while (acl==self.cl):
            #    acl=self.classes[numpy.random.random_integers(len(self.classes))].split("/")       
            return myimread(self.lts[i])
    
    def getImageByName2(self,name):
        return myimread(self.imagepath+name+".png")

    def getImageByName(self,name):
        #if sel.select=="tsall":
        #    ll=self.lts[i].split("/")
        #    nstr="/".join(ll[:-1])+"/"+ll[-1].split("_")[-1]
        return myimread(name)
    
    def getImageName(self,i):
        if self.select=="trpos":
        #item=self.selines[i]
            #print self.cl;raw_input()
            return self.lf[self.cl][i]
        if self.select=="trneg":
            acl=self.cl
            while (acl==self.cl):
                acl=self.classes[numpy.random.random_integers(len(self.classes))-1].split("/")[-1]
            #alf=glob.glob(self.path+acl+"/*.jpg")
            #numpy.random.shuffle(alf)
            #print alf[i]
            return self.lf[acl][i%self.num]
        if self.select=="tspos":
            return self.lf[self.cl][i+self.num]
        if self.select=="tsall":
            #ll=self.lts[i].split("/")
            #nstr="/".join(ll[:-1])+"/"+ll[-2]+"_"+ll[-1]
            return self.lts[i]

    def getImageRaw(self,i):
        item=self.selines[i]
        return im.open((self.imagepath+item.split(" ")[0])+".png")#pil.imread((self.imagepath+item.split(" ")[0])+".jpg")    
    
    def getStorageDir(self):
        return self.local
    
    def getBBox(self,i,cl=None,usetr=None,usedf=None):
        if self.select=="tsall":
            if self.lts[i].split("/")[-2]!=self.cl:
                return []
            else:
                aux=myimread(self.lts[i])
                dy=aux.shape[0]/10
                dx=aux.shape[1]/10
                bb=[[0,0,aux.shape[0],aux.shape[1]]]#ymin,xmin,ymax,xmax
                return bb
        aux=myimread(self.lf[self.cl][i])
        dy=aux.shape[0]/10
        dx=aux.shape[1]/10
        #bb=[[0,0,aux.shape[0],aux.shape[1]]]#ymin,xmin,ymax,xmax
        bb=[[0+dy,0+dx,aux.shape[0]-dy,aux.shape[1]-dx]]#ymin,xmin,ymax,xmax
        return bb

    def getBBoxByName(self,name):#,cl=None,usetr=None,usedf=None):
        if self.select=="tsall":
            if name.split("/")[-2]!=self.cl:
                return []
            else:
                aux=myimread(self.lts[i])
                dy=aux.shape[0]/10
                dx=aux.shape[1]/10
                bb=[[0,0,aux.shape[0],aux.shape[1]]]#ymin,xmin,ymax,xmax
                return bb
        aux=myimread(name)
        dy=aux.shape[0]/10
        dx=aux.shape[1]/10
        return [[0+dy,0+dx,aux.shape[0]-dy,aux.shape[1]-dx]]
        #[[0,0,aux.shape[0],aux.shape[1]]]
            
    def getTotal(self):
        if self.select=="tspos":
            return len(self.lf[self.cl])-self.num
        if self.select=="tsall":
            return len(self.lts)
        return self.num

class WebCam(VOC06Data):
    """
    VOC07 instance (you can choose positive or negative images with the option select)
    """
    def __init__(self,select="all",cl="person_train.txt",
                basepath="media/DADES-2/",
                trainfile="VOCdevkit/VOC2007/ImageSets/Main/",
                imagepath="VOCdevkit/VOC2007/JPEGImages/",
                annpath="",#VOCbase+"VOCdevkit/VOC2007/Annotations/",
                local="/tmp/",#VOCbase+"VOCdevkit/local/VOC2007/",
                usetr=False,usedf=False,ext=".png",cv=None,capture=None):
        import glob
        self.cl=cl
        self.usetr=usetr
        self.usedf=usedf
        self.local=basepath+local
        self.trainfile=basepath+trainfile+cl
        self.imagepath=imagepath
        self.annpath=basepath+annpath
        self.ext=ext
        self.selines=glob.glob(self.imagepath+"/*"+ext)
        self.c=0
        self.cv=cv
        self.capture=capture
        
    def getDBname(self):
        return "WebCam"
    
    def getStorageDir(self):
        return self.local
        
    def getImage(self,i):
        #item=self.selines[i]
        frame = self.cv.QueryFrame (self.capture)
        #self.cv.cvmat(frame)
        #print "after"
        img = numpy.asarray(self.cv.GetMat(frame))
        #img=img.astype(numpy.float)
        #img=opencv.cvIplImageAsNDarray(frame)
        self.c+=1
        return img
    
    def getImageByName(self,name):
        return myimread(name)
    
    def getImageName(self,i):
        item="WebCam.%d"%self.c#self.selines[i]
        return (item)
    
    def getTotal(self):
        return 100000

#import camera    

class Player(VOC06Data):
    """
    VOC07 instance (you can choose positive or negative images with the option select)
    """
    def __init__(self,select="all",cl="person_train.txt",
                basepath="media/DADES-2/",
                trainfile="VOCdevkit/VOC2007/ImageSets/Main/",
                imagepath="VOCdevkit/VOC2007/JPEGImages/",
                annpath="",#VOCbase+"VOCdevkit/VOC2007/Annotations/",
                local="/tmp/",#VOCbase+"VOCdevkit/local/VOC2007/",
                usetr=False,usedf=False,ext=".png",cv=None,capture=None):
        import glob
        self.cl=cl
        self.usetr=usetr
        self.usedf=usedf
        self.local=basepath+local
        self.trainfile=basepath+trainfile+cl
        self.imagepath=imagepath
        self.annpath=basepath+annpath
        self.ext=ext
        #self.selines=sort(glob.glob(self.imagepath+"/*"+ext))
        self.c=0
        self.client = camera.client_create(None,"158.109.9.201", 6665)
        #self.client = camera.client_create(None,"158.109.8.86", 6665)
        #self.client = camera.client_create(None,"158.109.9.212", 6665)
        camera.client_connect(self.client)
        self.cam = camera.camera_create(self.client, 0)
        #self.cam = camera.camera_create(self.client, 1)
        camera.camera_subscribe(self.cam, 1)
        
    def getDBname(self):
        return "WebCam"
    
    def getStorageDir(self):
        return self.local
        
    def getImage(self,i):
        #item=self.selines[i]
        #frame = self.cv.cvQueryFrame (self.capture)
        #img=self.cv.cvIplImageAsNDarray(frame)
        #self.c+=1
        camera.client_read(self.client)
        camera.camera_decompress(self.cam)
        #camera_save(cam, 'foo.ppm')
        # TESTING: This doesn't work...
        #print 'Width: %d'  % cam.contents.width;
        #print 'Height: %d' % cam.contents.height;
        data=camera.string_at(self.cam.contents.image,self.cam.contents.image_count)
        image=numpy.frombuffer(data,dtype=numpy.ubyte)
        image=image.reshape((self.cam.contents.height,self.cam.contents.width,3))
    #image1=image.copy()
        #pylab.clf()
        #pylab.ioff()
        #pylab.imshow(image)
        #pylab.show()
        #pylab.draw()
        #pylab.imshow(image)
        return image
    
    def getImageByName(self,name):
        return myimread(name)
    
    def getImageName(self,i):
        item="WebCam.%d"%self.c#self.selines[i]
        return (item)
    
    def getTotal(self):
        return 100000
    

class ImgFile(VOC06Data):
    """
    Read images and BB from a pascal format detection file
    """
    def __init__(self,trainfile,imgpath="",local="/tmp/"):
        import glob
        fd=open(self.trainfile,"r")
        trlines=fd.readlines()
        fd.close()
        images={}
        for l in trlines:
            line=l.split()[0]
            if images.has_key(line[0]):
                images[line[0]].append(line[1:])
            else:
                images[line[0]]=[line[1:]]
        self.limages=[l for l in images.iterkeys()]#images.keys()#[l for l in images.iterkeys()]
        self.tot=len(self.limages)
        
    def getDBname(self):
        return "Images+BB"
    
    def getStorageDir(self):
        return self.local
        
    def getImage(self,i):
        return myimread(self.limages[i])
    
    def getImageByName(self,name):
        return myimread(name)
    
    def getImageName(self,i):
        return self.limages[i]
    
    def getBBox(self,i,cl=None,usetr=None,usedf=None):
        bb=self.limages[i]
        return bb

    def getTotal(self):
        return self.tot


