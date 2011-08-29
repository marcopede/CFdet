#used to draw the HOG features
import numpy, pylab

def drawLine(img,py,px,ang,l,val=1):
    incr=numpy.tan(ang/180.0*numpy.pi)
    stp=+1
    while ang>180+45:
        ang=ang-360
    while ang<-135:
        ang=ang+360
    if ang>-135 and ang<=-45:
        ang=ang+180
        incr=-incr    
        stp=-1
    if ang>135 and ang<=180+45:
        ang=ang-180
        incr=-incr    
        stp=-1
    if ang>-45 and ang<=45:
        for x in range(l):
            dy=x*incr
            idy=numpy.floor(dy)
            w1=dy-idy
            img[px+x*stp,py+idy]=max(img[px+x*stp,py+idy],val*(1-w1))
            img[px+x*stp,py+idy+1]=max(img[px+x*stp,py+idy+1],val*(w1))
    if ang>45 and ang<=135:
        incr=1/incr
        for y in range(l):
            dx=y*incr
            idx=numpy.floor(dx)
            w1=dx-idx
            img[px+idx,py+y*stp]=max(img[px+idx,py+y*stp],val*(1-w1))
            img[px+idx+1,py+y*stp]=max(img[px+idx+1,py+y*stp],val*(w1))


def draw1HOG(img,hog,py,px,ry,rx):
    lhog=len(hog)
    for o in range(lhog):
        ang=o*numpy.pi/float(lhog)
        drawLine(img,px+rx-1,py+ry-1,-ang/numpy.pi*180,ry,hog[o])
        drawLine(img,px+rx-1,py+ry-1,-ang/numpy.pi*180+180,ry,hog[o])
        
def drawHOG(feat,hogpix=15):
    r=(hogpix+1)/2
    dimy=feat.shape[0]
    dimx=feat.shape[1]
    img=numpy.zeros(((dimy)*hogpix,(dimx)*hogpix))
    for y in range(dimy):
        for x in range(dimx):
            #draw1HOG(img,feat[y,x,:9]+0.5*feat[y,x,9:18]+0.5*feat[y,x,18:27],y*hogpix,x*hogpix,r,r)
            draw1HOG(img,feat[y,x,:9]+feat[y,x,9:18]+feat[y,x,18:27],y*hogpix,x*hogpix,r,r)
    return img

if __name__=="__main__":

    a=numpy.zeros((45,45))
    #drawLine(a,50,50,360,50,1)
    hog=[1,1,1,1,1,1,1,1,1,1]
    draw1HOG(a,hog,30,15,8,8)
    pylab.figure(1)
    pylab.imshow(a,cmap=pylab.cm.gray,interpolation="nearest")
    pylab.show()


    
