
#include<stdio.h>
#include<stdlib.h>
//#include<math.h>

#define ftype float
#define INFINITY 10000

static ftype k=1.0;//deformation coefficient

void setK(ftype pk)
{
    k=pk;
}

ftype getK()
{
    return k;
}

static compHOG=0; //number of HOG computed

void resetHOG()
{
    compHOG=0;
}

long getHOG()
{
    return compHOG;
}

//compute 3d correlation between an image feature img and a mask 
inline ftype corr3dpad(ftype *img,int imgy,int imgx,ftype *mask,int masky,int maskx,int dimz,int posy,int posx,ftype *prec,int pady,int padx,int occl)
{
    int dimzfull=dimz;
    if (occl!=0)
        //with occl
        dimz=dimz-occl;
        //printf("Occl:%d Dimzfull:%d Dimz:%d",occl,dimzfull,dimz);
    if (prec!=NULL)//memoization of the already computed locations
    {
        if (posy>=-pady && posy<imgy+pady && posx>=-padx && posx<imgx+padx)
            if (prec[(posy+pady)*(imgx+2*padx)+(posx+padx)]>-INFINITY)
            {
                return prec[(posy+pady)*(imgx+2*padx)+(posx+padx)];
            }
    }    
    ftype sum=0.0;
    int x,y,z,posi;
    for (x=0;x<maskx;x++)
        for (y=0;y<masky;y++)
        {
            compHOG++;
            if (((x+posx)>=0 && (x+posx<imgx)) && ((y+posy)>=0 && (y+posy<imgy)))
            //inside the image
            {
                for (z=0;z<dimz;z++)
                {   
                    //printf("%d:%f\n",z,mask[z+x*dimzfull+y*dimzfull*maskx]);
                    posi=z+(x+posx)*dimz+(y+posy)*dimz*imgx;
                    sum=sum+img[posi]*mask[z+x*dimzfull+y*dimzfull*maskx];      
                    /*posi=z+(x+posx)*dimzfull+(y+posy)*dimzfull*imgx;
                    {
                        sum=sum+img[posi]*mask[z+x*dimzfull+y*dimzfull*maskx];      
                    }*/
                }
            }
            else
            //occlusion using dimz
            {
                for (z=dimz;z<dimzfull;z++)
                {
                    //printf("%d:%f\n",z,mask[z+x*dimzfull+y*dimzfull*maskx]);
                    //posi=z+(x+posx)*dimz+(y+posy)*dimz*imgx;
                    sum=sum+mask[z+x*dimzfull+y*dimzfull*maskx];      
                }
            }
        }
    if (prec!=NULL)//save computed values in the buffer
    {
        if (posy>=-pady && posy<imgy+pady && posx>=-padx && posx<imgx+padx)
        {
            prec[(posy+pady)*(imgx+2*padx)+(posx+padx)]=sum;
        }
    }
    return sum;
}

static ftype buffer[5*5*9];//for BOW part should be smaller than 50
static int bcode[50*50];//for BOW
static ftype localhist[2000];
//static int table[10000];//(9 max orient+ null)^4
//static variables for the moment
//int sizevoc=2;
//int numvoc=100;
//ftype *voc
//ftype *mhist

/*void filltable(int *tab)
{
    int c;
    for (c=0;c<625;c++)
        table[c]=tab[c];
}*/

#define NUMOR 5
static int sel[NUMOR]={0,2,4,5,7};//selected orientations
static ftype pow5[]={1,5,5*5,5*5*5};
static ftype pow6[]={1,6,6*6,6*6*6};

//compute 3d correlation between an image feature img and a mask 
inline ftype corr3dpadbow(ftype *img,int imgy,int imgx,ftype *mask,int masky,int maskx,int dimz,int posy,int posx,ftype *prec,int pady,int padx,int occl,int sizevoc,int numvoc,int *code,int codex,ftype *mhist
)//sizevoc=2;numvoc=100;voc=2*2*9*100
{
    //printf("PARAM: %d %d\n",sizevoc,numvoc);
    int dimz2=9;//only first 9 oriented gradients
    int dimzfull=dimz;
    if (occl!=0)
        //with occl
        dimz=dimz-occl;
        //printf("Occl:%d Dimzfull:%d Dimz:%d",occl,dimzfull,dimz);
    if (prec!=NULL)//memoization of the already computed locations
    {
        if (posy>=-pady && posy<imgy+pady && posx>=-padx && posx<imgx+padx)
            if (prec[(posy+pady)*(imgx+2*padx)+(posx+padx)]>-INFINITY)
            {
                return prec[(posy+pady)*(imgx+2*padx)+(posx+padx)];
            }
    }    
    ftype sum=0.0,dist,mindist=10000.0;//,bowscr=0.0;
    int x,y,z,posi,c,cy,cx,bestc=0,toton=0;
    for (c=0;c<numvoc;c++)
        localhist[c]=0.0;
    for (x=0;x<maskx;x++)
        for (y=0;y<masky;y++)
        {
            compHOG++;
            if (((x+posx)>=0 && (x+posx<imgx)) && ((y+posy)>=0 && (y+posy<imgy)))
            //inside the image
            {
                for (z=0;z<dimz;z++)
                {   
                    //printf("%d:%f\n",z,mask[z+x*dimzfull+y*dimzfull*maskx]);
                    posi=z+(x+posx)*dimz+(y+posy)*dimz*imgx;
                    sum=sum+img[posi]*mask[z+x*dimzfull+y*dimzfull*maskx];      
                    /*posi=z+(x+posx)*dimzfull+(y+posy)*dimzfull*imgx;
                    {
                        sum=sum+img[posi]*mask[z+x*dimzfull+y*dimzfull*maskx];      
                    }*/
                }
            }
            else
            //occlusion using dimz
            {
                for (z=dimz;z<dimzfull;z++)
                {
                    //printf("%d:%f\n",z,mask[z+x*dimzfull+y*dimzfull*maskx]);
                    //posi=z+(x+posx)*dimz+(y+posy)*dimz*imgx;
                    sum=sum+mask[z+x*dimzfull+y*dimzfull*maskx];      
                }
            }
        }
//bow extraction
    /*for (x=0;x<maskx-sizevoc+1;x++)
    {
        for (y=0;y<masky-sizevoc+1;y++)
        {
            for (cy=0;cy<sizevoc;cy++)
            {
                for (cx=0;cx<sizevoc;cx++)
                {
                    //compHOG++;
                    if (((x+cx+posx)>=0 && (x+cx+posx<imgx)) && ((y+cy+posy)>=0 && (y+cy+posy<imgy)))
                    //inside the image
                    {
                        for (z=0;z<dimz2;z++)//using only 9 not oriented
                        {   
                            posi=18+z+(x+cx+posx)*dimz+(y+cy+posy)*dimz*imgx;
                            buffer[z+cx*dimz2+cy*dimz2*sizevoc]=img[posi];
                        }
                    }
                    else
                    //occlusion
                    {
                        for (z=0;z<dimz2;z++)
                        {
                            buffer[z+cx*dimz2+cy*dimz2*sizevoc]=0;
                        }
                    }
                }
            }
            //bow assignment (then with lookup table)
            /*numvoc=1;
            for (c=0;c<numvoc;c++)
            {
                dist=0.0;
                for (z=0;z<dimz2*sizevoc*sizevoc;z++)
                {
                    dist+=(voc[z+c*dimz2*sizevoc*sizevoc]-buffer[z])*(voc[z+c*dimz2*sizevoc*sizevoc]-buffer[z]);
                }
                //printf("Dist:%f\n",dist);
                if (dist<mindist)
                {
                    mindist=dist;
                    bestc=c;
                }
            }*/
            //lookup table
           /* ftype maxval,val;
            int maxpos=0;
            int aux,pos=0;
            for (c=0;c<sizevoc*sizevoc;c++)
            {
                maxval=0.0;
                maxpos=0;
                for (z=0;z<NUMOR;z++)
                {
                    aux=sel[z];
                    val=2.0*buffer[aux+c*dimz2]+buffer[aux+1+c*dimz2]+buffer[(aux+8)%9+c*dimz2];
                    //val=buffer[sel[z]+c*dimz2];
                    if (val>maxval)
                    {
                        maxval=val;
                        maxpos=z;
                    }
                }
                if (maxval==0.0)
                    maxpos=NUMOR;
                //if (voc[0]==10.0)
                //    printf("%d %3f ",maxpos,maxval);
                pos+=maxpos*pow6[c];//compute the hash code
            }
            //pos=0;
            //bestc=pos;//table[pos];
            //if (voc[0]==10.0)
            //    printf(" Best C:%d\n",pos);
            if (localhist[pos]!=1.0)
                toton++;
//            localhist[pos]=1.0;//maxpooling
            localhist[pos]=1.0;
            //localhist[pos]= (localhist[pos]==0.0)?toton++
        }
    }*/
    int pos;
    for (x=0;x<maskx-sizevoc+1;x++)
    {
        for (y=0;y<masky-sizevoc+1;y++)
        {
            pos=code[x+y*codex];
            //printf("code = %d!!!\n",pos);
            if (localhist[pos]!=1.0)
                toton++;
            localhist[pos]=1.0;
        }
    }
    //printf("here done!!!\n");
    /*ftype nval=0.0;
    for (c=0;c<numvoc;c++)
    {
        nval+=localhist[c];//*localhist[c];
    }*/
    //if (voc[0]==10.0)
    //    printf("Norm %f",sqrt(nval));
//normalization l2 + score
    ftype bowscr=0.0;
    for (c=0;c<numvoc;c++)
    {
        //printf("toton%d\n",toton);
        //if (localhist[c]==1.0)
        localhist[c]=localhist[c]/sqrt(toton);
        //localhist[c]=(ftype)(localhist[c])/sqrt(nval);
        //else
        //    localhist[c]=0.0;          
        bowscr+=(localhist[c])*(mhist[c]);
        /*if (mhist[c]>1000.0)
        {
            printf("%d %f %f %f \n",c,bowscr,localhist[c],mhist[c]);
            return;
            //scanf("%f",&voc[0]);
        }*/
            //flush();
            //exit(1);
        //if (voc[0]==10.0)
        //    printf("%f ",localhist[c]);
    }
    //printf("\nNorm:%d\n",toton);
    /*if (voc[0]==10.0)
    {
        printf("Scr:%f\n",sum);
        printf("Bowscr:%f\n",bowscr);
    }*/
//end bow
    if (prec!=NULL)//save computed values in the buffer
    {
        if (posy>=-pady && posy<imgy+pady && posx>=-padx && posx<imgx+padx)
        {
            prec[(posy+pady)*(imgx+2*padx)+(posx+padx)]=sum;
        }
    }
    //if (voc[0]==10.0)
    //    return bowscr;
    return sum+(ftype)bowscr;
}


//compute the score over the possible values of a defined neighborhood
inline ftype refineigh(ftype *img,int imgy,int imgx,ftype *mask,int masky,int maskx,int dimz,int posy,int posx,int rady,int radx,int *posedy, int *posedx, ftype *prec,int occl)
{
    int iy,ix;  
    ftype val,maxval=-1000;
    for (iy=-rady;iy<=rady;iy++)
    {
        for (ix=-radx;ix<=radx;ix++)
        {
            val=corr3dpad(img,imgy,imgx,mask,masky,maskx,dimz,posy+iy,posx+ix,prec,0,0,occl);
            if (val>maxval)
            {
                maxval=val;
                *posedy=iy;
                *posedx=ix;
            }
        }
    }
    return maxval;
}

inline ftype hog2bow(ftype *img,int imgx,int imgy,ftype *mask,int masky,int maskx,int dimz,int posy,int posx,int sizevoc,int *code)
{
    //bow extraction
    int dimz2=9;
    int x,y,z,cy,cx,posi,c,maxpos,aux,pos;
    ftype maxval,val;
    for (x=0;x<maskx-sizevoc+1;x++)
    {
        for (y=0;y<masky-sizevoc+1;y++)
        {
            for (cy=0;cy<sizevoc;cy++)
            {
                for (cx=0;cx<sizevoc;cx++)
                {
                    //compHOG++;
                    if (((x+cx+posx)>=0 && (x+cx+posx<imgx)) && ((y+cy+posy)>=0 && (y+cy+posy<imgy)))
                    //inside the image
                    {
                        for (z=0;z<dimz2;z++)//using only 9 not oriented
                        {   
                            posi=18+z+(x+cx+posx)*dimz+(y+cy+posy)*dimz*imgx;
                            buffer[z+cx*dimz2+cy*dimz2*sizevoc]=img[posi];
                        }
                    }
                    else
                    //occlusion
                    {
                        for (z=0;z<dimz2;z++)
                        {
                            buffer[z+cx*dimz2+cy*dimz2*sizevoc]=0;
                        }
                    }
                }
            }
            pos=0;
            for (c=0;c<sizevoc*sizevoc;c++)
            {
                maxval=0.0;
                maxpos=0;
                for (z=0;z<NUMOR;z++)
                {
                    aux=sel[z];
                    val=2.0*buffer[aux+c*dimz2]+buffer[aux+1+c*dimz2]+buffer[(aux+8)%9+c*dimz2];
                    //val=buffer[sel[z]+c*dimz2];
                    if (val>maxval)
                    {
                        maxval=val;
                        maxpos=z;
                    }
                }
                if (maxval==0.0)
                    maxpos=NUMOR;
                /*if (voc[0]==10.0)
                    printf("%d %3f ",maxpos,maxval);*/
                pos+=maxpos*pow6[c];//compute the hash code
            }
            code[x+y*(maskx-sizevoc+1)]=pos;
            //printf("code y=%d x=%d is %d\n",y,x,pos);
        }
    }            
}


/*inline void codify(ftype *buffer,int by,int bx,int dimz2,int sizevoc,int *code)
{
    ftype maxval,val;
    int maxpos=0;
    int aux,pos=0,y,x,c,z;
    for (x=0;x<bx;x++)
    {
        for (y=0;y<by;y++)
        {
            pos=0;
            for (c=0;c<sizevoc*sizevoc;c++)
            {
                maxval=0.0;
                maxpos=0;
                for (z=0;z<NUMOR;z++)
                {
                    aux=sel[z];
                    val=2.0*buffer[aux+c*dimz2]+buffer[aux+1+c*dimz2]+buffer[(aux+8)%9+c*dimz2];
                    //val=buffer[sel[z]+c*dimz2];
                    if (val>maxval)
                    {
                        maxval=val;
                        maxpos=z;
                    }
                }
                if (maxval==0.0)
                    maxpos=NUMOR;
                pos+=maxpos*pow6[c];//compute the hash code
            }
            code[x+y*bx]=pos;
            printf("code y=%d x=%d is %d\n",y,x,pos);
        }
    }
    //pos=0;
    //bestc=pos;//table[pos];
    //if (voc[0]==10.0)
//        printf(" Best C:%d\n",pos);
    //if (localhist[pos]!=1.0)
    //    toton++;
//            localhist[pos]=1.0;//maxpooling
    //localhist[pos]=1.0;
    //localhist[pos]= (localhist[pos]==0.0)?toton++
}*/

//compute the score over the possible values of a defined neighborhood
inline ftype refineighbow(ftype *img,int imgy,int imgx,ftype *mask,int masky,int maskx,int dimz,int posy,int posx,int rady,int radx,int *posedy, int *posedx, ftype *prec,int occl,int sizevoc,int numvoc,ftype *voc,ftype *mhist)
{
    int iy,ix;  
    ftype val,maxval=-1000;
    int codey=(masky+2*radx)-sizevoc+1;
    int codex=(maskx+2*radx)-sizevoc+1;
    //precompute sift
    hog2bow(img,imgx,imgy,mask,masky+2*rady,maskx+2*radx,dimz,posy-rady,posx-radx,sizevoc,bcode);
    //printf("this done!!!\n");
    //codify(buffer,codey,codex,9,sizevoc,bcode);
    //printf("that done!!! y=%d x=%d\n",codey,codex);
    //precompute sift
    for (iy=-rady;iy<=rady;iy++)
    {
        for (ix=-radx;ix<=radx;ix++)
        {
            val=corr3dpadbow(img,imgy,imgx,mask,masky,maskx,dimz,posy+iy,posx+ix,prec,0,0,occl,sizevoc,numvoc,bcode+(ix+radx)+(iy+rady)*codex,codex,mhist);
            if (val>maxval)
            {
                maxval=val;
                *posedy=iy;
                *posedx=ix;
            }
        }
    }
    return maxval;
}

inline ftype refineighfull(ftype *img,int imgy,int imgx,ftype *mask,int masky,int maskx,int dimz,ftype dy,ftype dx,int posy,int posx,int rady,int radx,ftype *scr,int *rdy,int *rdx,ftype *prec,int pady,int padx,int occl)
{
    int iy,ix;  
    ftype val,maxval=-1000;
    //printf("dy:%d,dx:%d",rady,radx);
    //printf("k=%f",k);
    for (iy=-rady;iy<=rady;iy++)
    {
        for (ix=-radx;ix<=radx;ix++)
        {
            val=corr3dpad(img,imgy,imgx,mask,masky,maskx,dimz,posy+iy,posx+ix,prec,pady,padx,occl)+k*k*dy*(iy*iy)+k*k*dx*(ix*ix);
            scr[(iy+rady)*(2*radx+1)+(ix+radx)]+=-val;
            if (val>maxval)
            {
                maxval=val;
                *rdy=iy;
                *rdx=ix;
            }
        }
    }
    return maxval;
}

void scaneigh(ftype *img,int imgy,int imgx,ftype *mask,int masky,int maskx,int dimz,int *posy,int *posx,ftype *val,int *posey,int *posex,int rady, int radx,int len,int occl)
{   
    //return;
    int i;
    for (i=0;i<len;i++)
    {
        //if (posy[i]==-1  && posx[i]==-1)
        //    val[i]=0.0;
        //else
        val[i]=refineigh(img,imgy,imgx,mask,masky,maskx,dimz,posy[i],posx[i],rady,radx,posey++,posex++,NULL,occl);       
    }
}

void scaneighbow(ftype *img,int imgy,int imgx,ftype *mask,int masky,int maskx,int dimz,int *posy,int *posx,ftype *val,int *posey,int *posex,int rady, int radx,int len,int occl,int sizevoc,int numvoc,ftype *voc,ftype *mhist)
{   
    //return;
    int i;
    for (i=0;i<len;i++)
    {
        //if (posy[i]==-1  && posx[i]==-1)
        //    val[i]=0.0;
        //else
        val[i]=refineighbow(img,imgy,imgx,mask,masky,maskx,dimz,posy[i],posx[i],rady,radx,posey++,posex++,NULL,occl,sizevoc,numvoc,voc,mhist);       
    }
}


#define maxrad 5
static ftype scr[4*(2*maxrad+1)*(2*maxrad+1)];

//compute the score using dinamic programming
#include"dynamic.h"

void buildef(ftype *def,ftype dfy,ftype dfx,int rad)
{
    int l1,l2,dy1,dy2,dx1,dx2;
    int maxl=(2*rad+1)*(2*rad+1);
    for (l1=0;l1<maxl;l1++)
        for (l2=0;l2<maxl;l2++)
        {
            dx1=l1%(2*rad+1);
            dy1=l1/(2*rad+1);
            dx2=l2%(2*rad+1);
            dy2=l2/(2*rad+1);
            def[l1*maxl+l2]=k*k*dfy*(dy2-dy1)*(dy2-dy1)+k*k*dfx*(dx2-dx1)*(dx2-dx1);
        }
}

static ftype scr2[4*(2*maxrad+1)*(2*maxrad+1)];

void scanDef2(ftype *ww1,ftype *ww2,ftype *ww3,ftype *ww4,int fy,int fx,int dimz,ftype *df1,ftype *df2,ftype *df3,ftype *df4,ftype *img,int imgy,int imgx,int *posy,int *posx,int *parts,ftype *res,int rad,int len,int usemrf,ftype *uscr,int useusrc,ftype *prec,int pady,int padx,int occl)
{
    //printf("dimz:%d occl:%d",dimz,occl);
    int sizedf=4;
    ftype maxscr1,maxscr2,maxscr3,maxscr4;
    ftype aux,aux0,aux1,aux2,aux3,myaux;
    int i,c,p,rdef[4],rdef1[4],dy[4],dx[4];
    ftype maxm[(2*maxrad+1)*(2*maxrad+1)];
    buildef(def,df3[2],df3[3],rad);
    buildef(def+(2*rad+1)*(2*rad+1)*(2*rad+1)*(2*rad+1),df1[2],df1[3],rad);
    buildef(def+2*(2*rad+1)*(2*rad+1)*(2*rad+1)*(2*rad+1),df2[2],df2[3],rad);
    buildef(def+3*(2*rad+1)*(2*rad+1)*(2*rad+1)*(2*rad+1),df4[2],df4[3],rad);  
    for (i=0;i<len;i++)//for each location
    {
        if (posy[i]<-100)//if score too low skip (used for some threshold based filtering) 
        {
            continue;
        }
        int aord[]={0,1,3,2};
        if (useusrc)//load partial score for BU
        { 
            for (p=0;p<4;p++)
                for (c=0;c<(2*rad+1)*(2*rad+1);c++)
                    scr[aord[p]*(2*rad+1)*(2*rad+1)+c]=-uscr[p*len*(2*rad+1)*(2*rad+1)+c*len+i];
        }
        else//initialize score to 0 for CtF
        {
            for (c=0;c<4*(2*rad+1)*(2*rad+1);c++)
                scr[c]=0;
        }          
        ftype *prec1=NULL,*prec2=NULL,*prec3=NULL,*prec4=NULL;
        if (prec!=NULL)//use buffer for memoization
        {
            prec1=prec;
            prec2=prec+(imgy+2*pady)*(imgx+2*padx);
            prec3=prec+2*(imgy+2*pady)*(imgx+2*padx);
            prec4=prec+3*(imgy+2*pady)*(imgx+2*padx);
        }
        //compute the score of the 4 parts
        maxscr1 = refineighfull(img,imgy,imgx,ww1,fy,fx,dimz,df1[0],df1[1],posy[i],posx[i],rad,rad,scr,dy,dx,prec1,pady,padx,0);
        maxscr2 = refineighfull(img,imgy,imgx,ww2,fy,fx,dimz,df2[0],df2[1],posy[i],posx[i]+fx,rad,rad,scr+(2*rad+1)*(2*rad+1),dy+1,dx+1,prec2,pady,padx,0);
        maxscr3 = refineighfull(img,imgy,imgx,ww3,fy,fx,dimz,df3[0],df3[1],posy[i]+fy,posx[i],rad,rad,scr+3*(2*rad+1)*(2*rad+1),dy+2,dx+2,prec3,pady,padx,0);
        maxscr4 = refineighfull(img,imgy,imgx,ww4,fy,fx,dimz,df4[0],df4[1],posy[i]+fy,posx[i]+fx,rad,rad,scr+2*(2*rad+1)*(2*rad+1),dy+3,dx+3,prec4,pady,padx,0);
        if (usemrf)//use lateral constraints
        {
            for (c=0;c<4*(2*rad+1)*(2*rad+1);c++)
                scr[c]=-scr[c];//invert the score to keep compatibility
            filltables(scr,def,4,(2*rad+1)*(2*rad+1));
            res[i]=map(rdef,0,maxm);  
            for (c=0;c<4;c++)
            {
                dy[c]=(rdef[c]/(2*rad+1))-rad;
                dx[c]=(rdef[c]%(2*rad+1))-rad;
            }
            aux=dy[2];dy[2]=dy[3];dy[3]=aux;
            aux=dx[2];dx[2]=dx[3];dx[3]=aux;
        }
        else
            res[i]=maxscr1+maxscr2+maxscr3+maxscr4;
        
        //mydef is referered to the next part in clockwise way
        parts[0*len*sizedf+0*len+i]=dy[0]; //dy0
        parts[0*len*sizedf+1*len+i]=dx[0]; //dx0
        parts[0*len*sizedf+2*len+i]=(dy[0]-dy[1])*(dy[0]-dy[1]); //dy01
        parts[0*len*sizedf+3*len+i]=(dx[0]-dx[1])*(dx[0]-dx[1]); //dx01
        parts[1*len*sizedf+0*len+i]=dy[1]; //dy1
        parts[1*len*sizedf+1*len+i]=dx[1]; //dx1
        parts[1*len*sizedf+2*len+i]=(dy[1]-dy[3])*(dy[1]-dy[3]); //dy13
        parts[1*len*sizedf+3*len+i]=(dx[1]-dx[3])*(dx[1]-dx[3]); //dx13
        parts[2*len*sizedf+0*len+i]=dy[2]; //dy2
        parts[2*len*sizedf+1*len+i]=dx[2]; //dx2
        parts[2*len*sizedf+2*len+i]=(dy[2]-dy[0])*(dy[2]-dy[0]); //dy20
        parts[2*len*sizedf+3*len+i]=(dx[2]-dx[0])*(dx[2]-dx[0]); //dx20
        parts[3*len*sizedf+0*len+i]=dy[3]; //dy3
        parts[3*len*sizedf+1*len+i]=dx[3]; //dx3
        parts[3*len*sizedf+2*len+i]=(dy[3]-dy[2])*(dy[3]-dy[2]); //dy32
        parts[3*len*sizedf+3*len+i]=(dx[3]-dx[2])*(dx[3]-dx[2]); //dx32
       
    }
}


