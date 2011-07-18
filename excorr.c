
#include<stdio.h>
#include<stdlib.h>
// gcc -fPIC -c excorr.c -O3
//gcc -shared -Wl,-soname,libexcorr.so -o libexcorr.so  excorr.o
#define ftype float
#define INFINITY 10000
//#define k 0.3

static ftype k=1.0;

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
inline ftype corr3dpad(ftype *img,int imgy,int imgx,ftype *mask,int masky,int maskx,int dimz,int posy,int posx,ftype *prec,int pady,int padx)
{
    //pady=0;
    //padx=0;
    //printf("Value of prec:%d\n",prec==NULL);
    //printf("Pady, Padx: %d %d \n",pady,padx);
    if (prec!=NULL)
    {
        if (posy>=-pady && posy<imgy+pady && posx>=-padx && posx<imgx+padx)
            if (prec[(posy+pady)*(imgx+2*padx)+(posx+padx)]>-INFINITY)
            {
                //printf("Value Already Computed at location:(%d,%d)\n",posy,posx);
                return prec[(posy+pady)*(imgx+2*padx)+(posx+padx)];
            }
    }    
    //img[y,x,z]
    ftype sum=0.0;
    int x,y,z,posi;
    //posy=posy//-(masky)/2;
    //posx=posx//-(maskx)/2;
    for (x=0;x<maskx;x++)
        for (y=0;y<masky;y++)
        {
            compHOG++;
            if (((x+posx)>=0 && (x+posx<imgx)) && ((y+posy)>=0 && (y+posy<imgy)))
            {
                for (z=0;z<dimz;z++)
                {   
                    posi=z+(x+posx)*dimz+(y+posy)*dimz*imgx;
                    {
                        sum=sum+img[posi]*mask[z+x*dimz+y*dimz*maskx];      
                    }
                }
            }
        }
    if (prec!=NULL)
    {
        if (posy>=-pady && posy<imgy+pady && posx>=-padx && posx<imgx+padx)
        {
            //printf("Writing location:(%d,%d)\n",posy,posx);
            prec[(posy+pady)*(imgx+2*padx)+(posx+padx)]=sum;
        }
    }
    return sum;
}

inline ftype refineigh(ftype *img,int imgy,int imgx,ftype *mask,int masky,int maskx,int dimz,int posy,int posx,int rady,int radx,int *posedy, int *posedx, ftype *prec)
{
    int iy,ix;  
    ftype val,maxval=-1000;
    //printf("dy:%d,dx:%d",rady,radx);
    for (iy=-rady;iy<=rady;iy++)
    {
        for (ix=-radx;ix<=radx;ix++)
        {
            //val=corr3d(img,imgy,imgx,mask,masky,maskx,dimz,posy+iy,posx+ix,prec);
            val=corr3dpad(img,imgy,imgx,mask,masky,maskx,dimz,posy+iy,posx+ix,prec,0,0);
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

inline ftype refineighpr(ftype *img,int imgy,int imgx,ftype *mask,int masky,int maskx,int dimz,int posy,int posx,int rady,int radx,int *posedy, int *posedx, ftype *prec,ftype *pr)
{
    int iy,ix;  
    ftype val,maxval=-100;
    //printf("dy:%d,dx:%d",rady,radx);
    for (iy=-rady;iy<=rady;iy++)
    {
        for (ix=-radx;ix<=radx;ix++)
        {
            //val=corr3d(img,imgy,imgx,mask,masky,maskx,dimz,posy+iy,posx+ix,prec);
            val=corr3dpad(img,imgy,imgx,mask,masky,maskx,dimz,posy+iy,posx+ix,prec,0,0);
            if (pr!=NULL)
            {
                if (posy+iy>=0 && posy+iy<imgy && posx+ix>=0 && posx+ix<imgx)
                {
                    if (pr[(posy+iy)*imgx+posx+ix]==0.0)
                    {
                        //printf("Not in prior!!\n");
                        val=-10;
                    }  
                    else
                    {
                        //printf("Prior in %d %d\n",posy+iy,posx+ix);                 
                        //val=10;//pr[(posy+iy)*imgx+posx+ix];
                        //val=pr[(posy+iy)*imgx+posx+ix];
                    }
                }
                //else
                //    val=-10;
            }
            else
            {
                //val=33;
                //printf("NULL pointer %f\n",val);
            }
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

inline ftype refineighfull(ftype *img,int imgy,int imgx,ftype *mask,int masky,int maskx,int dimz,ftype dy,ftype dx,int posy,int posx,int rady,int radx,ftype *scr,int *rdy,int *rdx,ftype *prec,int pady,int padx)
{
    int iy,ix;  
    ftype val,maxval=-1000;
    //printf("dy:%d,dx:%d",rady,radx);
    //printf("k=%f",k);
    for (iy=-rady;iy<=rady;iy++)
    {
        for (ix=-radx;ix<=radx;ix++)
        {
            val=corr3dpad(img,imgy,imgx,mask,masky,maskx,dimz,posy+iy,posx+ix,prec,pady,padx)+k*k*dy*(iy*iy)+k*k*dx*(ix*ix);
            //val=corr3d(img,imgy,imgx,mask,masky,maskx,dimz,posy+iy,posx+ix,prec)+dy*(iy*iy)+dx*(ix*ix);
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

void scaneigh(ftype *img,int imgy,int imgx,ftype *mask,int masky,int maskx,int dimz,int *posy,int *posx,ftype *val,int *posey,int *posex,int rady, int radx,int len)
{   
    //return;
    int i;
    for (i=0;i<len;i++)
    {
        //if (posy[i]==-1  && posx[i]==-1)
        //    val[i]=0.0;
        //else
        val[i]=refineigh(img,imgy,imgx,mask,masky,maskx,dimz,posy[i],posx[i],rady,radx,posey++,posex++,NULL);       
    }
}

void scaneighpr(ftype *img,int imgy,int imgx,ftype *mask,int masky,int maskx,int dimz,int *posy,int *posx,ftype *val,int *posey,int *posex,int rady, int radx,int len,ftype *pr)
{   
    //return;
    int i;
    for (i=0;i<len;i++)
    {
        //if (posy[i]==-1  && posx[i]==-1)
        //    val[i]=0.0;
        //else
        val[i]=refineighpr(img,imgy,imgx,mask,masky,maskx,dimz,posy[i],posx[i],rady,radx,posey++,posex++,NULL,pr);       
    }
}


/*void scanRCFL(ftype *img,int *imgy,int *imgx,int imglen,ftype *mask,int *masky,int *maskx,int dimz, int masklen,ftype *score,int *defy,int *defx,int initr, int radius)
{
    ftype *mimg=img;
    int scl;
    for (scl=0;scl<imglen;scl++)
    {
        mimg=mimg+(imgy[scl]*imgx[scl]*dimz);//point to the current HOG image
    }
}*/

#define maxrad 5
static ftype scr[4*(2*maxrad+1)*(2*maxrad+1)];

//extern ftype minimize(ftype *D,ftype *pd1,ftype *pd2,ftype *pd3,ftype *pd4,int *result, int sizeX,int sizeY,int ry,int rx);

/*void scanDef(ftype *ww1,ftype *ww2,ftype *ww3,ftype *ww4,int fy,int fx,int dimz,ftype *df1,ftype *df2,ftype *df3,ftype *df4,ftype *img,int imgy,int imgx,int *posy,int *posx,int *parts,ftype *res,int rad,int len,int usemrf,ftype *uscr,int useusrc,ftype *prec,int pady,int padx)
{
    int sizedf=4;
    ftype maxscr1,maxscr2,maxscr3,maxscr4;
    ftype aux;
    int i,c,p,rdef[4],dy[4],dx[4];
    //if (!usemrf)
    //    printf("Warning:Not using MRF!!");
    for (i=0;i<len;i++)
    {
        if (useusrc)
        { 
            for (p=0;p<4;p++)
                for (c=0;c<(2*rad+1)*(2*rad+1);c++)
                    scr[p*(2*rad+1)*(2*rad+1)+c]=-uscr[p*len*(2*rad+1)*(2*rad+1)+c*len+i];
        }
        else
        {
            for (c=0;c<4*(2*rad+1)*(2*rad+1);c++)
                scr[c]=0;
        }          
        ftype *prec1=NULL,*prec2=NULL,*prec3=NULL,*prec4=NULL;
        //printf("Initial velue of prec:%d\n",prec==NULL);
        //printf("USE src:%d\n",useusrc);
        if (prec!=NULL)//use buffer
        {
            prec1=prec;
            prec2=prec+(imgy+2*pady)*(imgx+2*padx);
            prec3=prec+2*(imgy+2*pady)*(imgx+2*padx);
            prec4=prec+3*(imgy+2*pady)*(imgx+2*padx);
        }
        maxscr1 = refineighfull(img,imgy,imgx,ww1,fy,fx,dimz,df1[0],df1[1],posy[i],posx[i],rad,rad,scr,dy,dx,prec1,pady,padx);
        maxscr2 = refineighfull(img,imgy,imgx,ww2,fy,fx,dimz,df2[0],df2[1],posy[i],posx[i]+fx,rad,rad,scr+(2*rad+1)*(2*rad+1),dy+1,dx+1,prec2,pady,padx);
        maxscr3 = refineighfull(img,imgy,imgx,ww3,fy,fx,dimz,df3[0],df3[1],posy[i]+fy,posx[i],rad,rad,scr+2*(2*rad+1)*(2*rad+1),dy+2,dx+2,prec3,pady,padx);
        maxscr4 = refineighfull(img,imgy,imgx,ww4,fy,fx,dimz,df4[0],df4[1],posy[i]+fy,posx[i]+fx,rad,rad,scr+3*(2*rad+1)*(2*rad+1),dy+3,dx+3,prec4,pady,padx);
        //printf("Part LOCAL:%d %d %d %d \n",dy[0],dy[1],dy[2],dy[3]);
        if (usemrf)
        {
            res[i] = -minimize(scr,df1+2,df2+2,df3+2,df4+2,rdef,2,2,(2*rad+1),(2*rad+1));        
            for (c=0;c<4;c++)
            {
                dy[c]=rdef[c]/(2*rad+1)-rad;
                dx[c]=rdef[c]%(2*rad+1)-rad;
            }
        }
        else
            res[i]=maxscr1+maxscr2+maxscr3+maxscr4;
        //printf("Part MRF:%d %d %d %d \n",dy[0],dy[1],dy[2],dy[3]);
        //printf("Scr: MRF %.3f, LOCAL %.3f\n",res[i],maxscr1+maxscr2+maxscr3+maxscr4);
        //mydef is referered to the next part in clockwise way
        parts[0*len*sizedf+0*len+i]=dy[0]; //felz dy
        parts[0*len*sizedf+1*len+i]=dx[0]; //felz dx
        parts[0*len*sizedf+2*len+i]=(dy[0]-dy[1])*(dy[0]-dy[1]); //my defy
        parts[0*len*sizedf+3*len+i]=(dx[0]-dx[1])*(dx[0]-dx[1]); //my defx
        parts[1*len*sizedf+0*len+i]=dy[1]; //felz dy
        parts[1*len*sizedf+1*len+i]=dx[1]; //felz dx
        parts[1*len*sizedf+2*len+i]=(dy[1]-dy[3])*(dy[1]-dy[3]); //my defy
        parts[1*len*sizedf+3*len+i]=(dx[1]-dx[3])*(dx[1]-dx[3]); //my defx
        parts[2*len*sizedf+0*len+i]=dy[2]; //felz dy
        parts[2*len*sizedf+1*len+i]=dx[2]; //felz dx
        parts[2*len*sizedf+2*len+i]=(dy[2]-dy[0])*(dy[2]-dy[0]); //my defy
        parts[2*len*sizedf+3*len+i]=(dx[2]-dx[0])*(dx[2]-dx[0]); //my defx
        parts[3*len*sizedf+0*len+i]=dy[3]; //felz dy
        parts[3*len*sizedf+1*len+i]=dx[3]; //felz dx
        parts[3*len*sizedf+2*len+i]=(dy[3]-dy[2])*(dy[3]-dy[2]); //my defy
        parts[3*len*sizedf+3*len+i]=(dx[3]-dx[2])*(dx[3]-dx[2]); //my defx
       
    }
}*/


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
            //def[l2*maxl+l1]=dfy*(dy2-dy1)*(dy2-dy1)+dfx*(dx2-dx1)*(dx2-dx1);
            //printf("DEF(%d,%d)=%f\n",l1,l2,def[l1*maxl+l2]);
        }
}

static ftype scr2[4*(2*maxrad+1)*(2*maxrad+1)];

void scanDef2(ftype *ww1,ftype *ww2,ftype *ww3,ftype *ww4,int fy,int fx,int dimz,ftype *df1,ftype *df2,ftype *df3,ftype *df4,ftype *img,int imgy,int imgx,int *posy,int *posx,int *parts,ftype *res,int rad,int len,int usemrf,ftype *uscr,int useusrc,ftype *prec,int pady,int padx)
{
    int sizedf=4;
    ftype maxscr1,maxscr2,maxscr3,maxscr4;
    ftype aux,aux0,aux1,aux2,aux3,myaux;
    int i,c,p,rdef[4],rdef1[4],dy[4],dx[4];
    ftype maxm[(2*maxrad+1)*(2*maxrad+1)];
    buildef(def,df3[2],df3[3],rad);
    buildef(def+(2*rad+1)*(2*rad+1)*(2*rad+1)*(2*rad+1),df1[2],df1[3],rad);
    buildef(def+2*(2*rad+1)*(2*rad+1)*(2*rad+1)*(2*rad+1),df2[2],df2[3],rad);
    buildef(def+3*(2*rad+1)*(2*rad+1)*(2*rad+1)*(2*rad+1),df4[2],df4[3],rad);  
    for (i=0;i<len;i++)
    {
        if (posy[i]<-100) 
        {
            //printf("Skip for thr!\n");
            continue;
        }
        int aord[]={0,1,3,2};
        if (useusrc)
        { 
            for (p=0;p<4;p++)
                for (c=0;c<(2*rad+1)*(2*rad+1);c++)
                    scr[aord[p]*(2*rad+1)*(2*rad+1)+c]=-uscr[p*len*(2*rad+1)*(2*rad+1)+c*len+i];
        }
        else
        {
            for (c=0;c<4*(2*rad+1)*(2*rad+1);c++)
                scr[c]=0;
        }          
        ftype *prec1=NULL,*prec2=NULL,*prec3=NULL,*prec4=NULL;
        //printf("Initial velue of prec:%d\n",prec==NULL);
        //printf("USE src:%d\n",useusrc);
        if (prec!=NULL)//use buffer
        {
            prec1=prec;
            prec2=prec+(imgy+2*pady)*(imgx+2*padx);
            prec3=prec+2*(imgy+2*pady)*(imgx+2*padx);
            prec4=prec+3*(imgy+2*pady)*(imgx+2*padx);
        }
        maxscr1 = refineighfull(img,imgy,imgx,ww1,fy,fx,dimz,df1[0],df1[1],posy[i],posx[i],rad,rad,scr,dy,dx,prec1,pady,padx);
        maxscr2 = refineighfull(img,imgy,imgx,ww2,fy,fx,dimz,df2[0],df2[1],posy[i],posx[i]+fx,rad,rad,scr+(2*rad+1)*(2*rad+1),dy+1,dx+1,prec2,pady,padx);
        maxscr3 = refineighfull(img,imgy,imgx,ww3,fy,fx,dimz,df3[0],df3[1],posy[i]+fy,posx[i],rad,rad,scr+3*(2*rad+1)*(2*rad+1),dy+2,dx+2,prec3,pady,padx);
        maxscr4 = refineighfull(img,imgy,imgx,ww4,fy,fx,dimz,df4[0],df4[1],posy[i]+fy,posx[i]+fx,rad,rad,scr+2*(2*rad+1)*(2*rad+1),dy+3,dx+3,prec4,pady,padx);
        if (usemrf)
        {
            for (c=0;c<4*(2*rad+1)*(2*rad+1);c++)
                scr[c]=-scr[c];//invert the score to keep compatibility
            filltables(scr,def,4,(2*rad+1)*(2*rad+1));
            aux0=map(rdef,0,maxm);  
            /*aux1=map(rdef,1,maxm);  
            aux2=map(rdef,2,maxm);  
            aux3=map(rdef,3,maxm);  
            /*if ((aux0-aux1>0.001 || aux1-aux0>0.001))
                printf("Error aux0(%.3f) different than aux1(%.3f)\n",aux0,aux1);
            if ((aux1-aux2>0.001 || aux1-aux2>0.001))
                printf("Error aux1(%.3f) different than aux2(%.3f)\n",aux1,aux2);
            if ((aux2-aux3>0.001 || aux3-aux2>0.001))
                printf("Error aux2(%.3f) different than aux3(%.3f)\n",aux2,aux3);*/
            //myaux = -minimize(scr,df1+2,df2+2,df3+2,df4+2,rdef,2,2,(2*rad+1),(2*rad+1));  
            //to remove   
            /*for (c=0;c<4*(2*rad+1)*(2*rad+1);c++)
                scr[c]=0;   
            maxscr1 = refineighfull(img,imgy,imgx,ww1,fy,fx,dimz,df1[0],df1[1],posy[i],posx[i],rad,rad,scr,dy,dx,prec1,pady,padx);
            maxscr2 = refineighfull(img,imgy,imgx,ww2,fy,fx,dimz,df2[0],df2[1],posy[i],posx[i]+fx,rad,rad,scr+(2*rad+1)*(2*rad+1),dy+1,dx+1,prec2,pady,padx);
            maxscr3 = refineighfull(img,imgy,imgx,ww3,fy,fx,dimz,df3[0],df3[1],posy[i]+fy,posx[i],rad,rad,scr+2*(2*rad+1)*(2*rad+1),dy+2,dx+2,prec3,pady,padx);
            maxscr4 = refineighfull(img,imgy,imgx,ww4,fy,fx,dimz,df4[0],df4[1],posy[i]+fy,posx[i]+fx,rad,rad,scr+3*(2*rad+1)*(2*rad+1),dy+3,dx+3,prec4,pady,padx);
            myaux = -minimize(scr,df1+2,df2+2,df3+2,df4+2,rdef,2,2,(2*rad+1),(2*rad+1));        */
            //to remove
            res[i]=aux0;
            //if (abs(myaux-res[i])>0.0001)
            //    printf("Different dp:%f mrf:%f\n",res[i],myaux);
            //else
            //    printf("Equal\n");
            /*if (res[i]!=0)
            {
                //printf("RES:%.3f\n",res[i]);
                //printf("MAXscr:%.3f\n",maxscr1+maxscr2+maxscr3+maxscr4);  
                //for (c=0;c<4*(2*rad+1)*(2*rad+1);c++)
                //   printf("Scr:%.3f\n",scr[c]);
            }*/
            for (c=0;c<4;c++)
            {
                //if (res[i]!=0) printf("Dy:%d Path:%d\n",dy[c],rdef[c]);
                dy[c]=(rdef[c]/(2*rad+1))-rad;
                //if (res[i]!=0) printf("Dy:%d Path:%d\n",(rdef[c]/(2*rad+1))-rad,rdef[c]);
                //if (res[i]!=0) printf("Dx:%d\n",dx[c]);
                dx[c]=(rdef[c]%(2*rad+1))-rad;
                //if (res[i]!=0) printf("Dx:%d\n",(rdef[c]%(2*rad+1))-rad);
            }
            //printf("Scores:%.3f,%.3f,%.3f\n",aux1,res[i],maxscr1+maxscr2+maxscr3+maxscr4);      
            /*if ((aux1-res[i])>0.001)
                printf("MRF is better than Dyn:%.3f,%.3f\n",aux1,res[i]);
            if ((res[i]-aux1)>0.001)
                printf("Dyn is better than MRF:%.3f,%.3f\n",aux1,res[i]);*/
            //switch defs
            aux=dy[2];dy[2]=dy[3];dy[3]=aux;
            aux=dx[2];dx[2]=dx[3];dx[3]=aux;
            //res[i]=aux1;
        }
        else
            res[i]=maxscr1+maxscr2+maxscr3+maxscr4;
        
        //mydef is referered to the next part in clockwise way
        parts[0*len*sizedf+0*len+i]=dy[0]; //felz dy
        parts[0*len*sizedf+1*len+i]=dx[0]; //felz dx
        parts[0*len*sizedf+2*len+i]=(dy[0]-dy[1])*(dy[0]-dy[1]); //my defy
        parts[0*len*sizedf+3*len+i]=(dx[0]-dx[1])*(dx[0]-dx[1]); //my defx
        parts[1*len*sizedf+0*len+i]=dy[1]; //felz dy
        parts[1*len*sizedf+1*len+i]=dx[1]; //felz dx
        parts[1*len*sizedf+2*len+i]=(dy[1]-dy[3])*(dy[1]-dy[3]); //my defy
        parts[1*len*sizedf+3*len+i]=(dx[1]-dx[3])*(dx[1]-dx[3]); //my defx
        parts[2*len*sizedf+0*len+i]=dy[2]; //felz dy
        parts[2*len*sizedf+1*len+i]=dx[2]; //felz dx
        parts[2*len*sizedf+2*len+i]=(dy[2]-dy[0])*(dy[2]-dy[0]); //my defy
        parts[2*len*sizedf+3*len+i]=(dx[2]-dx[0])*(dx[2]-dx[0]); //my defx
        parts[3*len*sizedf+0*len+i]=dy[3]; //felz dy
        parts[3*len*sizedf+1*len+i]=dx[3]; //felz dx
        parts[3*len*sizedf+2*len+i]=(dy[3]-dy[2])*(dy[3]-dy[2]); //my defy
        parts[3*len*sizedf+3*len+i]=(dx[3]-dx[2])*(dx[3]-dx[2]); //my defx
       
    }
}















