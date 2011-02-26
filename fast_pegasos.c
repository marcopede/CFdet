#include <stdio.h>
#include <stdlib.h>

//w is the weight vector to estimate with size wx
//ex is the example matrix of size exy x wx

// gcc -fPIC -c fast_pegasos.c -O3
//gcc -shared -Wl,-soname,libfast_pegasos.so -o libfast_pegasos.so  fast_pegasos.o

#define ftype float

static inline ftype add(ftype *a,ftype *b,int len)
{
    int c;
    for (c=0;c<len;c++)
    {
        a[c]=a[c]+b[c];
    }
}

static inline ftype mul(ftype *a,ftype b,int len)
{
    int c;
    for (c=0;c<len;c++)
    {
        a[c]=a[c]*b;
    }
}

static inline ftype addmul(ftype *a,ftype *b,ftype c,int len)
{
    int cn;
    for (cn=0;cn<len;cn++)
    {
        a[cn]=a[cn]+b[cn]*c;
    }
}

static inline ftype score(ftype *x,ftype *w,int len)
{
    int c;
    ftype scr=0;
    for (c=0;c<len;c++)
    {
        scr+=x[c]*w[c];
    }
    return scr;
}

void fast_pegasos(ftype *w,int wx,ftype *ex,int exy,ftype *label,ftype lambda,int iter,int part)
{
    srand48(3);
    int c,y,t,pex;
    ftype *x,n,scr,norm,val;
    printf("Parts:%d \n Lambda:%g\n",part,lambda);
    //#pragma omp parallel for
    for (c=0;c<iter;c++)
    {
        t=c+part*iter+1;
        pex=(int)(drand48()*(exy-0.5));
        x=ex+pex*wx;
        y=label[pex];
        n=1.0/(lambda*t);
        scr=score(x,w,wx);
        //printf("rnd:%d y=%d scr=%g eta=%g\n",pex,y,scr,n);
        mul(w,1-n*lambda,wx);
        if (scr*y<1.0)
        {
            //mul(x,y*n,wx)
            addmul(w,x,y*n,wx);            
        }
        //printf("W0:%g",w[0]);
        norm=sqrt(score(w,w,wx));
        val=1/(sqrt(lambda)*(norm+0.0001));
        if (val<1.0)
            mul(w,val,wx);
    }
    printf("N:%g t:%d\n",n,t);
}

void fast_pegasos_comp(ftype *w,int numcomp,int *compx,int *compy,ftype **ptrsamplescomp,int totsamples,int *label,int *comp,ftype lambda,int iter,int part)
{
    int wx=0,wxtot=0;
    srand48(3);
    int c,d,y,t,pex,pexcomp,totsz,sumszx[10],sumszy[10];//max 10 components
    ftype *x,n,scr,norm,val,ptrc;
    totsz=0;
    //printf("Num Comp %d \n",numcomp);
    //printf("Tot Samples %d \n",totsamples);
    sumszx[0]=0;
    sumszy[0]=0;
    for (c=0;c<numcomp;c++)
    {
        wxtot+=compx[c];
        totsz+=compy[c];
        //printf("Compx %d\n",compx[c]);
        //printf("Compy %d\n",compy[c]);
        sumszx[c+1]=wxtot;
        sumszy[c+1]=totsz;
        //printf("Sum x %d \n",sumszx[c]);
        //printf("Sum y %d \n",sumszy[c]);
    }
    //printf("Wx tot:%d \n",wxtot);
    //x=ptrsamplescomp[0];
    //printf("X0: %f %f %f %f %f %f %f \n",x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7]);
    //x=ptrsamplescomp[1];
    //printf("X1: %f %f %f %f %f %f %f \n",x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7]);
    //printf("Parts:%d \n Lambda:%g\n",part,lambda);
    for (c=0;c<iter;c++)
    {
        t=c+part*iter+1;
        pex=(int)(drand48()*(totsamples-0.5));
        //printf("S: %d\n",pex);
        wx=compx[comp[pex]];
        //printf("Cluster:%d\n",comp[pex]);
        //printf("Wx:%d\n",compx[comp[pex]]);
        //x=ptrsamplescomp[comp[pex]];
        //printf("Computed location %d,%d\n",pex,sumszy[comp[pex]]);
        x=ptrsamplescomp[comp[pex]]+(pex-sumszy[comp[pex]])*wx;
        //printf("X:%.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f \n",*(x),*(x+1),*(x+2),*(x+3),*(x+4),*(x+5),*(x+6),*(x+7),*(x+8),*(x+9));
        //x=ex+pex*wx;
        y=label[pex];
        //printf("Y %d ",y);
        n=1.0/(lambda*t);
        //printf("C %d ",comp[pex]);
        scr=score(x,w+sumszx[comp[pex]],wx);
        //printf("Computed Score %f\n",scr);
        //printf("rnd:%d y=%d scr=%g eta=%g\n",pex,y,scr,n);
        //only the component l2_max
        mul(w+sumszx[comp[pex]],1-n*lambda,wx);    
        //all the vector
        //mul(w,1-n*lambda,wxtot);
        if (scr*y<1.0)
        {
            //mul(x,y*n,wx)
            //printf("W comp %d\n",comp[pex]);
            addmul(w+sumszx[comp[pex]],x,y*n,wx);            
            //addmul(w,x,y*n,wx);            
        }
        //printf("Computed location %d,%d\n",comp[pex],sumszy[comp[pex]]);
        //printf("W:%.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f \n",*(w),*(w+1),*(w+2),*(w+3),*(w+4),*(w+5),*(w+6),*(w+7),*(w+8),*(w+9));
        //printf("W0:%g",w[0]);
        //norm=sqrt(score(w,w,wxtot));
        //val=1/(sqrt(lambda)*(norm+0.0001));
        //if (val<1.0)
        //    mul(w,val,wxtot);
    }
    printf("N:%g t:%d\n",n,t);
}

ftype objective_comp(ftype *w,int wx,ftype *ex, int exy,ftype *label,ftype lambda)
{
    int c,y;
    ftype val,err=0,errpos=0,errneg=0,norm;
    for (c=0;c<exy;c++)
    {
        y=label[c];
        val=score(w,ex+c*wx,wx);
        if (val*y<1)
        {
            err+=1-y*val;
            if (y>0)
                errpos+=err;
            else
                errneg+=err;
        }
    } 
    err=err/exy;
    errpos=errpos/exy;
    errneg=errneg/exy;
    norm=lambda/2.0*score(w,w,wx);
    printf("lambda/2*|w|**2=%g Loss=%g \n", norm, err);
    printf("Pos Loss=%g Neg Loss=%g \n", errpos, errneg);
    return norm+err;
}

void fast_pegasos_noproj(ftype *w,int wx,ftype *ex,int exy,ftype *label,ftype lambda,int iter,int part)
{
    srand48(3);
    int c,y,t,pex;
    ftype *x,n,scr,norm,val;
    printf("Parts:%d \n Lambda:%g\n",part,lambda);
    //#pragma omp parallel for
    for (c=0;c<iter;c++)
    {
        t=c+part*iter+1;
        pex=(int)(drand48()*(exy-0.5));
        x=ex+pex*wx;
        y=label[pex];
        n=1.0/(lambda*t);
        scr=score(x,w,wx);
        //printf("rnd:%d y=%d scr=%g eta=%g\n",pex,y,scr,n);
        mul(w,1-n*lambda,wx);
        if (scr*y<1.0)
        {
            //mul(x,y*n,wx)
            addmul(w,x,y*n,wx);            
        }
        //printf("W0:%g",w[0]);
        /*norm=sqrt(score(w,w,wx));
        val=1/(sqrt(lambda)*(norm+0.0001));
        if (val<1.0)
            mul(w,val,wx);*/
    }
    printf("N:%g t:%d\n",n,t);
}

ftype objective(ftype *w,int wx,ftype *ex, int exy,ftype *label,ftype lambda)
{
    int c,y;
    ftype val,err=0,errpos=0,errneg=0,norm;
    for (c=0;c<exy;c++)
    {
        y=label[c];
        val=score(w,ex+c*wx,wx);
        if (val*y<1)
        {
            err+=1-y*val;
            if (y>0)
                errpos+=err;
            else
                errneg+=err;
        }
    } 
    err=err/exy;
    errpos=errpos/exy;
    errneg=errneg/exy;
    norm=lambda/2.0*score(w,w,wx);
    printf("lambda/2*|w|**2=%g Loss=%g \n", norm, err);
    printf("Pos Loss=%g Neg Loss=%g \n", errpos, errneg);
    return norm+err;
}


