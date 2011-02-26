#include <math.h> //for the definition of INFINITY
#include "dynamic.h"
//#ifndef INFINITY
//   #define INFINITY 10000
//#endif

//#define maxlab 100
//#define maxnodes 10
//#define ftype float

//static ftype scr[maxnodes*maxlab]; //score for each location
//static ftype sumscr[maxnodes*maxlab]; //sum score for each location
//static int auxpos[maxnodes*maxlab]; //position for each location
//static ftype def[maxnodes*maxlab*maxlab]; //cost of deformation for each location if it is a "good" function distance transform can be used
//static float maxmarg[maxlab]; //max marginals
//static int numlab;
//static int maxnode;

void compute(int node,int *pos)//here maxnode=4
{
    int i,id,idp,iddf,p;
    ftype aux;
    for (i=0;i<numlab;i++)
    {
        id=node*numlab+i;
        //sumscr[id]=scr[id];
        sumscr[id]=-INFINITY;
        for (p=0;p<numlab;p++)
        {   
            idp=(node-1+maxnode)%maxnode*numlab+p;
            iddf=(node+maxnode)%maxnode*numlab*numlab+p*numlab+i;
            aux=scr1[id]+def[iddf]+sumscr[idp];
            if (aux>sumscr[id])
            {
                //printf("  Aux(%d)=%.2f\n",p,aux);
                sumscr[id]=aux;
                pos[id]=p;
            }
        }
        //printf("Scr(%d,%d)=%.2f\n",node,i,scr[id]);
        //printf("SumScr(%d,%d)=%.2f\n",node,i,sumscr[id]);
        //printf("Pos(%d,%d)=%d\n",node,i,pos[id]);
    }
    //printf("\n");
}

void initmaxmarginal(int node,int p)//initialize for computing max marginals in node,pos
{
    int i,id;
    for (i=0;i<numlab;i++)
    {
        id=node*numlab+i;
        if (i==p)
            sumscr[id]=scr1[id];
        else
            sumscr[id]=-INFINITY;
        //printf("Scr(%d,%d)=%.2f\n",node,i,scr[id]);
        //printf("SumScr(%d,%d)=%.2f\n",node,i,sumscr[id]);
    }
    //printf("\n");
}

ftype maxmarginal(int node,int p,int *pos)//compute the max marginals in node,pos
{
    int i,n,id,idn,iddf;
    ftype maxm=-INFINITY;
    initmaxmarginal(node,p);
    for (n=1;n<maxnode;n++)
    {
        compute((node+n+maxnode)%maxnode,pos);
    }
    //add last connection with the first one
    for (i=0;i<numlab;i++)
    {
        id=((node-1+maxnode)%maxnode)*numlab+i;
        idn=((node+maxnode)%maxnode)*numlab+p;
        iddf=((node+maxnode)%maxnode)*numlab*numlab+i*numlab+p;
        //printf("Before Sumscr(%d)=%f\n",i,sumscr[id]);
        sumscr[id]+=def[iddf];
        //printf("Sumscr(%d)=%f\n",i,sumscr[id]);
        if (sumscr[id]>maxm)
        {
            maxm=sumscr[id];
            pos[node*numlab+p]=i;
        }
    }  
    return maxm;
}

ftype map(int *mapos,int node,ftype *maxmarg)//compute map and avrg using max marginals
{
    int i,n;
    ftype best=-INFINITY,scr;
    if (node>=maxnode)
    {
        printf("Error, node >= %d\n",maxnode);
        return -1;
    }
    for (i=0;i<numlab;i++)
    {
        scr=maxmarginal(node,i,auxpos);
        maxmarg[i]=scr;
        //printf("Score in (%d,%d): %f \n",node,i,scr);
        if (scr>best)
        {
            best=scr;
            mapos[(node-1+maxnode)%maxnode]=auxpos[(node)%maxnode*numlab+i];
            for (n=1;n<maxnode;n++)
            {
                mapos[(node-1-n+maxnode)%maxnode]=auxpos[(node-n+maxnode)%maxnode*numlab+mapos[(node-n+maxnode)%maxnode]];
            }
        }
    }
    return best;
}

void filltables(ftype *pscr,ftype *pdef,int pnodes,int pnumlab)//fill scr,pos,def
{
    int i;
    ftype *sscr=scr1,*sdef=def;
    for (i=0;i<pnodes*pnumlab;i++)
        *sscr++=*pscr++;
    for (i=0;i<pnodes*pnumlab*pnumlab;i++)
        *sdef++=*pdef++;
    numlab=pnumlab;
    maxnode=pnodes;
}



