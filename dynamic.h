#include <math.h> //for the definition of INFINITY

#ifndef INFINITY
   #define INFINITY 10000
#endif

//maximum number of labels and nodes
#define maxlab 100
#define maxnodes 10
//type used for data
#define ftype float

static ftype scr1[maxnodes*maxlab]; //score for each location
static ftype sumscr[maxnodes*maxlab]; //sum score for each location
static int auxpos[maxnodes*maxlab]; //position for each location
static ftype def[maxnodes*maxlab*maxlab]; //cost of deformation for each location if it is a "good" function distance transform can be used
//static float maxmarg[maxlab]; //max marginals
static int numlab; //number of labels
static int maxnode; //number of nodes

//void compute(int node,int *pos);
void initmaxmarginal(int node,int p);//initialize for computing max marginals in node,pos
ftype maxmarginal(int node,int p,int *pos);//compute the max marginals in node,pos
ftype map(int *mapos,int node,ftype *maxmarg);//compute map and avrg using max marginals
void filltables(ftype *pscr,ftype *pdef,int pnodes,int pnumlab);//fill scr,pos,def



