// example.cpp -- illustrates calling the MRF code

static char *usage = "usage: %s [energyType] (a number between 0 and 3)\n";

// uncomment "#define COUNT_TRUNCATIONS" in energy.h to enable counting of truncations

#include "mrf.h"
#include "ICM.h"
#include "GCoptimization.h"
#include "MaxProdBP.h"
#include "TRW-S.h"
#include "BP-S.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <new>

const int sizeX = 2;
const int sizeY = 2;
const int numLabels = 9;

//MRF::CostVal D[sizeX*sizeY*numLabels];
//MRF::CostVal V[numLabels*numLabels];
//MRF::CostVal hCue[sizeX*sizeY];
//MRF::CostVal vCue[sizeX*sizeY];

#ifdef COUNT_TRUNCATIONS
int truncCnt, totalCnt;
#endif

/*
EnergyFunction* generate_DataARRAY_SmoothFIXED_FUNCTION(double *D, int sizeX, int sizeY, int numLabels)
{
    int i, j,cnt;

    // generate function
    for (i=0; i<numLabels; i++) {
	for (j=i; j<numLabels; j++) {
	    V[i*numLabels+j] = V[j*numLabels+i] = (i == j) ? 0 : (MRF::CostVal)2.3;
	}
    }
    MRF::CostVal* ptr;
    cnt=0;
    //for (ptr=&D[0]; ptr<&D[sizeX*sizeY*numLabels]; ptr++) *ptr = (MRF::CostVal)data[cnt++];//(rand() % 100))/10 + 1;
    for (ptr=&hCue[0]; ptr<&hCue[sizeX*sizeY]; ptr++) *ptr = rand() % 3 + 1;
    for (ptr=&vCue[0]; ptr<&vCue[sizeX*sizeY]; ptr++) *ptr = rand() % 3 + 1;

    // allocate energy
    DataCost *data         = new DataCost(D);
    SmoothnessCost *smooth = new SmoothnessCost(V,hCue,vCue);
    EnergyFunction *energy    = new EnergyFunction(data,smooth);

    return energy;
}

EnergyFunction* generate_DataARRAY_SmoothTRUNCATED_LINEAR(double *D, int sizeX, int sizeY, int numLabels)
{
    // generate function
    MRF::CostVal* ptr;
    for (ptr=&D[0]; ptr<&D[sizeX*sizeY*numLabels]; ptr++) *ptr = ((MRF::CostVal)(rand() % 100))/10 + 1;
    for (ptr=&hCue[0]; ptr<&hCue[sizeX*sizeY]; ptr++) *ptr = rand() % 3;
    for (ptr=&vCue[0]; ptr<&vCue[sizeX*sizeY]; ptr++) *ptr = rand() % 3;
    MRF::CostVal smoothMax = (MRF::CostVal)25.5, lambda = (MRF::CostVal)2.7;

    // allocate energy
    DataCost *data         = new DataCost(D);
    SmoothnessCost *smooth = new SmoothnessCost(1,smoothMax,lambda,hCue,vCue);
    EnergyFunction *energy    = new EnergyFunction(data,smooth);

    return energy;
}

EnergyFunction* generate_DataARRAY_SmoothTRUNCATED_QUADRATIC(double *D, int sizeX, int sizeY, int numLabels)
{
    
    // generate function
    MRF::CostVal* ptr;
    for (ptr=&D[0]; ptr<&D[sizeX*sizeY*numLabels]; ptr++) *ptr = ((MRF::CostVal)(rand() % 100))/10 + 1;
    for (ptr=&hCue[0]; ptr<&hCue[sizeX*sizeY]; ptr++) *ptr = rand() % 3;
    for (ptr=&vCue[0]; ptr<&vCue[sizeX*sizeY]; ptr++) *ptr = rand() % 3;
    MRF::CostVal smoothMax = (MRF::CostVal)5.5, lambda = (MRF::CostVal)2.7;

    // allocate energy
    DataCost *data         = new DataCost(D);
    SmoothnessCost *smooth = new SmoothnessCost(2,smoothMax,lambda,hCue,vCue);
    EnergyFunction *energy    = new EnergyFunction(data,smooth);

    return energy;
}


MRF::CostVal dCost(int pix, int i)
{
    return ((pix*i + i + pix) % 30) / ((MRF::CostVal) 3);
}*/


//neigh=numpy.array([[-1,-1],[0,-1],[1,-1],[-1,0],[0,0],[1,0],[-1,1],[0,1],[1,1]])
int ny[]={0,-1,0,1,-1,1,-1,0,1};
int nx[]={0,-1,-1,-1,0,0,1,1,1};
double mhCue[sizeX*sizeY];//maximum size
double mvCue[sizeX*sizeY];
double mhCue2[sizeX*sizeY];//maximum size
double mvCue2[sizeX*sizeY];

double k=0;

MRF::CostVal fnCost(int pix1, int pix2, int i, int j)
{
    if (pix2 < pix1) { // ensure that fnCost(pix1, pix2, i, j) == fnCost(pix2, pix1, j, i)
	int tmp;
	tmp = pix1; pix1 = pix2; pix2 = tmp; 
	tmp = i; i = j; j = tmp;
    }
    //MRF::CostVal answer = k*(mhCue[pix1]*(nx[i]-nx[j])*(nx[i]-nx[j])+mvCue[pix1]*(ny[i]-ny[j])*(ny[i]-ny[j]));
    MRF::CostVal answer;
    //printf("pixels: %d %d %d %d\n",pix1,pix2,i,j);
    //printf("Cues: %f %f %f %f\n",mhCue[pix1],mvCue[pix1],mhCue2[pix1],mvCue2[pix1]);
    if (pix2-pix1==1)//horizontal edge
        answer = k*(mhCue[pix1]*(nx[i]-nx[j])*(nx[i]-nx[j])+mvCue[pix1]*(ny[i]-ny[j])*(ny[i]-ny[j]));
    else //vertical edge
        answer = k*(mhCue2[pix1]*(nx[i]-nx[j])*(nx[i]-nx[j])+mvCue2[pix1]*(ny[i]-ny[j])*(ny[i]-ny[j]));
    //MRF::CostVal answer = k*(mhCue[pix1]*abs(nx[i]-nx[j])+mvCue[pix1]*abs(ny[i]-ny[j]));
    //MRF::CostVal answer = (pix1*(i+1)*(j+2) + pix2*i*j*pix1 - 2*i*j*pix1) % 100;
    return answer;
}

/*MRF::CostVal fnCost(int pix1, int pix2, int i, int j)
{
    if (pix2 < pix1) { // ensure that fnCost(pix1, pix2, i, j) == fnCost(pix2, pix1, j, i)
	int tmp;
	tmp = pix1; pix1 = pix2; pix2 = tmp; 
	tmp = i; i = j; j = tmp;
    }
    MRF::CostVal answer = (pix1*(i+1)*(j+2) + pix2*i*j*pix1 - 2*i*j*pix1) % 100;
    return answer / 10;
}


EnergyFunction* generate_DataFUNCTION_SmoothGENERAL_FUNCTION(double *D, int sizeX, int sizeY, int numLabels)
{
    DataCost *data         = new DataCost(dCost);
    SmoothnessCost *smooth = new SmoothnessCost(fnCost);
    EnergyFunction *energy = new EnergyFunction(data,smooth);

    return energy;
}*/

extern "C" {
double mymrf(double *D, double *V, double *hCue, double *vCue,double *hCue2, double *vCue2,int *result, int sizeX,int sizeY,int numLabels, double valk)
{
    //int argc=1;
    //char **argv
    //MRF::CostVal D[sizeX*sizeY*numLabels];
    //MRF::CostVal V[numLabels*numLabels];
    //MRF::CostVal hCue[sizeX*sizeY];
    //MRF::CostVal vCue[sizeX*sizeY];

    MRF* mrf;
    //EnergyFunction *energy;
    MRF::EnergyVal E;
    double lowerBound;
    float t,tot_t;
    int iter;

    int seed = 1124285485;
    srand(seed);
    //k=valk;
    k=1.0;
    //printf("k=%g *************\n",k);
    for (int i=0;i<sizeX*sizeY;i++)
    {
        //printf("h1=%g v1=%g h2=%g v2=%g \n",mhCue[i],mvCue[i],mhCue2[i],mvCue2[i]);
        mhCue[i]=hCue[i];
        mvCue[i]=vCue[i];
        mhCue2[i]=hCue2[i];
        mvCue2[i]=vCue2[i];
    }
    int Etype = 0;
    MRF::CostVal* ptr;
    MRF::Label *res,*aux,*aux1;
    //for (ptr=&D[0]; ptr<&D[sizeX*sizeY*numLabels]; ptr++) printf("%f ",*ptr);//(rand() % 100))/10 + 1;
    //printf("%d %d %d",sizeX,sizeY,numLabels);
    //if (argc != 2) {
	//fprintf(stderr, usage, argv[0]);
	//exit(1);
    //}
    
    //if (argc > 1)
	//Etype = atoi(argv[1]);

/*    try {
	switch(Etype) {
	    // Here are 4 sample energies to play with.
	case 0:
	    energy = generate_DataARRAY_SmoothFIXED_FUNCTION(data,sizeX,sizeY,numLabels);
	    fprintf(stderr, "using fixed (array) smoothness cost\n");
	    break;
	case 1:
	    energy = generate_DataARRAY_SmoothTRUNCATED_LINEAR(data,sizeX,sizeY,numLabels);
	    fprintf(stderr, "using truncated linear smoothness cost\n");
	    break;
	case 2:
	    energy = generate_DataARRAY_SmoothTRUNCATED_QUADRATIC(data,sizeX,sizeY,numLabels);
	    fprintf(stderr, "using truncated quadratic smoothness cost\n");
	    break;
	case 3:
	    energy = generate_DataFUNCTION_SmoothGENERAL_FUNCTION(data,sizeX,sizeY,numLabels);
	    fprintf(stderr, "using general smoothness functions\n");
	    break;
	default:
	    //fprintf(stderr, usage, argv[0]);
	    exit(1);
	}*/
    DataCost *data         = new DataCost(D);
    SmoothnessCost *smooth = new SmoothnessCost(fnCost);
    EnergyFunction *energy    = new EnergyFunction(data,smooth);

	bool runICM       = false;
	bool runExpansion = false;
	bool runSwap      = false;
	bool runMaxProdBP = false;
	bool runTRWS      = true;
	bool runBPS       = false;


	////////////////////////////////////////////////
	//                     ICM                    //
	////////////////////////////////////////////////
	if (runICM) {
	    //printf("\n*******Started ICM *****\n");

	    mrf = new ICM(sizeX,sizeY,numLabels,energy);
	    mrf->initialize();
	    mrf->clearAnswer();

	    E = mrf->totalEnergy();
	    //printf("Energy at the Start= %g (%g,%g)\n", (float)E,(float)mrf->smoothnessEnergy(), (float)mrf->dataEnergy());

	    tot_t = 0;
	    for (iter=0; iter<6; iter++) {
		    mrf->optimize(10, t);
		
		    E = mrf->totalEnergy();
		    tot_t = tot_t + t ;
		    //printf("energy = %g (%f secs)\n", (float)E, tot_t);
            res=mrf->getAnswerPtr();
            aux1=result;
            for (aux=res;aux<&res[sizeX*sizeY];aux++) 
            {
                *(aux1++)=*aux;
                printf("%d ",*aux);
            }
	    }
        //printf("%g ",(double)E);
	    delete mrf;
	}
    //return 0;

	////////////////////////////////////////////////
	//          Graph-cuts expansion              //
	////////////////////////////////////////////////
/*	if (runExpansion) {
	    printf("\n*******Started graph-cuts expansion *****\n");
	    mrf = new Expansion(sizeX,sizeY,numLabels,energy);
	    mrf->initialize();
	    mrf->clearAnswer();
	    
	    E = mrf->totalEnergy();
	    printf("Energy at the Start= %g (%g,%g)\n", (float)E,
		   (float)mrf->smoothnessEnergy(), (float)mrf->dataEnergy());

#ifdef COUNT_TRUNCATIONS
	    truncCnt = totalCnt = 0;
#endif
	    tot_t = 0;
	    for (iter=0; iter<6; iter++) {
		mrf->optimize(1, t);

		E = mrf->totalEnergy();
		tot_t = tot_t + t ;
		printf("energy = %g (%f secs)\n", (float)E, tot_t);
	    }
#ifdef COUNT_TRUNCATIONS
	    if (truncCnt > 0)
		printf("***WARNING: %d terms (%.2f%%) were truncated to ensure regularity\n", 
		       truncCnt, (float)(100.0 * truncCnt / totalCnt));
#endif

	    delete mrf;
	}

	////////////////////////////////////////////////
	//          Graph-cuts swap                   //
	////////////////////////////////////////////////
	if (runSwap) {
	    printf("\n*******Started graph-cuts swap *****\n");
	    mrf = new Swap(sizeX,sizeY,numLabels,energy);
	    mrf->initialize();
	    mrf->clearAnswer();
	    
	    E = mrf->totalEnergy();
	    printf("Energy at the Start= %g (%g,%g)\n", (float)E,
		   (float)mrf->smoothnessEnergy(), (float)mrf->dataEnergy());

#ifdef COUNT_TRUNCATIONS
	    truncCnt = totalCnt = 0;
#endif
	    tot_t = 0;
	    for (iter=0; iter<8; iter++) {
		mrf->optimize(1, t);

		E = mrf->totalEnergy();
		tot_t = tot_t + t ;
		printf("energy = %g (%f secs)\n", (float)E, tot_t);
	    }
#ifdef COUNT_TRUNCATIONS
	    if (truncCnt > 0)
		printf("***WARNING: %d terms (%.2f%%) were truncated to ensure regularity\n", 
		       truncCnt, (float)(100.0 * truncCnt / totalCnt));
#endif

   
	    delete mrf;
	}*/

	////////////////////////////////////////////////
	//          Belief Propagation                //
	////////////////////////////////////////////////
	if (runMaxProdBP) {
	    //printf("\n*******  Started MaxProd Belief Propagation *****\n");
	    mrf = new MaxProdBP(sizeX,sizeY,numLabels,energy);
	    mrf->initialize();
	    mrf->clearAnswer();
	    
	    E = mrf->totalEnergy();
	    //printf("Energy at the Start= %g (%g,%g)\n", (float)E,(float)mrf->smoothnessEnergy(), (float)mrf->dataEnergy());

	    tot_t = 0;
	    for (iter=0; iter < 10; iter++) {
		mrf->optimize(1, t);

		E = mrf->totalEnergy();
		tot_t = tot_t + t ;
		//printf("energy = %g (%f secs)\n", (float)E, tot_t);
	    }

	    
	    delete mrf;
	}

	////////////////////////////////////////////////
	//                  TRW-S                     //
	////////////////////////////////////////////////
	if (runTRWS) {
	    //printf("\n*******Started TRW-S *****\n");
	    mrf = new TRWS(sizeX,sizeY,numLabels,energy);

	    // can disable caching of values of general smoothness function:
	    //mrf->dontCacheSmoothnessCosts();

	    mrf->initialize();
	    mrf->clearAnswer();

	    
	    E = mrf->totalEnergy();
	    //printf("Energy at the Start= %g (%g,%g)\n", (float)E,(float)mrf->smoothnessEnergy(), (float)mrf->dataEnergy());

	    tot_t = 0;
	    for (iter=0; iter<1; iter++) {
		mrf->optimize(10, t);

		E = mrf->totalEnergy();
		lowerBound = mrf->lowerBound();
		tot_t = tot_t + t ;
        res=mrf->getAnswerPtr();
        aux1=result;
        for (aux=res;aux<&res[sizeX*sizeY];aux++) 
        {
            *(aux1++)=*aux;
            //printf("%d ",*aux);
        }
		//printf("energy = %g, lower bound = %f (%f secs)\n", (float)E, lowerBound, tot_t);
	    }

	    delete mrf;
	}

	////////////////////////////////////////////////
	//                  BP-S                     //
	////////////////////////////////////////////////
	if (runBPS) {
	    //printf("\n*******Started BP-S *****\n");
	    mrf = new BPS(sizeX,sizeY,numLabels,energy);

	    // can disable caching of values of general smoothness function:
	    //mrf->dontCacheSmoothnessCosts();
		
	    mrf->initialize();
	    mrf->clearAnswer();
	    
	    E = mrf->totalEnergy();
	    //printf("Energy at the Start= %g (%g,%g)\n", (float)E,(float)mrf->smoothnessEnergy(), (float)mrf->dataEnergy());

	    tot_t = 0;
	    for (iter=0; iter<10; iter++) {
		mrf->optimize(10, t);

		E = mrf->totalEnergy();
		tot_t = tot_t + t ;
		//printf("energy = %g (%f secs)\n", (float)E, tot_t);
	    }

	    delete mrf;
	}
 //   }
    /*catch (std::bad_alloc) {
	fprintf(stderr, "*** Error: not enough memory\n");
	exit(1);
    }*/
    //printf("%g ",E);
    return E;
}
}
