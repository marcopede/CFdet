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

#define dtype float

//const int sizeX = 50;
//const int sizeY = 50;
//const int numLabels = 20;

//MRF::CostVal D[sizeX*sizeY*numLabels];
//MRF::CostVal hCue[sizeX*sizeY];
//MRF::CostVal vCue[sizeX*sizeY];

#ifdef COUNT_TRUNCATIONS
int truncCnt, totalCnt;
#endif

dtype *st_cost;
int st_parts_y;
int st_parts_x;
dtype * st_cost_y;
dtype * st_cost_x;
int st_numlab;
int maxlab=500;
//dtype* V[maxlab*maxlab];

//struct ForDataFn{
//	int numLab;
//	dtype *data;
//};

//MRF::CostVal dataFn(int p, int l, void *data)
//{
//	ForDataFn *myData = (ForDataFn *) data;
//	int numLab = myData->numLab;
//	
//	return( myData->data[p*numLab+l] );
//}


/*EnergyFunction* generate_DataARRAY_SmoothFIXED_FUNCTION()
{
    int i, j;

    // generate function
    for (i=0; i<numLabels; i++) {
	for (j=i; j<numLabels; j++) {
	    V[i*numLabels+j] = V[j*numLabels+i] = (i == j) ? 0 : (MRF::CostVal)2.3;
	}
    }
    MRF::CostVal* ptr;
    for (ptr=&D[0]; ptr<&D[sizeX*sizeY*numLabels]; ptr++) *ptr = ((MRF::CostVal)(rand() % 100))/10 + 1;
    for (ptr=&hCue[0]; ptr<&hCue[sizeX*sizeY]; ptr++) *ptr = rand() % 3 + 1;
    for (ptr=&vCue[0]; ptr<&vCue[sizeX*sizeY]; ptr++) *ptr = rand() % 3 + 1;

    // allocate energy
    DataCost *data         = new DataCost(D);
    SmoothnessCost *smooth = new SmoothnessCost(V,hCue,vCue);
    EnergyFunction *energy    = new EnergyFunction(data,smooth);

    return energy;
}

EnergyFunction* generate_DataARRAY_SmoothTRUNCATED_LINEAR()
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

EnergyFunction* generate_DataARRAY_SmoothTRUNCATED_QUADRATIC()
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
}
*/
//MRF::CostVal smoothApp(int p1, int p2, int l1, int l2)
//{
//    if (p1>p2)
//    {
//        int tmp;
//        tmp=p2;p2=p1; p1=tmp;
//        tmp = l1; l1 = l2; l2 = tmp;
//    }
//    return st_cost[p1*st_nparts*st_numlab*st_numlab+p2*st_numlab*st_numlab+l2*st_numlab+l1];
//}
MRF::CostVal smoothApp(int p1, int p2, int l1, int l2)
{
    if (p1>p2)
    {
        int tmp;
        tmp=p2;p2=p1; p1=tmp;
        tmp = l1; l1 = l2; l2 = tmp;
    }
    int p1x=p1%st_parts_x;
    int p1y=p1/st_parts_x;
    int p2x=p2%st_parts_x;
    int p2y=p2/st_parts_x;
    int x1=l1%st_parts_x;
    int y1=l1/st_parts_x;
    int x2=l2%st_parts_x;    
    int y2=l2/st_parts_x;
    return st_cost_x[p1]*abs((p1x-x1)-(p2x-x2))+st_cost_y[p1]*abs((p1y-y1)-(p2y-y2)); 
    //st_cost[p1*st_nparts*st_numlab*st_numlab+p2*st_numlab*st_numlab+l2*st_numlab+l1];
}


MRF::CostVal smoothApp2(int p1, int p2, int l1, int l2)
{
    if (p1>p2)
    {
        int tmp;
        tmp=p2;p2=p1; p1=tmp;
        tmp = l1; l1 = l2; l2 = tmp;
    }
    int p1x=p1%st_parts_x;
    int p1y=p1/st_parts_x;
    int p2x=p2%st_parts_x;
    int p2y=p2/st_parts_x;
    int x1=l1%(st_parts_x*2-1);
    int y1=l1/(st_parts_x*2-1);
    int x2=l2%(st_parts_x*2-1);    
    int y2=l2/(st_parts_x*2-1);
    return st_cost_x[p1]*abs((p1x*2-x1)-(p2x*2-x2))+st_cost_y[p1]*abs((p1y*2-y1)-(p2y*2-y2)); 
}

MRF::CostVal smoothApp3(int p1, int p2, int l1, int l2)
{
    if (p1>p2)
    {
        int tmp;
        tmp=p2;p2=p1; p1=tmp;
        tmp = l1; l1 = l2; l2 = tmp;
    }
//    int p1x=p1%st_parts_x;
//    int p1y=p1/st_parts_x;
//    int p2x=p2%st_parts_x;
//    int p2y=p2/st_parts_x;
    int x1=l1%(st_parts_x*2-1)/2;
    int y1=l1/(st_parts_x*2-1)/2;
    int x2=l2%(st_parts_x*2-1)/2;    
    int y2=l2/(st_parts_x*2-1)/2;
    return st_cost_x[p1]*abs(x1-x2)+st_cost_y[p1]*abs(y1-y2); 
}

//MRF::CostVal smoothApp4(int p1, int p2, int l1, int l2)
//{
//    if (p1>p2)
//    {
//        int tmp;
//        tmp=p2;p2=p1; p1=tmp;
//        tmp = l1; l1 = l2; l2 = tmp;
//    }
//    return cost[p1*st_numlab*st_numlab+l1*st_numlab+l2];//st_cost_x[p1]*abs(x1-x2)+st_cost_y[p1]*abs(y1-y2); 
//}


/*
MRF::CostVal fnCost(int pix1, int pix2, int i, int j)
{
    if (pix2 < pix1) { // ensure that fnCost(pix1, pix2, i, j) == fnCost(pix2, pix1, j, i)
	int tmp;
	tmp = pix1; pix1 = pix2; pix2 = tmp; 
	tmp = i; i = j; j = tmp;
    }
    MRF::CostVal answer = (pix1*(i+1)*(j+2) + pix2*i*j*pix1 - 2*i*j*pix1) % 100;
    return answer / 10;
}


EnergyFunction* generate_DataFUNCTION_SmoothGENERAL_FUNCTION()
{
    DataCost *data         = new DataCost(dCost);
    SmoothnessCost *smooth = new SmoothnessCost(fnCost);
    EnergyFunction *energy = new EnergyFunction(data,smooth);

    return energy;
}*/

extern "C" {

//dtype* VV[maxlab*maxlab];

/*void fill_cache(dtype* cache,int m_height,int m_width,int m_nLabels,int num_parts_y,int num_parts_x,dtype *costs)
{
    dtype* m_V=cache,*ptr;
    int x,y,i,kj,ki;
    st_cost_x=costs;
    st_cost_y=costs+num_parts_x*num_parts_y;
    st_parts_x=num_parts_x;
    st_parts_y=num_parts_y;    
    for (ptr=m_V,i=0,y=0; y<m_height; y++)
	for (x=0; x<m_width; x++, i++)
	    {
		if (x < m_width-1)
		    {
			for (kj=0; kj<m_nLabels; kj++)
			    for (ki=0; ki<m_nLabels; ki++)
				{
				    *ptr++ = smoothApp3(i,i+1,ki,kj);
				}
		    }
		else ptr += m_nLabels*m_nLabels;

		if (y < m_height-1)
		    {
			for (kj=0; kj<m_nLabels; kj++)
			    for (ki=0; ki<m_nLabels; ki++)
				{
				    *ptr++ = smoothApp3(i,i+m_width,ki,kj);
				}
		    }
		else ptr += m_nLabels*m_nLabels;
	    }
}*/


dtype compute_graph(int num_parts_y,int num_parts_x,dtype *costs,int num_lab_y,int num_lab_x,dtype *data,int *reslab)
{
    MRF* mrf;
    //EnergyFunction *energy;
    MRF::EnergyVal E;
    double lowerBound;
    float t,tot_t;
    int iter;

    int seed = 1124285485;
    srand(seed);

    dtype scr;
    //copy costs in local memory
    dtype V[maxlab*maxlab];
    st_cost_y=costs;
    st_cost_x=costs+num_parts_x*num_parts_y;
    st_parts_x=num_parts_x;
    st_parts_y=num_parts_y;
    st_numlab=num_lab_y*num_lab_x;//num_parts_x*num_parts_y;//num_labels;
    clock_t t0,t1;
    t0 = clock ();
    int l1,l2;

    for (l1=0;l1<st_numlab;l1++)
    {
        for (l2=0;l2<st_numlab;l2++)
        {
//            int x1=l1%((st_parts_x*2-1)+1);
//            int y1=l1/((st_parts_x*2-1)+1);
//            int x2=l2%((st_parts_x*2-1)+1);    
//            int y2=l2/((st_parts_x*2-1)+1);            
            int x1=l1%num_lab_x;
            int y1=l1/num_lab_x;
            int x2=l2%num_lab_x;    
            int y2=l2/num_lab_x;            
            V[l1*st_numlab+l2]=abs(x1-x2)+abs(y1-y2); 
            //V[l2*st_numlab+l1]=abs(x1-x2)+abs(y1-y2); 
            //printf("V[%d,%d]%f ",l1,l2,V[l1*st_numlab+l2]);
        }
    }
    t1 = clock ();
    //printf("t0=%d t1=%d Diff %f",t0,t1,float(t1-t0)/CLOCKS_PER_SEC);
	//try{
        DataCost *dt         = new DataCost(data);
        //SmoothnessCost *sm   = new SmoothnessCost(smoothApp2);
        //SmoothnessCost *sm   = new SmoothnessCost(smoothApp3);
        SmoothnessCost *sm   = new SmoothnessCost(V, st_cost_x, st_cost_y);
        //SmoothnessCost *sm   = new SmoothnessCost(1, 100, 1, st_cost_x, st_cost_y);
        EnergyFunction *energy = new EnergyFunction(dt,sm);

//		GCoptimizationGeneralGraph *gc = new GCoptimizationGeneralGraph(num_parts,num_labels);

//		// set up the needed data to pass to function for the data costs
//		//ForDataFn toFn;
//		//toFn.data = data;
//		//toFn.numLab = num_labels;
//		//gc->setDataCost(&dataFn,&toFn);

//        gc->setDataCost(data);
//		
//		// smoothness comes from function pointer
//		gc->setSmoothCost(&smoothApp);

//        // now set up a graph neighborhood system
//        // use only the upper part of the matrix
//        for (int py=0; py<num_parts; py++ )
//            for (int px=py+1; px<num_parts; px++)
//                if (connect[py*num_parts+px]>0)
//                {
//                    //gc->setNeighbors(px,py);
//                    gc->setNeighbors(py,px);
//                }
//		
//		//printf("\nBefore optimization energy is %f",gc->compute_energy());
//		gc->expansion(10);// run expansion for 2 iterations. For swap use gc->swap(num_iterations);
//        scr=gc->compute_energy();
//		//printf("\nAfter optimization energy is %f \n",scr);

        ///new way
        //printf("%d,%d\n",num_parts_x,num_parts_y);
        //mrf = new MaxProdBP(num_parts_x,num_parts_y,num_parts_x*num_parts_y,energy);
        //mrf = new Swap(num_parts_x,num_parts_y,num_lab_y*num_lab_x,energy);
        mrf = new Expansion(num_parts_x,num_parts_y,num_lab_y*num_lab_x,energy);
#include<time.h>
        //clock_t t0,t1;
        //mrf = new TRWS(num_parts_x,num_parts_y,num_lab_y*num_lab_x,energy);
        //mrf = new ICM(num_parts_x,num_parts_y,st_numlab,energy);

	    // can disable caching of values of general smoothness function:
	    //mrf->dontCacheSmoothnessCosts();
        t0 = clock ();
	    mrf->initialize();
        t1 = clock ();
        //mrf->setCues(st_cost_x,st_cost_y);
	    //mrf->clearAnswer();

//        for ( int  i = 0; i < num_parts_y*num_parts_x; i++ )
//        {
//            mrf->setLabel(i,reslab[i]);
//			reslab[i] = mrf->getLabel(i);//gc->whatLabel(i);
//            printf("%d ",reslab[i]);
//        }           
//	    printf("+++++\n");
//	    E = mrf->totalEnergy();
	    //printf("Energy at the Start= %g (%g,%g)\n", (float)E,
		//   (float)mrf->smoothnessEnergy(), (float)mrf->dataEnergy());
        //printf("t0=%d t1=%d Diff %f",t0,t1,float(t1-t0)/CLOCKS_PER_SEC);
	    tot_t = 0;
        //printf("Before C\n");
	    for (iter=0; iter<1; iter++) 
        {
		    mrf->optimize(2, t);
            //printf("After C\n");
    		E = mrf->totalEnergy();
    		lowerBound = mrf->lowerBound();
    		tot_t = tot_t + t ;
            //printf("Energy= %g (%g,%g)\n", (float)E,
		   //(float)mrf->smoothnessEnergy(), (float)mrf->dataEnergy());
    		//printf("energy = %g, lower bound = %f (%f secs)\n", (float)E, lowerBound, tot_t);
	    }
        for ( int  i = 0; i < num_parts_y*num_parts_x; i++ )
        {
			reslab[i] = mrf->getLabel(i);//gc->whatLabel(i);
            //printf("%d ",reslab[i]);
        }   
	    delete mrf;
	//}
	//catch (GCException e){
	//	e.Report();
	//}

	//delete [] result;
	//delete [] data;
    return -E;

}

dtype compute(int width,int height,dtype *data,int *result)
{

	//int *result = new int[num_pixels];   // stores result of optimization

	// first set up the array for data costs
	/*int *data = new int[num_pixels*num_labels];
	for ( int i = 0; i < num_pixels; i++ )
		for (int l = 0; l < num_labels; l++ )
			data[i*num_labels+l] = 0;*/
/*    int num_labels=5;
    int num_pixels=width*height;
    dtype scr;

	try{
		GCoptimizationGridGraph *gc = new GCoptimizationGridGraph(width,height,num_labels);

		// set up the needed data to pass to function for the data costs
		ForDataFn toFn;
		toFn.data = data;
		toFn.numLab = num_labels;

		gc->setDataCost(&dataFn,&toFn);

		// smoothness comes from function pointer
		gc->setSmoothCost(&smoothFn);

		printf("\nBefore optimization energy is %f",gc->compute_energy());
		gc->expansion(10);// run expansion for 2 iterations. For swap use gc->swap(num_iterations);
        scr=gc->compute_energy();
		printf("\nAfter optimization energy is %f \n",scr);

		for ( int  i = 0; i < num_pixels; i++ )
			result[i] = gc->whatLabel(i);

		delete gc;
	}
	catch (GCException e){
		e.Report();
	}

	//delete [] result;
	//delete [] data;
    return scr;
*/
}


} //end extern C


/*int main(int argc, char **argv)
{
    MRF* mrf;
    EnergyFunction *energy;
    MRF::EnergyVal E;
    double lowerBound;
    float t,tot_t;
    int iter;

    int seed = 1124285485;
    srand(seed);

    int Etype = 0;

    if (argc != 2) {
	fprintf(stderr, usage, argv[0]);
	exit(1);
    }
    
    if (argc > 1)
	Etype = atoi(argv[1]);

    try {
	switch(Etype) {
	    // Here are 4 sample energies to play with.
	case 0:
	    energy = generate_DataARRAY_SmoothFIXED_FUNCTION();
	    fprintf(stderr, "using fixed (array) smoothness cost\n");
	    break;
	case 1:
	    energy = generate_DataARRAY_SmoothTRUNCATED_LINEAR();
	    fprintf(stderr, "using truncated linear smoothness cost\n");
	    break;
	case 2:
	    energy = generate_DataARRAY_SmoothTRUNCATED_QUADRATIC();
	    fprintf(stderr, "using truncated quadratic smoothness cost\n");
	    break;
	case 3:
	    energy = generate_DataFUNCTION_SmoothGENERAL_FUNCTION();
	    fprintf(stderr, "using general smoothness functions\n");
	    break;
	default:
	    fprintf(stderr, usage, argv[0]);
	    exit(1);
	}

	bool runICM       = true;
	bool runExpansion = true;
	bool runSwap      = true;
	bool runMaxProdBP = true;
	bool runTRWS      = true;
	bool runBPS       = true;

	////////////////////////////////////////////////
	//                     ICM                    //
	////////////////////////////////////////////////
	if (runICM) {
	    printf("\n*******Started ICM *****\n");

	    mrf = new ICM(sizeX,sizeY,numLabels,energy);
	    mrf->initialize();
	    mrf->clearAnswer();

	    E = mrf->totalEnergy();
	    printf("Energy at the Start= %g (%g,%g)\n", (float)E,
		   (float)mrf->smoothnessEnergy(), (float)mrf->dataEnergy());

	    tot_t = 0;
	    for (iter=0; iter<6; iter++) {
		mrf->optimize(10, t);
		
		E = mrf->totalEnergy();
		tot_t = tot_t + t ;
		printf("energy = %g (%f secs)\n", (float)E, tot_t);
	    }

	    delete mrf;
	}

	////////////////////////////////////////////////
	//          Graph-cuts expansion              //
	////////////////////////////////////////////////
	if (runExpansion) {
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
	}

	////////////////////////////////////////////////
	//          Belief Propagation                //
	////////////////////////////////////////////////
	if (runMaxProdBP) {
	    printf("\n*******  Started MaxProd Belief Propagation *****\n");
	    mrf = new MaxProdBP(sizeX,sizeY,numLabels,energy);
	    mrf->initialize();
	    mrf->clearAnswer();
	    
	    E = mrf->totalEnergy();
	    printf("Energy at the Start= %g (%g,%g)\n", (float)E,
		   (float)mrf->smoothnessEnergy(), (float)mrf->dataEnergy());

	    tot_t = 0;
	    for (iter=0; iter < 10; iter++) {
		mrf->optimize(1, t);

		E = mrf->totalEnergy();
		tot_t = tot_t + t ;
		printf("energy = %g (%f secs)\n", (float)E, tot_t);
	    }

	    
	    delete mrf;
	}

	////////////////////////////////////////////////
	//                  TRW-S                     //
	////////////////////////////////////////////////
	if (runTRWS) {
	    printf("\n*******Started TRW-S *****\n");
	    mrf = new TRWS(sizeX,sizeY,numLabels,energy);

	    // can disable caching of values of general smoothness function:
	    //mrf->dontCacheSmoothnessCosts();

	    mrf->initialize();
	    mrf->clearAnswer();

	    
	    E = mrf->totalEnergy();
	    printf("Energy at the Start= %g (%g,%g)\n", (float)E,
		   (float)mrf->smoothnessEnergy(), (float)mrf->dataEnergy());

	    tot_t = 0;
	    for (iter=0; iter<10; iter++) {
		mrf->optimize(10, t);

		E = mrf->totalEnergy();
		lowerBound = mrf->lowerBound();
		tot_t = tot_t + t ;
		printf("energy = %g, lower bound = %f (%f secs)\n", (float)E, lowerBound, tot_t);
	    }

	    delete mrf;
	}

	////////////////////////////////////////////////
	//                  BP-S                     //
	////////////////////////////////////////////////
	if (runBPS) {
	    printf("\n*******Started BP-S *****\n");
	    mrf = new BPS(sizeX,sizeY,numLabels,energy);

	    // can disable caching of values of general smoothness function:
	    //mrf->dontCacheSmoothnessCosts();
		
	    mrf->initialize();
	    mrf->clearAnswer();
	    
	    E = mrf->totalEnergy();
	    printf("Energy at the Start= %g (%g,%g)\n", (float)E,
		   (float)mrf->smoothnessEnergy(), (float)mrf->dataEnergy());

	    tot_t = 0;
	    for (iter=0; iter<10; iter++) {
		mrf->optimize(10, t);

		E = mrf->totalEnergy();
		tot_t = tot_t + t ;
		printf("energy = %g (%f secs)\n", (float)E, tot_t);
	    }

	    delete mrf;
	}
    }
    catch (std::bad_alloc) {
	fprintf(stderr, "*** Error: not enough memory\n");
	exit(1);
    }

    return 0;
}*/
