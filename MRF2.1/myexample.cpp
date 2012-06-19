//////////////////////////////////////////////////////////////////////////////
// Example illustrating the use of GCoptimization.cpp
//
/////////////////////////////////////////////////////////////////////////////
//
//  Optimization problem:
//  is a set of sites (pixels) of width 10 and hight 5. Thus number of pixels is 50
//  grid neighborhood: each pixel has its left, right, up, and bottom pixels as neighbors
//  7 labels
//  Data costs: D(pixel,label) = 0 if pixel < 25 and label = 0
//            : D(pixel,label) = 10 if pixel < 25 and label is not  0
//            : D(pixel,label) = 0 if pixel >= 25 and label = 5
//            : D(pixel,label) = 10 if pixel >= 25 and label is not  5
// Smoothness costs: V(p1,p2,l1,l2) = min( (l1-l2)*(l1-l2) , 4 )
// Below in the main program, we illustrate different ways of setting data and smoothness costs
// that our interface allow and solve this optimizaiton problem

// For most of the examples, we use no spatially varying pixel dependent terms. 
// For some examples, to demonstrate spatially varying terms we use
// V(p1,p2,l1,l2) = w_{p1,p2}*[min((l1-l2)*(l1-l2),4)], with 
// w_{p1,p2} = p1+p2 if |p1-p2| == 1 and w_{p1,p2} = p1*p2 if |p1-p2| is not 1

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include "GCoptimization.h"

#define dtype float

struct ForDataFn{
	int numLab;
	dtype *data;
};


/*int smoothFn(int p1, int p2, int l1, int l2)
{
	if ( (l1-l2)*(l1-l2) <= 4 ) return((l1-l2)*(l1-l2));
	else return(4);
}*/

dtype dataFn(int p, int l, void *data)
{
	ForDataFn *myData = (ForDataFn *) data;
	int numLab = myData->numLab;
	
	return( myData->data[p*numLab+l] );
}



////////////////////////////////////////////////////////////////////////////////
// smoothness and data costs are set up one by one, individually
// grid neighborhood structure is assumed
//
/*void GridGraph_Individually(int width,int height,int num_pixels,int num_labels)
{

	int *result = new int[num_pixels];   // stores result of optimization



	try{
		GCoptimizationGridGraph *gc = new GCoptimizationGridGraph(width,height,num_labels);

		// first set up data costs individually
		for ( int i = 0; i < num_pixels; i++ )
			for (int l = 0; l < num_labels; l++ )
				if (i < 25 ){
					if(  l == 0 ) gc->setDataCost(i,l,0);
					else gc->setDataCost(i,l,10);
				}
				else {
					if(  l == 5 ) gc->setDataCost(i,l,0);
					else gc->setDataCost(i,l,10);
				}

		// next set up smoothness costs individually
		for ( int l1 = 0; l1 < num_labels; l1++ )
			for (int l2 = 0; l2 < num_labels; l2++ ){
				int cost = (l1-l2)*(l1-l2) <= 4  ? (l1-l2)*(l1-l2):4;
				gc->setSmoothCost(l1,l2,cost); 
			}

		printf("\nBefore optimization energy is %d",gc->compute_energy());
		gc->expansion(2);// run expansion for 2 iterations. For swap use gc->swap(num_iterations);
		printf("\nAfter optimization energy is %d",gc->compute_energy());

		for ( int  i = 0; i < num_pixels; i++ )
			result[i] = gc->whatLabel(i);

		delete gc;
	}
	catch (GCException e){
		e.Report();
	}

	delete [] result;
}

////////////////////////////////////////////////////////////////////////////////
// in this version, set data and smoothness terms using arrays
// grid neighborhood structure is assumed
//
void GridGraph_DArraySArray(int width,int height,int num_pixels,int num_labels)
{

	int *result = new int[num_pixels];   // stores result of optimization

	// first set up the array for data costs
	int *data = new int[num_pixels*num_labels];
	for ( int i = 0; i < num_pixels; i++ )
		for (int l = 0; l < num_labels; l++ )
			if (i < 25 ){
				if(  l == 0 ) data[i*num_labels+l] = 0;
				else data[i*num_labels+l] = 10;
			}
			else {
				if(  l == 5 ) data[i*num_labels+l] = 0;
				else data[i*num_labels+l] = 10;
			}
	// next set up the array for smooth costs
	int *smooth = new int[num_labels*num_labels];
	for ( int l1 = 0; l1 < num_labels; l1++ )
		for (int l2 = 0; l2 < num_labels; l2++ )
			smooth[l1+l2*num_labels] = (l1-l2)*(l1-l2) <= 4  ? (l1-l2)*(l1-l2):4;


	try{
		GCoptimizationGridGraph *gc = new GCoptimizationGridGraph(width,height,num_labels);
		gc->setDataCost(data);
		gc->setSmoothCost(smooth);
		printf("\nBefore optimization energy is %d",gc->compute_energy());
		gc->expansion(2);// run expansion for 2 iterations. For swap use gc->swap(num_iterations);
		printf("\nAfter optimization energy is %d",gc->compute_energy());

		for ( int  i = 0; i < num_pixels; i++ )
			result[i] = gc->whatLabel(i);

		delete gc;
	}
	catch (GCException e){
		e.Report();
	}

	delete [] result;
	delete [] smooth;
	delete [] data;

}*/
////////////////////////////////////////////////////////////////////////////////
// in this version, set data and smoothness terms using arrays
// grid neighborhood structure is assumed
//

#define numcl 5

int D[]={0,1,1,1,1,
         1,0,1,1,1,
         1,1,0,1,1,
         1,1,1,0,1,
         1,1,1,1,0};

int O[]={0,0,1,1,1,
         0,0,1,1,1,
         1,1,0,0,1,
         1,1,0,0,1,
         1,1,1,1,0};

int V[]={0,1,0,1,1,
         1,0,1,0,1,
         0,1,0,1,1,
         1,0,1,0,1,
         1,1,1,1,0};

dtype smoothFn(int p1, int p2, int l1, int l2)
{
	//if ( (l1-l2)*(l1-l2) <= 4 ) return((l1-l2)*(l1-l2));
	//else return(4);
    //printf("p1:%d,p2:%d ",p1,p2);
    if (abs(p1-p2)==1) //orizontal
    {
        //printf("O\n");
        return 0.1*O[l1+numcl*l2];
    }
    else
    {
        //printf("V\n");
        return 0.1*V[l1+numcl*l2];        
    }
}

//nparts=4;
//nlab=2;
//dtype costs[nparts][nparts][nlab][nlab];
//int connect[nparts][nparts];

dtype *st_cost;
int st_nparts;
int st_numlab;


dtype smoothApp(int p1, int p2, int l1, int l2)
{
	//if ( (l1-l2)*(l1-l2) <= 4 ) return((l1-l2)*(l1-l2));
	//else return(4);
    //printf("p1:%d,p2:%d ",p1,p2);
    //if (abs(p1-p2)==1) //orizontal
    //{
        //printf("O\n");
    //    return 0.1*O[l1+numcl*l2];
    //}
    //else
    //{
    //    //printf("V\n");
    //    return 0.1*V[l1+numcl*l2];        
    //}
    //printf("p1 %d p2 %d l1 %d l2 %d cost %f \n",p1,p2,l1,l2,st_cost[p1*st_nparts*st_numlab*st_numlab+p2*st_numlab*st_numlab+l1*st_numlab+l2]);
    if (p1>p2)
    {
        int tmp;
        tmp=p2;
        p2=p1;
        p1=tmp;
    }
    //return st_cost[p1*st_nparts*st_numlab*st_numlab+p2*st_numlab*st_numlab+l1*st_numlab+l2];
    return st_cost[p1*st_nparts*st_numlab*st_numlab+p2*st_numlab*st_numlab+l2*st_numlab+l1];
}

extern "C" {

dtype compute(int width,int height,dtype *data,int *result)
{

	//int *result = new int[num_pixels];   // stores result of optimization

	// first set up the array for data costs
	/*int *data = new int[num_pixels*num_labels];
	for ( int i = 0; i < num_pixels; i++ )
		for (int l = 0; l < num_labels; l++ )
			data[i*num_labels+l] = 0;*/
    int num_labels=5;
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

}


// connect[nparts][nparts];costs[nparts][nparts][nlabels][nlabels];data[nparts][nlabels];reslab[nparts];
dtype compute_graph(int *connect,int num_parts,dtype *costs,int num_labels,dtype *data,int *reslab)
{

	//int *result = new int[num_pixels];   // stores result of optimization

	// first set up the array for data costs
	/*int *data = new int[num_pixels*num_labels];
	for ( int i = 0; i < num_pixels; i++ )
		for (int l = 0; l < num_labels; l++ )
			data[i*num_labels+l] = 0;*/
    //int num_labels=5;
    //int num_pixels=width*height;
    dtype scr;
    //copy costs in local memory
    st_cost=costs;
    st_nparts=num_parts;
    st_numlab=num_labels;

	try{
		GCoptimizationGeneralGraph *gc = new GCoptimizationGeneralGraph(num_parts,num_labels);

		// set up the needed data to pass to function for the data costs
		//ForDataFn toFn;
		//toFn.data = data;
		//toFn.numLab = num_labels;
		//gc->setDataCost(&dataFn,&toFn);

        gc->setDataCost(data);
		
		// smoothness comes from function pointer
		gc->setSmoothCost(&smoothApp);

        // now set up a graph neighborhood system
        // use only the upper part of the matrix
        for (int py=0; py<num_parts; py++ )
            for (int px=py+1; px<num_parts; px++)
                if (connect[py*num_parts+px]>0)
                {
                    //gc->setNeighbors(px,py);
                    gc->setNeighbors(py,px);
                }
		
		//printf("\nBefore optimization energy is %f",gc->compute_energy());
		gc->expansion(10);// run expansion for 2 iterations. For swap use gc->swap(num_iterations);
        scr=gc->compute_energy();
		//printf("\nAfter optimization energy is %f \n",scr);

		for ( int  i = 0; i < num_parts; i++ )
			reslab[i] = gc->whatLabel(i);

		delete gc;
	}
	catch (GCException e){
		e.Report();
	}

	//delete [] result;
	//delete [] data;
    return scr;

}


}

////////////////////////////////////////////////////////////////////////////////
// Uses spatially varying smoothness terms. That is 
// V(p1,p2,l1,l2) = w_{p1,p2}*[min((l1-l2)*(l1-l2),4)], with 
// w_{p1,p2} = p1+p2 if |p1-p2| == 1 and w_{p1,p2} = p1*p2 if |p1-p2| is not 1
/*void GridGraph_DArraySArraySpatVarying(int width,int height,int num_pixels,int num_labels)
{
	int *result = new int[num_pixels];   // stores result of optimization

	// first set up the array for data costs
	int *data = new int[num_pixels*num_labels];
	for ( int i = 0; i < num_pixels; i++ )
		for (int l = 0; l < num_labels; l++ )
			if (i < 25 ){
				if(  l == 0 ) data[i*num_labels+l] = 0;
				else data[i*num_labels+l] = 10;
			}
			else {
				if(  l == 5 ) data[i*num_labels+l] = 0;
				else data[i*num_labels+l] = 10;
			}
	// next set up the array for smooth costs
	int *smooth = new int[num_labels*num_labels];
	for ( int l1 = 0; l1 < num_labels; l1++ )
		for (int l2 = 0; l2 < num_labels; l2++ )
			smooth[l1+l2*num_labels] = (l1-l2)*(l1-l2) <= 4  ? (l1-l2)*(l1-l2):4;

	// next set up spatially varying arrays V and H

	int *V = new int[num_pixels];
	int *H = new int[num_pixels];

	
	for ( int i = 0; i < num_pixels; i++ ){
		H[i] = i+(i+1)%3;
		V[i] = i*(i+width)%7;
	}


	try{
		GCoptimizationGridGraph *gc = new GCoptimizationGridGraph(width,height,num_labels);
		gc->setDataCost(data);
		gc->setSmoothCostVH(smooth,V,H);
		printf("\nBefore optimization energy is %d",gc->compute_energy());
		gc->expansion(2);// run expansion for 2 iterations. For swap use gc->swap(num_iterations);
		printf("\nAfter optimization energy is %d",gc->compute_energy());

		for ( int  i = 0; i < num_pixels; i++ )
			result[i] = gc->whatLabel(i);

		delete gc;
	}
	catch (GCException e){
		e.Report();
	}

	delete [] result;
	delete [] smooth;
	delete [] data;


}

////////////////////////////////////////////////////////////////////////////////
// in this version, set data and smoothness terms using arrays
// grid neighborhood is set up "manually"
//
void GeneralGraph_DArraySArray(int width,int height,int num_pixels,int num_labels)
{

	int *result = new int[num_pixels];   // stores result of optimization

	// first set up the array for data costs
	int *data = new int[num_pixels*num_labels];
	for ( int i = 0; i < num_pixels; i++ )
		for (int l = 0; l < num_labels; l++ )
			if (i < 25 ){
				if(  l == 0 ) data[i*num_labels+l] = 0;
				else data[i*num_labels+l] = 10;
			}
			else {
				if(  l == 5 ) data[i*num_labels+l] = 0;
				else data[i*num_labels+l] = 10;
			}
	// next set up the array for smooth costs
	int *smooth = new int[num_labels*num_labels];
	for ( int l1 = 0; l1 < num_labels; l1++ )
		for (int l2 = 0; l2 < num_labels; l2++ )
			smooth[l1+l2*num_labels] = (l1-l2)*(l1-l2) <= 4  ? (l1-l2)*(l1-l2):4;


	try{
		GCoptimizationGeneralGraph *gc = new GCoptimizationGeneralGraph(num_pixels,num_labels);
		gc->setDataCost(data);
		gc->setSmoothCost(smooth);

		// now set up a grid neighborhood system
		// first set up horizontal neighbors
		for (int y = 0; y < height; y++ )
			for (int  x = 1; x < width; x++ )
				gc->setNeighbors(x+y*width,x-1+y*width);

		// next set up vertical neighbors
		for (int y = 1; y < height; y++ )
			for (int  x = 0; x < width; x++ )
				gc->setNeighbors(x+y*width,x+(y-1)*width);

		printf("\nBefore optimization energy is %d",gc->compute_energy());
		gc->expansion(2);// run expansion for 2 iterations. For swap use gc->swap(num_iterations);
		printf("\nAfter optimization energy is %d",gc->compute_energy());

		for ( int  i = 0; i < num_pixels; i++ )
			result[i] = gc->whatLabel(i);

		delete gc;
	}
	catch (GCException e){
		e.Report();
	}

	delete [] result;
	delete [] smooth;
	delete [] data;

}
////////////////////////////////////////////////////////////////////////////////
// in this version, set data and smoothness terms using arrays
// grid neighborhood is set up "manually". Uses spatially varying terms. Namely
// V(p1,p2,l1,l2) = w_{p1,p2}*[min((l1-l2)*(l1-l2),4)], with 
// w_{p1,p2} = p1+p2 if |p1-p2| == 1 and w_{p1,p2} = p1*p2 if |p1-p2| is not 1

void GeneralGraph_DArraySArraySpatVarying(int width,int height,int num_pixels,int num_labels)
{
	int *result = new int[num_pixels];   // stores result of optimization

	// first set up the array for data costs
	int *data = new int[num_pixels*num_labels];
	for ( int i = 0; i < num_pixels; i++ )
		for (int l = 0; l < num_labels; l++ )
			if (i < 25 ){
				if(  l == 0 ) data[i*num_labels+l] = 0;
				else data[i*num_labels+l] = 10;
			}
			else {
				if(  l == 5 ) data[i*num_labels+l] = 0;
				else data[i*num_labels+l] = 10;
			}
	// next set up the array for smooth costs
	int *smooth = new int[num_labels*num_labels];
	for ( int l1 = 0; l1 < num_labels; l1++ )
		for (int l2 = 0; l2 < num_labels; l2++ )
			smooth[l1+l2*num_labels] = (l1-l2)*(l1-l2) <= 4  ? (l1-l2)*(l1-l2):4;


	try{
		GCoptimizationGeneralGraph *gc = new GCoptimizationGeneralGraph(num_pixels,num_labels);
		gc->setDataCost(data);
		gc->setSmoothCost(smooth);

		// now set up a grid neighborhood system
		// first set up horizontal neighbors
		for (int y = 0; y < height; y++ )
			for (int  x = 1; x < width; x++ ){
				int p1 = x-1+y*width;
				int p2 =x+y*width;
				gc->setNeighbors(p1,p2,p1+p2);
			}

		// next set up vertical neighbors
		for (int y = 1; y < height; y++ )
			for (int  x = 0; x < width; x++ ){
				int p1 = x+(y-1)*width;
				int p2 =x+y*width;
				gc->setNeighbors(p1,p2,p1*p2);
			}

		printf("\nBefore optimization energy is %d",gc->compute_energy());
		gc->expansion(2);// run expansion for 2 iterations. For swap use gc->swap(num_iterations);
		printf("\nAfter optimization energy is %d",gc->compute_energy());

		for ( int  i = 0; i < num_pixels; i++ )
			result[i] = gc->whatLabel(i);

		delete gc;
	}
	catch (GCException e){
		e.Report();
	}

	delete [] result;
	delete [] smooth;
	delete [] data;


}
////////////////////////////////////////////////////////////////////////////////
*/
