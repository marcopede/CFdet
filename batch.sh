#!/bin/bash 

apply() 
{ 
    for ARG in "$@" 
    do 
        echo "Training on " $ARG 
        python trainFAST.py $ARG batch
    done 
} 

apply aeroplane bicycle bird boat bottle bus car cat chair cow dinigtable dog horse motorbike person pottedplant sheep sofa train tvmonitor 

