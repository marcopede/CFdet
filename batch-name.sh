#!/bin/bash 

IMPORT=$1
echo $IMPORT

apply() 
{ 
    for ARG in "$@" 
    do 
        echo "Training on " $ARG 
        python trainBOW.py $ARG batch $IMPORT
    done 
} 

#apply aeroplane bicycle bird boat bottle bus car cat chair cow diningtable dog horse motorbike person pottedplant sheep sofa train tvmonitor 
apply aeroplane bicycle bird boat bus cat chair cow diningtable dog horse motorbike person pottedplant sheep sofa train tvmonitor 
