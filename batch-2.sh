#!/bin/bash 

apply() 
{ 
    for ARG in "$@" 
    do 
        echo "Training on " $ARG 
        python trainBOW.py $ARG batch
    done 
} 

apply diningtable dog horse motorbike pottedplant sheep sofa train tvmonitor person

