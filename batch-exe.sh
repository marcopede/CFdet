#!/bin/bash 

EXE=$1
echo $EXE

apply() 
{ 
    for ARG in "$@" 
    do 
        echo "Training using" $EXE "on" $ARG 
        python $EXE $ARG batch
    done 
} 

#apply aeroplane bicycle bird boat bottle bus car cat chair cow diningtable dog horse motorbike person pottedplant sheep sofa train tvmonitor 
#apply aeroplane bicycle bird boat bus cat chair cow diningtable dog horse motorbike person pottedplant sheep sofa train tvmonitor 
apply tvmonitor 
