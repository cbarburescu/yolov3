#!/bin/bash

log_dir="./runs"
pattern=$1  #ex '*_HFP8' 

for subdir in $(find $log_dir -mindepth 1 -maxdepth 1 -name "$pattern")
do
    # echo $subdir
    for file in $(ls $subdir | head -n -2)
    do
        to_del="$subdir/$file"
        echo $to_del
        rm -rf $to_del
    done
done
