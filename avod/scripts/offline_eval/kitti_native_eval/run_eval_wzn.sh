#!/bin/bash


echo "$3" | tee -a ./$4_results_$2.txt
./evaluate_object_3d_offline ~/Kitti/object/training/label_2/ $1/$2/$3 | tee -a ./$4_results_$2.txt

cp $4_results_$2.txt $5
