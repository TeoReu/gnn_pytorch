#!/bin/bash
for integ in 'Clin+CNA' #'CNA+mRNA' 'Clin+mRNA'
do
    for lsize in 32
    do
        for dtype in  'ER' 'DR' 'IC' 'PAM' #whole data
        do
            for graph_type in  'simple'
            do
                for fold in 1 2 3 4 5
                do
                    for k in 15
                    do
                        for epochs in 500
                        do
                            python run_infomax.py --fold=${fold} --integration=${integ}  --epochs=${epochs} --dtype=${dtype}  --ls=${lsize} --k=${k} --writedir='results'
                        done
                    done
                done
            done
        done
    done
done