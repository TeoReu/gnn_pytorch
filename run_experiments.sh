#!/bin/bash
for integ in 'Clin+mRNA' 'CNA+mRNA' 'Clin+CNA'
do
    for lsize in 16 64
    do
        for distance in 'kl' #'mmd'
        do
            for beta in 1 #10 15 25 50 100
            do
                for dtype in  'W' #whole data
                do
                    for k in 10 30 60
                    do
                        for epochs in 1500
                        do
                            python run_gcncvae.py --integration=${integ}  --epochs=${epochs} --dtype=${dtype}  --ls=${lsize} --distance=${distance} --beta=${beta} --k=${k} --writedir='results'
                        done
                    done
                done
            done
        done
    done
done