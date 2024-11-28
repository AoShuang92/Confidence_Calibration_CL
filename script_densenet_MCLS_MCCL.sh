for init_ratio in 0.78 0.75 0.72 0.7 0.65 ; do
    for epiid in 5 10 15 20 25 30 35 40 50 60 ; do
        CUDA_VISIBLE_DEVICES=0 python main_densenet.py \
            --init_ratio ${init_ratio} \
            --end_epoch ${epiid} \
            --mcls \
            --mccl \
            --smoothing 0.12 \
            --mcls_weight 400
        if [ $init_ratio -eq 1 ]
        then
            break
        fi
    done
done