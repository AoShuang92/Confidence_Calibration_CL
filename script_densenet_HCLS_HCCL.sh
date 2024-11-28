for init_ratio in 1 0.96 0.93 0.9 0.87 0.83 0,8 ; do
    for epiid in 5 10 15 20 25 30 35 40 50 60 ; do
        CUDA_VISIBLE_DEVICES=1 python main_densenet.py \
            --init_ratio ${init_ratio} \
            --end_epoch ${epiid} \
            --hcls \
            --hccl \
            --smoothing 0.1 \
            --hcls_weight 300
        if [ $init_ratio -eq 1 ]
        then
            break
        fi
    done
done