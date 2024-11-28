for init_ratio in 1 0.96 0.93 0.9; do
    for epiid in 5 10 15 20 25 30 35 40 50 60; do
        CUDA_VISIBLE_DEVICES=0 python main_text_bert.py \
            --init_ratio ${init_ratio} \
            --end_epoch ${epiid} \
            --mcls \
            --mccl \
            --smoothing 0.12 \
            --mcls_weight 500
        if [ $init_ratio -eq 1 ]
        then
            break
        fi
    done
done