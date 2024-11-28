for hcls_weight in 500 400 300 200 100 110 120 130 90 80 ; do
    for sm in 0.1 0.11 0.12 0.13 0.14 0.09 0.08 0.07; do
        CUDA_VISIBLE_DEVICES=1 python main_text.py \
            --smoothing ${sm}  \
            --hcls_weight ${hcls_weight} \
            --hcls \
            --nocl 
    done
done