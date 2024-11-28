for sm in 0.1 0.11 0.12 0.13 0.14 0.09 0.08 0.07; do
        CUDA_VISIBLE_DEVICES=0 python main_text_bert.py \
            --smoothing ${sm}  \
            --nocl 
done

for mcls_weight in 500 400 300 200 100 110 120 130 90 80 ; do
    for sm in 0.1 0.11 0.12 0.13 0.14 0.09 0.08 0.07; do
        CUDA_VISIBLE_DEVICES=0 python main_text_bert.py \
            --smoothing ${sm}  \
            --mcls_weight ${mcls_weight} \
            --mcls \
            --nocl 
    done
done