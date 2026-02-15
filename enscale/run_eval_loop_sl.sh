DOMAINS=(NZ SA ALPS)
EXPERIMENTS=(
    "Emulator_hist_future"
)
ORO_OPTIONS=("")
for DOMAIN in "${DOMAINS[@]}"; do
    for ORO_FLAG in "${ORO_OPTIONS[@]}"; do
        for EXPERIMENT in "${EXPERIMENTS[@]}"; do
        echo "Starting evaluation for domain ${DOMAIN} with experiment ${EXPERIMENT} and orography flag ${ORO_FLAG}"

        if [[ "$DOMAIN" == "ALPS" ]]; then
            ALPHA=100
        else
            ALPHA=1000
        fi

        # Run evaluation script for each variable and version
        CUDA_VISIBLE_DEVICES=0 python eval_multi_step_coarse_from_super.py \
            --variable tasmax \
            --domain ${DOMAIN} \
            --version linex_v1 \
            --training_experiment ${EXPERIMENT} \
            --norm_option subtract_linear \
            --alpha_subtract_linear ${ALPHA} \
            --nicolai_layers \
            --mode train &

        CUDA_VISIBLE_DEVICES=0 python eval_multi_step_coarse_from_super.py \
            --variable tasmax \
            --domain ${DOMAIN} \
            --version linex_v1 \
            --training_experiment ${EXPERIMENT} \
            --norm_option subtract_linear \
            --alpha_subtract_linear ${ALPHA} \
            --nicolai_layers \
            --mode inference 
        
        wait

        CUDA_VISIBLE_DEVICES=1 python eval_multi_step_coarse_from_super.py \
            --variable pr \
            --domain ${DOMAIN} \
            --version linex_v1 \
            --training_experiment ${EXPERIMENT} \
            --norm_option subtract_linear \
            --alpha_subtract_linear ${ALPHA} \
            --nicolai_layers \
            --mode train &

        CUDA_VISIBLE_DEVICES=1 python eval_multi_step_coarse_from_super.py \
            --variable pr \
            --domain ${DOMAIN} \
            --version linex_v1 \
            --training_experiment ${EXPERIMENT} \
            --norm_option subtract_linear \
            --alpha_subtract_linear ${ALPHA} \
            --nicolai_layers \
            --mode inference &
        wait
        done
    done
done
