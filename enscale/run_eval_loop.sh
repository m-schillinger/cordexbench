DOMAINS=(SA)
EXPERIMENTS=(
    "ESD_pseudo_reality"
  #  "Emulator_hist_future"
)
ORO_OPTIONS=("--add_orography") 
# "--add_orography")
for DOMAIN in "${DOMAINS[@]}"; do
    for ORO_FLAG in "${ORO_OPTIONS[@]}"; do
        for EXPERIMENT in "${EXPERIMENTS[@]}"; do
        echo "Starting evaluation for domain ${DOMAIN} with experiment ${EXPERIMENT} and orography flag ${ORO_FLAG}"

        # # Run evaluation script for each variable and version
        # CUDA_VISIBLE_DEVICES=2 python eval_multi_step_coarse_from_super.py \
        #     --variable tasmax \
        #     --domain ${DOMAIN} \
        #     --version normpw_v2 \
        #     --training_experiment ${EXPERIMENT} \
        #     --norm_option pw \
        #     --nicolai_layers \
        #     $ORO_FLAG \
        #     --mode train &

        # CUDA_VISIBLE_DEVICES=2 python eval_multi_step_coarse_from_super.py \
        #     --variable tasmax \
        #     --domain ${DOMAIN} \
        #     --version normpw_v2 \
        #     --training_experiment ${EXPERIMENT} \
        #     --norm_option pw \
        #     --nicolai_layers \
        #     $ORO_FLAG \
        #     --mode inference 
        
        # wait

        CUDA_VISIBLE_DEVICES=2 python eval_multi_step_coarse_from_super.py \
            --variable pr \
            --domain ${DOMAIN} \
            --version scalepw_v2 \
            --training_experiment ${EXPERIMENT} \
            --norm_option scale_pw \
            --nicolai_layers \
            $ORO_FLAG \
            --mode train &

        CUDA_VISIBLE_DEVICES=2 python eval_multi_step_coarse_from_super.py \
            --variable pr \
            --domain ${DOMAIN} \
            --version scalepw_v2 \
            --training_experiment ${EXPERIMENT} \
            --norm_option scale_pw \
            --nicolai_layers \
            $ORO_FLAG \
            --mode inference &
        wait
        done
    done
done
