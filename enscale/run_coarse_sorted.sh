DOMAINS=(SA ALPS NZ)
ORO_OPTIONS=("")
EXPERIMENTS=(
    "ESD_pseudo_reality"
    "Emulator_hist_future"
)

for DOMAIN in "${DOMAINS[@]}"; do

    if [[ "$DOMAIN" == "ALPS" ]]; then
        ALPHA=100
        ALPHA_STRING=""
    else
        ALPHA=1000
        ALPHA_STRING="_alpha-1000"
    fi
    for ORO_FLAG in "${ORO_OPTIONS[@]}"; do
        for EXPERIMENT in "${EXPERIMENTS[@]}"; do
            if [[ -z "$ORO_FLAG" ]]; then
                echo "Starting domain ${DOMAIN} WITHOUT orography and experiment ${EXPERIMENT}"
            else
                echo "Starting domain ${DOMAIN} WITH orography and experiment ${EXPERIMENT}"
            fi
            CUDA_VISIBLE_DEVICES=1 python train_only-coarse.py \
                --training_experiment ${EXPERIMENT} \
                --variables pr \
                --domain ${DOMAIN} \
                --variables_lr all \
                --save_name "_norm-in-normalise_scalar_nd-100${ALPHA_STRING}_sorted-pred" \
                --num_layer 6 \
                --hidden_dim 100 \
                --preproc_layer \
                --preproc_dim 20 \
                --kernel_size_lr 16 \
                --layer_shrinkage 1 \
                --noise_dim 100 \
                --out_act None \
                --method eng_2step \
                --norm_method_input normalise_scalar \
                --norm_method_output subtract_linear \
                --alpha_subtract_linear ${ALPHA} \
                --sqrt_transform_in \
                --sqrt_transform_out \
                --agg_norm_loss mean \
                --lambda_norm_loss_loc 1 \
                --p_norm_loss_loc 4 \
                --p_norm_loss_batch 4 \
                --save_model_every 100 \
                --num_epochs 1000 \
                --batch_size 1024 \
                > logs/log_gpu4-pr-coarse-${DOMAIN}-$(date +"%Y%m%d_%H%M%S").txt 2>&1 &

            # CUDA_VISIBLE_DEVICES=0 python train_only-coarse.py \
            #     --training_experiment ${EXPERIMENT} \
            #     --variables pr \
            #     --domain ${DOMAIN} \
            #     --variables_lr all \
            #     --save_name '_relu_norm-in-normalise_scalar_nd-100_sorted-pred' \
            #     --num_layer 6 \
            #     --hidden_dim 100 \
            #     --preproc_layer \
            #     --preproc_dim 20 \
            #     --kernel_size_lr 16 \
            #     --layer_shrinkage 1 \
            #     --noise_dim 100 \
            #     --out_act relu \
            #     --method eng_2step \
            #     --norm_method_input normalise_scalar \
            #     --norm_method_output scale_pw \
            #     --sqrt_transform_in \
            #     --sqrt_transform_out \
            #     --agg_norm_loss mean \
            #     --lambda_norm_loss_loc 1 \
            #     --p_norm_loss_loc 4 \
            #     --p_norm_loss_batch 4 \
            #     --save_model_every 100 \
            #     --num_epochs 1000 \
            #     --batch_size 1024 \
            #     > logs/log_gpu4-pr-coarse-${DOMAIN}-$(date +"%Y%m%d_%H%M%S").txt 2>&1 &

            CUDA_VISIBLE_DEVICES=1 python train_only-coarse.py \
                --variables tasmax \
                --training_experiment ${EXPERIMENT} \
                --domain ${DOMAIN} \
                --variables_lr all \
                --save_name "_norm-in-normalise_scalar_nd-100${ALPHA_STRING}_sorted-pred" \
                --num_layer 6 \
                --hidden_dim 100 \
                --preproc_layer \
                --preproc_dim 20 \
                --kernel_size_lr 16 \
                --layer_shrinkage 1 \
                --noise_dim 100 \
                --out_act None \
                --method eng_2step \
                --norm_method_input normalise_scalar \
                --norm_method_output subtract_linear \
                --alpha_subtract_linear ${ALPHA} \
                --sqrt_transform_in \
                --sqrt_transform_out \
                --agg_norm_loss mean \
                --lambda_norm_loss_loc 1 \
                --p_norm_loss_loc 4 \
                --p_norm_loss_batch 4 \
                --save_model_every 100 \
                --num_epochs 1000 \
                --batch_size 1024 \
                > logs/log_gpu4-tasmax-coarse-${DOMAIN}-$(date +"%Y%m%d_%H%M%S").txt 2>&1 &

            # CUDA_VISIBLE_DEVICES=0 python train_only-coarse.py \
            #     --variables tasmax \
            #     --training_experiment ${EXPERIMENT} \
            #     --domain ${DOMAIN} \
            #     --variables_lr all \
            #     --save_name '_norm-in-normalise_scalar_nd-100_sorted-pred' \
            #     --num_layer 6 \
            #     --hidden_dim 100 \
            #     --preproc_layer \
            #     --preproc_dim 20 \
            #     --kernel_size_lr 16 \
            #     --layer_shrinkage 1 \
            #     --noise_dim 100 \
            #     --out_act None \
            #     --method eng_2step \
            #     --norm_method_input normalise_scalar \
            #     --norm_method_output normalise_pw \
            #     --sqrt_transform_in \
            #     --sqrt_transform_out \
            #     --agg_norm_loss mean \
            #     --lambda_norm_loss_loc 1 \
            #     --p_norm_loss_loc 4 \
            #     --p_norm_loss_batch 4 \
            #     --save_model_every 100 \
            #     --num_epochs 1000 \
            #     --batch_size 1024 \
            #     > logs/log_gpu4-tasmax-coarse-${DOMAIN}-$(date +"%Y%m%d_%H%M%S").txt 2>&1 &

            wait
        done
    done
done