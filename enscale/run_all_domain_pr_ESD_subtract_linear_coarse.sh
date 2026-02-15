#!/bin/bash

# List of domains
DOMAINS=(SA NZ)

# Loop over orography options
# "" = no orography, "--add_orography" = with orography
ORO_OPTIONS=("")

for DOMAIN in "${DOMAINS[@]}"; do
    for ORO_FLAG in "${ORO_OPTIONS[@]}"; do
        if [[ -z "$ORO_FLAG" ]]; then
            echo "Starting domain ${DOMAIN} WITHOUT orography"
        else
            echo "Starting domain ${DOMAIN} WITH orography"
        fi

        if [[ "$DOMAIN" == "ALPS" ]]; then
            ALPHA=100
            ALPHA_STRING=""
        else
            ALPHA=1000
            ALPHA_STRING="_alpha-1000"
        fi

        # CUDA_VISIBLE_DEVICES=0 python train_only-super_multivariate.py \
        # --training_experiment ESD_pseudo_reality \
        # --variables pr \
        # --domain ${DOMAIN} \
        # $ORO_FLAG \
        # --variables_lr all \
        # --out_act None \
        # --save_name "_dec0_lam-mse0_split-residTrue${ALPHA_STRING}" \
        # --lambda_mse_loss 0 \
        # --weight_decay 0 \
        # --nicolai_layers \
        # --num_neighbors_res 25 \
        # --mlp_depth 2 \
        # --noise_dim_mlp 0 \
        # --hidden_dim 12 \
        # --latent_dim 12 \
        # --method eng_2step \
        # --sqrt_transform_in \
        # --sqrt_transform_out \
        # --norm_method_input None \
        # --norm_method_output subtract_linear \
        # --alpha_subtract_linear ${ALPHA} \
        # --sample_every_nepoch 100 \
        # --save_model_every 100 \
        # --kernel_size_hr 1 \
        # --kernel_size_lr 2 \
        # --num_epochs 1000 \
        # --batch_size 128 \
        # > logs/log_gpu0-pr-${DOMAIN}-$(date +"%Y%m%d_%H%M%S").txt 2>&1 &

        # CUDA_VISIBLE_DEVICES=0 python train_only-super_multivariate.py \
        # --training_experiment ESD_pseudo_reality \
        # --variables pr \
        # --domain ${DOMAIN} \
        # $ORO_FLAG \
        # --variables_lr all \
        # --out_act None \
        # --save_name "_dec0_lam-mse0_split-residTrue${ALPHA_STRING}" \
        # --lambda_mse_loss 0 \
        # --weight_decay 0 \
        # --nicolai_layers \
        # --num_neighbors_res 25 \
        # --mlp_depth 2 \
        # --noise_dim_mlp 0 \
        # --hidden_dim 12 \
        # --latent_dim 12 \
        # --method eng_2step \
        # --sqrt_transform_in \
        # --sqrt_transform_out \
        # --norm_method_input None \
        # --norm_method_output subtract_linear \
        # --alpha_subtract_linear ${ALPHA} \
        # --sample_every_nepoch 200 \
        # --save_model_every 200 \
        # --kernel_size_hr 2 \
        # --kernel_size_lr 4 \
        # --num_epochs 2000 \
        # --batch_size 256 \
        # > logs/log_gpu1-pr-${DOMAIN}-$(date +"%Y%m%d_%H%M%S").txt 2>&1 &

        # CUDA_VISIBLE_DEVICES=0 python train_only-super_multivariate.py \
        # --training_experiment ESD_pseudo_reality \
        # --variables pr \
        # --domain ${DOMAIN} \
        # $ORO_FLAG \
        # --variables_lr all \
        # --out_act None \
        # --save_name "_dec0_lam-mse0_split-residTrue${ALPHA_STRING}" \
        # --lambda_mse_loss 0 \
        # --weight_decay 0 \
        # --nicolai_layers \
        # --num_neighbors_res 25 \
        # --mlp_depth 2 \
        # --noise_dim_mlp 0 \
        # --hidden_dim 12 \
        # --latent_dim 12 \
        # --method eng_2step \
        # --sqrt_transform_in \
        # --sqrt_transform_out \
        # --norm_method_input None \
        # --norm_method_output subtract_linear \
        # --alpha_subtract_linear ${ALPHA} \
        # --sample_every_nepoch 200 \
        # --save_model_every 200 \
        # --kernel_size_hr 4 \
        # --kernel_size_lr 8 \
        # --num_epochs 5000 \
        # --batch_size 512 \
        # > logs/log_gpu2-pr-${DOMAIN}-$(date +"%Y%m%d_%H%M%S").txt 2>&1 &

        # CUDA_VISIBLE_DEVICES=0 python train_only-super_multivariate.py \
        # --training_experiment ESD_pseudo_reality \
        # --variables pr \
        # --domain ${DOMAIN} \
        # $ORO_FLAG \
        # --variables_lr all \
        # --out_act None \
        # --save_name "_dec0_lam-mse0_split-residTrue${ALPHA_STRING}" \
        # --lambda_mse_loss 0 \
        # --weight_decay 0 \
        # --nicolai_layers \
        # --num_neighbors_res 25 \
        # --mlp_depth 2 \
        # --noise_dim_mlp 0 \
        # --hidden_dim 12 \
        # --latent_dim 12 \
        # --method eng_2step \
        # --sqrt_transform_in \
        # --sqrt_transform_out \
        # --norm_method_input None \
        # --norm_method_output subtract_linear \
        # --alpha_subtract_linear ${ALPHA} \
        # --sample_every_nepoch 200 \
        # --save_model_every 1000 \
        # --kernel_size_hr 8 \
        # --kernel_size_lr 16 \
        # --num_epochs 10000 \
        # --batch_size 1024 \
        # > logs/log_gpu3-pr-${DOMAIN}-$(date +"%Y%m%d_%H%M%S").txt 2>&1 &

        if [[ -n "$ORO_FLAG" ]]; then
            echo "Orography is enabled"
        else
            CUDA_VISIBLE_DEVICES=3 python train_only-coarse.py \
            --training_experiment ESD_pseudo_reality \
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
        fi
        wait
        echo "✅ Finished domain: ${DOMAIN} with orography: ${ORO_FLAG}"
    done
done