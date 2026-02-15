DOMAINS=(ALPS NZ SA)
EXPERIMENTS=(
    "ESD_pseudo_reality"
    "Emulator_hist_future"
)
TARGETVARS=(pr tasmax)

for DOMAIN in "${DOMAINS[@]}"; do
    for EXPERIMENT in "${EXPERIMENTS[@]}"; do
        for TARGETVAR in "${TARGETVARS[@]}"; do
            python linear_model_inference.py --domain ${DOMAIN} --target_var ${TARGETVAR} --training_experiment ${EXPERIMENT}
        done
    done
done