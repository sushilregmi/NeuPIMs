#!/bin/bash

# --- Configuration ---
# These are the same experimental parameters as the npu-only sweep.
BATCH_SIZES=(64 128 256 512)
SEQ_LENS=(128 256 512 1024 2048)

# Define the path to the system configuration file we will be modifying.
# This now points to the config that enables sub-batching for NeuPIMs.
SYS_CONFIG_FILE="./configs/system_configs/sub-batch-off.json"

# --- Script Logic ---



# Loop through each defined batch size.
for bs in "${BATCH_SIZES[@]}"; do
  for sl in "${SEQ_LENS[@]}"; do
    echo "-----------------------------------------------------"
    echo "Running NEUPIMS experiment: Batch Size=${bs}, Sequence Length=${sl}"
    echo "-----------------------------------------------------"

    # --- Step 1: Update the system configuration file ---
    # Use jq to modify the JSON file for the current batch size.
    # We set max_active_reqs slightly higher as seen in the example config.
    jq ".max_batch_size = ${bs} |.max_active_reqs = ($bs + 2)" "$SYS_CONFIG_FILE" > "${SYS_CONFIG_FILE}.tmp" && mv "${SYS_CONFIG_FILE}.tmp" "$SYS_CONFIG_FILE"
    
    echo "Updated ${SYS_CONFIG_FILE} with batch size ${bs}."

    # --- Step 2: Define paths for this specific run ---
    # We use a different directory name to keep these results separate from the npu-only run.
    LOG_DIR="experiment_logs/neupims_off_bs${bs}_sl${sl}"
    CLI_CONFIG="./request-traces/uniform/requests_bs${bs}_sl${sl}.csv"

    # --- Step 3: Run the simulator with the updated configs ---
    mkdir -p "$LOG_DIR"
    
    # This command now uses the correct memory and system configs for NeuPIMs.
  ./build/bin/Simulator \
        --config ./configs/systolic_ws_128x128_dev.json \
        --mem_config ./configs/memory_configs/neupims.json \
        --cli_config "$CLI_CONFIG" \
        --model_config ./configs/model_configs/gpt3-7B.json \
        --sys_config "$SYS_CONFIG_FILE" \
        --log_dir "$LOG_DIR"
  done
done

echo "====================================================="
echo "Experimental sweep for NeuPIMs (main branch) is complete."
echo "====================================================="