
import os
import csv
import re
import pandas as pd
import numpy as np
from collections import defaultdict

# --- Configuration ---
LOG_BASE_DIR = 'experiment_logs'
MODEL_NAME = 'GPT3-7B'
OUTPUT_DIR = 'performance_models'
TARGET_KV_RANGE = range(0, 2049)
HARDWARE_NAME = 'neupims_off'

def parse_neupims_logs():
    """
    Parses the staged SA_*.tsv and PIM_*.tsv log files from the NeuPIMs
    (sub-batch OFF) experimental sweep. It aggregates cycle counts for each
    layer across all stages of a single forward pass.
    """
    # Initialize an empty list to collect data from all log files.
    all_data = []
    dir_pattern = re.compile(r'neupims_off_bs(\d+)_sl(\d+)')
    parsed_dirs_count = 0

    print(f"Scanning for '{HARDWARE_NAME}' log directories in '{LOG_BASE_DIR}'...")

    if not os.path.exists(LOG_BASE_DIR):
        print(f"Error: Log directory '{LOG_BASE_DIR}' not found.")
        return None

    for dir_name in os.listdir(LOG_BASE_DIR):
        if not dir_name.startswith(HARDWARE_NAME):
            continue

        match = dir_pattern.match(dir_name)
        if not match:
            continue

        batch_size, seq_len = map(int, match.groups())
        layer_total_cycles = defaultdict(int)
        experiment_dir = os.path.join(LOG_BASE_DIR, dir_name)

        # --- Corrected File Finding Logic ---
        # This block actively searches the directory for the required files.
        try:
            all_files_in_dir = os.listdir(experiment_dir)
        except FileNotFoundError:
            print(f"Warning: Directory not found: {experiment_dir}, skipping.")
            continue

        # This line now correctly filters the list to find only the files we care about.
        stage_files = [f for f in all_files_in_dir if (f.startswith('SA_') or f.startswith('PIM_')) and f.endswith('.tsv')]

        if not stage_files:
            print(f"Warning: No stage log files (SA_*.tsv, PIM_*.tsv) found in {dir_name}, skipping.")
            continue
        # --- End Corrected Logic ---

        for file_name in stage_files:
            file_path = os.path.join(experiment_dir, file_name)
            with open(file_path, 'r') as tsvfile:
                reader = csv.reader(tsvfile, delimiter='\t')
                try:
                    header = next(reader)
                    opname_idx = header.index('OpName')
                    totalcycle_idx = header.index('TotalCycle')
                except (StopIteration, ValueError):
                    continue # Skip empty or malformed files

                for row in reader:
                    if not row: continue # Skip empty rows
                    op_name = row[opname_idx]
                    total_cycles = int(row[totalcycle_idx])
                    # Sum the cycles for each operator across all stages (A, B, E, etc.)
                    layer_total_cycles[op_name] += total_cycles

        # Aggregate the data for this experiment, keeping the original detailed names for now.
        for op_name, total_cycles in layer_total_cycles.items():
            all_data.append({
                'hardware': HARDWARE_NAME, 'model': MODEL_NAME,
                'layer_name': op_name, 'input': batch_size,
                'kv_cache': seq_len, 'latency(ns)': total_cycles
            })
        
        parsed_dirs_count += 1

    if not all_data:
        print("Error: No valid log files were successfully parsed. Please double-check that the simulations completed successfully.")
        return None

    print(f"Successfully parsed data from {parsed_dirs_count} directories.")
    return pd.DataFrame(all_data)

def simplify_and_interpolate(df, hardware_name):
    """
    Simplifies layer names, aggregates data, performs linear interpolation,
    and saves the final CSV performance model.
    """
    print(f"\nProcessing and interpolating data for '{hardware_name}'...")
    
    if df is None or df.empty:
        print(f"No data to process for '{hardware_name}'.")
        return

    # --- Step 1: More Specific Layer Name Simplification ---
    def simplify_name(op_name):
        if 'NeuPIMS' in op_name:
            return 'attn' # Group all PIM operations into 'attn'
        
        parts = op_name.split('.')
        if len(parts) > 2:
            # Creates specific names like 'attn_res', 'ffn_ln', 'ffn_fc1'
            # This correctly separates the two different 'res' layers.
            return f"{parts[-2]}_{parts[-1]}"
        return parts[-1]

    df['simple_name'] = df['layer_name'].apply(simplify_name)

    # --- Step 2: Aggregate Data ---
    # Sum up latencies for layers that now have the same simple name.
    # This correctly combines 'NeuPIMSLogitSoftmax' and 'NeuPIMSAttend' into a single 'attn' value.
    agg_df = df.groupby(['hardware', 'model', 'simple_name', 'input', 'kv_cache'])['latency(ns)'].sum().reset_index()
    agg_df.rename(columns={'simple_name': 'layer_name'}, inplace=True)

    # --- Step 3: Interpolation ---
    unique_layers = agg_df['layer_name'].unique()
    unique_batch_sizes = sorted(agg_df['input'].unique())
    # Initialize an empty list to store the final, dense data.
    interpolated_rows = []

    for layer in unique_layers:
        for bs in unique_batch_sizes:
            subset_df = agg_df[(agg_df['layer_name'] == layer) & (agg_df['input'] == bs)].sort_values('kv_cache')

            if len(subset_df) < 2:
                interpolated_rows.extend(subset_df.to_dict('records'))
                continue

            interpolated_latencies = np.interp(
                TARGET_KV_RANGE,
                subset_df['kv_cache'],
                subset_df['latency(ns)']
            )

            for kv, lat in zip(TARGET_KV_RANGE, interpolated_latencies):
                interpolated_rows.append({
                    'hardware': hardware_name, 'model': MODEL_NAME,
                    'layer_name': layer, 'input': bs,
                    'kv_cache': kv, 'latency(ns)': int(round(lat))
                })

    if not interpolated_rows:
        print(f"Interpolation failed for '{hardware_name}'. No output generated.")
        return

    final_df = pd.DataFrame(interpolated_rows)
    output_filename = os.path.join(OUTPUT_DIR, f"{hardware_name}.csv")
    
    print(f"Saving final dense performance model to '{output_filename}'...")
    final_df.to_csv(output_filename, index=False)
    print(f"Successfully saved {len(final_df)} rows for {hardware_name}.")

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sparse_df = parse_neupims_logs()
    simplify_and_interpolate(sparse_df, HARDWARE_NAME)
    print("\nAggregation and interpolation for NeuPIMs complete.")