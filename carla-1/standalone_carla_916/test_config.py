import pickle
import os

config_path = '/storage/training_logs/cshenron_town01_radar_v1/config.pickle'
if not os.path.exists(config_path):
    print(f"File not found: {config_path}")
else:
    with open(config_path, 'rb') as f:
        config = pickle.load(f)
        
    print(f"use_wp_gru: {getattr(config, 'use_wp_gru', 'MISSING')}")
    print(f"use_controller_input_prediction: {getattr(config, 'use_controller_input_prediction', 'MISSING')}")
    print(f"inference_direct_controller: {getattr(config, 'inference_direct_controller', 'MISSING')}")
    print(f"ignore_index: {getattr(config, 'ignore_index', 'MISSING')}")
