import json
import numpy as np
import pandas as pd
from trident import Simulation
from utils import save_to_csv


#get values from params json
with open("./Trident/params.json", mode = "r", encoding = "utf-8") as f:
    params = json.load(f)
    timeseries_save_batch_size = params["random"]["timeseries_save_batch_size"]
    total_num_timeseries = params["random"]["total_num_timeseries"] #total number of reversal timeseries to generate
    total_num_timesteps = params["random"]["total_num_timesteps"]
    per_seed_num_timeseries = params["random"]["per_seed_num_timeseries"]
    dataset_name = params["random"]["dataset"]["name"]
    dataset_root_path = params["random"]["dataset"]["root_path"]
    epsilon = params["random"]["epsilon"]
    N_0_squared = params["random"]["N_0_squared"]
    r_m = params["random"]["r_m"]
    k = 2 * np.pi * 6
    m = 2 * np.pi * 3
    m_u = 2 * np.pi * 7
    dt = params["random"]["dt"]
    total_time = params["random"]["total_time"]
    initial_U = params["random"]["initial_U"]
    selected_features = params["random"]["selected_features"]





count_timeseries = 0
count_timeseries_per_seed = 0
seed_val = 0
start_idx = 0
dataset = []

while(count_timeseries < total_num_timeseries):
    print(f"==================== Random Seed: {seed_val} ====================")
    np.random.seed(seed_val)

    while(count_timeseries_per_seed < per_seed_num_timeseries):
        #create simulation instance
        sim = Simulation(
            epsilon = epsilon,
            N_0_squared = N_0_squared,
            r_m = r_m,
            k = k,
            m = m,
            m_u = m_u,
            dt = dt,
            total_time = total_time,
            randomness = True
        )
        # print("Simulation running...")
        #simulate data
        sim.simulate(
            phi_e = np.array([0.0, 0.0]),
            phi_plus = np.array([0.0, 0.0]),
            U = initial_U
        )

        #get all data
        all_data = []
        all_data.extend(sim.get_json_simulation_data(target_features = selected_features))
        save_to_csv(
                dataset = all_data,
                selected_features = selected_features,
                start_idx = 0,
                end_idx = 1,
                dataset_save_path = "C:/Users/Will6/OneDrive/Documents/Memorial University/Code/full_data.csv"
        )
        raise SystemExit

        #get the reversal data
        sim.extract_compressed_reversal_data(window_size = total_num_timesteps * 10 // 2) # *10: compression, //2: left & right
        num_reversals = len(sim.reversals["timesteps"]) #get number of reversals; equal to length of reversals[timesteps] length
        if(num_reversals == 0):
            continue

        #add reversal events to dataset
        dataset.extend(sim.get_json_reversal_data(target_features = selected_features))
        count_timeseries_per_seed += num_reversals
        print(f"Seed value: {seed_val} - {num_reversals} new timeseries")

    count_timeseries += count_timeseries_per_seed
    print(f"Seed value: {seed_val} - {count_timeseries_per_seed} total new timeseries")
    count_timeseries_per_seed = 0

    if((count_timeseries - start_idx) >= timeseries_save_batch_size):
        save_to_csv(
            dataset = dataset,
            selected_features = selected_features,
            start_idx = start_idx,
            end_idx = count_timeseries,
            dataset_save_path = f"{dataset_root_path}/{dataset_name}"
        )
        
        dataset = []
        start_idx = count_timeseries

    seed_val += 1

if(len(dataset) != 0):
    save_to_csv(
        dataset = dataset,
        selected_features = selected_features,
        start_idx = start_idx,
        end_idx = count_timeseries,
        dataset_save_path = f"{dataset_root_path}/{dataset_name}"
    )