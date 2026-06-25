import pandas as pd
import os

def save_to_csv(dataset, selected_features, start_idx, end_idx, dataset_save_path):
    print(f"Saving {len(dataset)} new data points to file...")

    df = pd.DataFrame(dataset, columns = selected_features)
    df.index = range(start_idx, end_idx)
    df.index.name = "id"

    if(os.path.exists(dataset_save_path)):
        df.to_csv(dataset_save_path, index = True, header = False, mode = "a")
        return

    df.to_csv(dataset_save_path, index = True, header = True, mode = "a")