import pandas as pd

def rename_columns(data, rename_map):
    """
    Returns: dict mapping renamed column names to value lists
    """
    df = pd.DataFrame(data)
    
    return df.rename(rename_map, axis=1).to_dict("list")