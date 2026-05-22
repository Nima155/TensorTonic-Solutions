import pandas as pd

def handle_missing(data, fill_value):
    """
    Returns: dict with 'null_counts' (dict) and 'cleaned_data' (dict)
    """
    df = pd.DataFrame(data)
    counts = df.isna().sum(axis=0)
    df = df.fillna(fill_value)
    return {
        "null_counts": counts.to_dict(),
        "cleaned_data": df.to_dict("list")
    }