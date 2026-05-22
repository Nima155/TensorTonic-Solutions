import pandas as pd

def replace_values(data, column, old_val, new_val):
    """
    Returns: dict with 'data' (dict) and 'count' (int)
    """
    df = pd.DataFrame(data)
    sm = (df[column] == old_val).sum()
    df.loc[df[column] == old_val, column] = new_val
    return {
        "data": df.to_dict("list"),
        "count": sm
    }