import pandas as pd

def groupby_basics(data, group_col, value_col):
    """
    Returns: dict with 'sum', 'mean', 'count' (each a dict)
    """
    df = pd.DataFrame(data).groupby(group_col)[value_col].agg(["sum", "mean", "count"])
    return df.to_dict("dict")