import pandas as pd

def change_dtype(data, column, target_type):
    """
    Returns: list [dtypes_before, dtypes_after] (both dicts)
    """
    df = pd.DataFrame(data)
    before = df.dtypes
    before = {k: v.name for v, k in list(zip(df.dtypes.to_list(), df.columns))}
    df[column] = df[column].astype(target_type)
    after = {k: v.name for v, k in list(zip(df.dtypes.to_list(), df.columns))}
    # print(list(zip(df.dtypes.to_list())))
    return [before, after]