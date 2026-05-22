import pandas as pd
from functools import reduce
def concat_dataframes(dfs):
    """
    Returns: list [shape, data] where shape is [rows, cols]
    """
    dfs = [pd.DataFrame(v) for v in dfs]
    res = pd.concat(dfs)
    return [res.shape, res.to_dict("list")]