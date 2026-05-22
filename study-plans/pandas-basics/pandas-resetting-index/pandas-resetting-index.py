import pandas as pd

def reset_index_demo(data, index_col):
    """
    Returns: list [columns_before_reset, columns_after_reset]
    """
    cols = []
    df = pd.DataFrame(data)
    df.set_index(index_col, inplace=True)
    cols.append(list(df.columns))
    df.reset_index(inplace=True)
    cols.append(list(df.columns))
    return cols