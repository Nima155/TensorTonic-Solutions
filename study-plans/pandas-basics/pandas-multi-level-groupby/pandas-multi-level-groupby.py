import pandas as pd

def multi_groupby(data, group_cols, value_col, aggfunc):
    """
    Returns: dict of lists (flat table with group columns + value column)
    """
    df = pd.DataFrame(data)

    s1 = df.groupby(group_cols)[value_col].agg(aggfunc)
    ret = {
        v: [z[i] for z in s1.index.to_list()] for i, v in enumerate(group_cols)
    }
    # print(df)
    ret[value_col] = s1.values

    return ret