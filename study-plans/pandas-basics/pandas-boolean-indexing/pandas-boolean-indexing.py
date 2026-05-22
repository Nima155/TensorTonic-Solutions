import pandas as pd

def boolean_filter(data, column, threshold):
    """
    Returns: dict with 'filtered_data' (dict) and 'count' (int)
    """
    df = pd.DataFrame(data)
    ret = df.loc[df[column] > threshold]
    # print(ret)
    return {
        "filtered_data": ret.to_dict("list"),
        "count": ret.shape[0]
    }