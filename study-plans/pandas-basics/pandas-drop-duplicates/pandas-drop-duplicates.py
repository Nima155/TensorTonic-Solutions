import pandas as pd

def drop_duplicates(data):
    """
    Returns: list [rows_before, rows_after, cleaned_data]
    """
    df = pd.DataFrame(data)
    rows_before = df.shape[0]
    after = df.drop_duplicates()
    rows_after = after.shape[0]
    return [rows_before, rows_after, after.to_dict("list")]