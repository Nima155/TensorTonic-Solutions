import pandas as pd
from collections import Counter
def data_types_overview(data):
    """
    Returns: dict with 'dtypes', 'type_counts', 'num_columns'
    """
    df = pd.DataFrame(data)
    return {"dtypes": {k: v.name for k, v in df.dtypes.to_dict().items()},
            "type_counts": Counter({k: v.name for k, v in df.dtypes.to_dict().items()}.values()), "num_columns": len(df.columns)}