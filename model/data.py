"""
The file that stores functions related to the modification/reading of the data
"""

import pandas as pd

def num_categories(df: pd.DataFrame) -> int:
    """
    Returns the number of categories in the dataframe.
    """
    return len(df['category'].unique())