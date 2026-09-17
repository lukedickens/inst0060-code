import numpy as np
import pandas as pd
import csv


def import_1d_regression_data(
        ifname, input_col=0, target_col=1, **kwargs):
    """
    Imports 1d regression data (univariate input and target) from a
     tab/comma/semi-colon/... separated data file.

    parameters
    ----------
    ifname -- filename/path of data file.
    input_col -- the index of column used for inputs
    target_col -- the index of column used for targets
    <other keyword arguments supported by pandas read_csv function>
        See the use of **kwargs

    returns
    -------
    inputs -- input values (1d array)  
    targets -- target values (1d array)
    """
    # a dataframe object with all data from file
    df = pd.read_csv(ifname, **kwargs)
    # extract column index input_col as inputs
    inputs = df.iloc[:,input_col].to_numpy()
    # extract column index target_col as targets
    targets = df.iloc[:,target_col].to_numpy()
    return inputs, targets


