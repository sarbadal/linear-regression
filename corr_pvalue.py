from scipy.stats import pearsonr
import numpy as np
import pandas as pd
    
def corr_pvalue(df):

    numeric_df = df.dropna()._get_numeric_data()
    cols = numeric_df.columns
    mat = numeric_df.values

    arr = np.zeros((len(cols),len(cols)), dtype=object)

    for xi, x in enumerate(mat.T):
        for yi, y in enumerate(mat.T[xi:]):
            arr[xi, yi+xi] = pearsonr(x,y)
            arr[yi+xi, xi] = arr[xi, yi+xi]

    df_tmp = pd.DataFrame(arr, index=cols, columns=cols)
    # df_tmp.reset_index(inplace=True)
    # df_tmp = df_tmp.rename(columns={'index': 'column'})
    return df_tmp