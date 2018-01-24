import statsmodels.formula.api as smf
import pandas as pd
import numpy as np

def vif_cal(input_data, dependent_col):
    """ Code for VIF Calculation. Writing a function to calculate the VIF values """
    
    x_vars=input_data.drop([dependent_col], axis=1)
    xvar_names=x_vars.columns.tolist()
    x_var_col, vif_list = [], []
    str_gap = max([len(c) for c in xvar_names])+2

    # print("{:*^20s}".format("VIF Summary"))
    str_len = str_gap + 2 + 7 + 3 + 6 - len(' VIF Summary ')
    star_str = '*'*int(str_len/2)
    str_to_print = ''.join((star_str,' VIF Summary ',star_str))
    print(str_to_print)

    for xvar in xvar_names:
        y=xvar 
        x=xvar_names.copy()
        x.remove(xvar)

        formula = "{} ~ {} + 1".format(y, ' + '.join(x))
        rsq=smf.ols(formula, data=x_vars).fit().rsquared  
        if rsq==1: vif=np.inf
        else: vif=round(1/(1-rsq),10)
        x_var_col.append(xvar)
        vif_list.append(vif)
        print('vif of {:<{width}} = {:.6}'.format(xvar, vif, width=str_gap))

    str_len = str_gap + 2 + 7 + 3 + 6 - len(' VIF Summary END ')
    star_str = '*'*int(str_len/2)
    str_to_print = ''.join((star_str,' VIF Summary END ',star_str))
    print(str_to_print)

    vif_df = pd.DataFrame({'x_variable': x_var_col, 'vif': vif_list})
    vif_df = vif_df[['x_variable', 'vif']]
    return vif_df