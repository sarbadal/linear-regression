import pandas as pd
import statsmodels.api as smf

def stepwise_selection(data, y, Intercept=True, initial_list=[], threshold_in=0.01, threshold_out=0.05, verbose=True):
    """ Perform a forward-backward feature selection 
    based on p-value from statsmodels.api.ols
    Arguments:
        X - pandas.DataFrame with candidate features
        y - list-like with the target
        initial_list - list of features to start with (column names of X)
        threshold_in - include a feature if its p-value < threshold_in
        threshold_out - exclude a feature if its p-value > threshold_out
        verbose - whether to print the sequence of inclusions and exclusions
    Returns: list of selected features 
    Always set threshold_in < threshold_out to avoid infinite looping.
    """
    X = data.columns.tolist()
    X.remove(y)
    str_gap = max([len(c) for c in X])+2
    included = list(initial_list)
    x_vars = data.copy()

    while True:
        changed=False
        
        # forward step
        excluded = list(set(X)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            if Intercept==True: formula = "{} ~ {} + 1".format(y, ' + '.join(included+[new_column]))
            else: formula = "{} ~ {} + 0".format(y, ' + '.join(included+[new_column])) 
            model = smf.ols(formula, data=x_vars).fit()
            new_pval[new_column] = model.pvalues[new_column]

        best_pval = new_pval.min()
        if best_pval <= threshold_in:
            best_feature = new_pval.argmin()
            included.append(best_feature)
            changed=True
            if verbose:
                print('Add  {:<{width}} with p-value {:.6}'.format(best_feature, best_pval, width=str_gap))

        # backward step
        if Intercept==True: formula = "{} ~ {} + 1".format(y, ' + '.join(included))
        else: formula = "{} ~ {} + 0".format(y, ' + '.join(included))
        model = smf.ols(formula, data=x_vars).fit()

        if Intercept==True: 
            # use all coefs except intercept
            pvalues = model.pvalues.iloc[1:]
        else: pvalues = model.pvalues
        worst_pval = pvalues.max() # null if pvalues is empty
        if worst_pval > threshold_out:
            changed=True
            worst_feature = pvalues.argmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:<{width}} with p-value {:.6}'.format(worst_feature, worst_pval, width=str_gap))
        if not changed:
            break
    if Intercept==True: formula = "{} ~ {} + 1".format(y, ' + '.join(included))
    else: formula = "{} ~ {} + 0".format(y, ' + '.join(included))
    model = smf.ols(formula, data=x_vars).fit()
    print('')
    print(model.summary())
    return [included, model]
