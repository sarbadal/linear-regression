import pandas as pd
import statsmodels.formula.api as smf

def backward_elimination(data, y, Intercept=True, threshold_out=0.05, verbose=True):
    """ Perform a forward-backward feature selection 
    based on p-value from statsmodels.api.ols
    Arguments:
        X - pandas.DataFrame with candidate features and y
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
    x_vars = data.copy()

    # setting the first line of summary print...
    str_len = str_gap + 17 - len(' Backward Elimination Summary ')
    star_str = '='*int(str_len/2)
    str_to_print = ''.join((star_str,' Backward Elimination Summary ',star_str))
    if verbose: print('')
    if verbose: print(str_to_print)


    current_x_list = X.copy()
    sl_no = 0

    while len(current_x_list)>1:

        if Intercept==True: formula = "{} ~ {} + 1".format(y, ' + '.join(current_x_list))
        else: formula = "{} ~ {} + 0".format(y, ' + '.join(current_x_list)) 
        model = smf.ols(formula, data=x_vars).fit()

        if Intercept==True: 
            pvalues_x = model.pvalues
            pvalues_x = pvalues_x.drop(['Intercept'], axis=0)
            max_pvalue = max(pvalues_x.values)
        else:
            pvalues_x = model.pvalues 
            max_pvalue = max(pvalues_x.values)
        
        if max_pvalue > threshold_out: 
            current_x_list.remove((pvalues_x==max_pvalue).argmax())
            if verbose: # variable add print 
                if sl_no==0:
                    print('Sl.No. {:<{width}} P>|t|'.format('Variable Dropped', width=str_gap))
                    print('-'*int(str_gap+17))

            sl_no += 1
            print('{:<6} {:<{width}} {:.6}'.format(sl_no, (pvalues_x==max_pvalue).argmax(), max_pvalue, width=str_gap))
        else:
            break

    # setting the first line of summary print...
    str_len = str_gap + 17 - len(' Backward Elimination Summary End ')
    star_str = '='*int(str_len/2)
    str_to_print = ''.join((star_str,' Backward Elimination Summary End ',star_str))
    if verbose: print(str_to_print)


    if Intercept==True: formula = "{} ~ {} + 1".format(y, ' + '.join(current_x_list))
    else: formula = "{} ~ {} + 0".format(y, ' + '.join(current_x_list))
    model = smf.ols(formula, data=x_vars).fit()
    print('')
    print(model.summary())
    return [current_x_list, model]
