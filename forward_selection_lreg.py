import statsmodels.formula.api as smf

def forward_selection(data, y, Intercept=True, verbose=True):
    """
    Linear model designed by forward selection.
    Parameters:
    -----------
    data : pandas DataFrame with all possible predictors and y
    y : string, name of y column in data
    Intercept : bool parameter. Whether we want Intercept in regression model or not
    verbose : bool parameter. if True the it will print out the steps of adding variables

    Returns:
    --------
    model: an "optimal" fitted statsmodels linear model
           with an intercept
           selected by forward selection
           evaluated by adjusted R-squared
    """

    step = 1
    result = []
    reg_df_tmp = data.copy()   
    remaining = set(reg_df_tmp.columns)
    str_gap = max([len(c) for c in list(remaining)])+2
    
    # setting the first line of summary print...
    str_len = str_gap + 22 - len(' Forward Selection Summary ')
    star_str = '='*int(str_len/2)
    str_to_print = ''.join((star_str,' Forward Selection Summary ',star_str))
    if verbose: print('')
    if verbose: print(str_to_print)

    remaining.remove(y)
    selected = []
    current_score, best_new_score = 0.0, 0.0
    sl_no = 0

    while remaining and current_score == best_new_score:
        scores_with_candidates = []

        for candidate in remaining:
            if Intercept==True: formula = "{} ~ {} + 1".format(y, ' + '.join(selected + [candidate]))
            else: formula = "{} ~ {} + 0".format(y, ' + '.join(selected + [candidate]))
            score = smf.ols(formula, reg_df_tmp).fit().rsquared_adj
            scores_with_candidates.append((score, candidate))
        scores_with_candidates.sort()
        best_new_score, best_candidate = scores_with_candidates.pop()

        if current_score <= best_new_score:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score

            if verbose: # variable add print 
                if sl_no==0:
                    print('Sl.No. {:<{width}} Adj. R-squared'.format('Variable Added', width=str_gap))
                    print('-'*int(str_gap+22))
                
                sl_no += 1
                print('{:<6} {:<{width}} {:.6}'.format(sl_no, best_candidate, best_new_score, width=str_gap))
    
    # setting the last line of summary print...
    str_len = str_gap + 22 - len(' Forward Selection End ')
    star_str = '='*int(str_len/2)
    str_to_print = ''.join((star_str,' Forward Selection End ',star_str))
    if verbose: print(str_to_print)

    if Intercept==True: formula = "{} ~ {} + 1".format(y, ' + '.join(selected))
    else: formula = "{} ~ {} + 0".format(y, ' + '.join(selected))
    model = smf.ols(formula, reg_df_tmp).fit()
    result.append([selected, model.rsquared_adj, model])
    step += 1
    if verbose: print('')
    if verbose: print(model.summary())
    del reg_df_tmp
    return [selected, model]