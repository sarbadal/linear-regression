from linearreg import forward_selection_lreg as fs
from linearreg import backward_elimination_lreg as be
from linearreg import stepwise_selection_lreg as ss
from linearreg import corr_pvalue as corrp
from linearreg import vif_lreg as vif

def linearreg(data, y, Intercept=True, regtype='stepwise',
			  initial_list=[], threshold_in=0.01, threshold_out=0.05, 
			  verbose=True, vif=False, corr=False):

	"""
	Parameters:
	data : pandas.DataFrame with candidate features and y
	y : dependent variable
	regtype : ['forward', 'stepwise', 'backward']
	intial_list, threshold_in, threshold_out : optional only for stepwise selection process
	verbose : to print details
	vif : for vif data
	corr : for corr data

	Outputs:
	produce a list of reg model, selected variables, vif data and corr data
	"""

	if vif: 
		vif_data = vif.vif_cal(data=train[xy_col], y=y_var)
	else: 
		vif_data = 'vif option had not been selected'

	if corr: 
		corr_data = corr.corr_pvalue(data)
	else:
		corr_data = 'corr option had not been selected'

	if regtype == 'forward':
		final_var, model = fs.forward_selection(data=data, y=y, Intercept=Intercept, verbose=verbose)
	if regtype == 'backward':
		final_var, model = be.backward_elimination(data=data, y=y, Intercept=Intercept, threshold_out=threshold_out, verbose=verbose)
	if regtype == 'stepwise':
		final_var, model = ss.stepwise_selection(data=data, y=y, Intercept=Intercept, verbose=verbose, initial_list=initial_list, threshold_in=threshold_in, threshold_out=threshold_out)

	return [model, final_var, vif_data, corr_data]





