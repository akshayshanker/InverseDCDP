import time
import numpy as np
np.seterr(all='ignore') # ignoring all warnings
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, kurtosis
from G2EGM.G2EGMModel import G2EGMModelClass
import G2EGM.figs as figs
import G2EGM.simulate as simulate

# Timings plot 
def plot_timing_data(results, plot_path, NM_list, labels):
	"""
	Plots the timing data for both time iteration and Bellman iteration methods.
	
	"""
	sns.set(style="white", rc={
		"font.size": 14, "axes.titlesize": 14, "axes.labelsize": 14})
	

	palette = sns.color_palette("cubehelix", 3)
	palette1 = sns.color_palette("cubehelix", 5)
	color1 = palette[0] 
	color2 = palette[1]
	color3 = palette[2]
	palette[2] = palette1[0]
	markers = ['o', 'x', 'D']
	
	models = ['G2EGM', 'RFC', 'NEGM']
	modelsRHS = ['G2EGM_cons', 'RFC_cons']
	labels_cons = ['G2EGM', 'RFC with Delaunay']
	# Plotting
	#plt.figure(figsize=(8, 6))
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
	# fig with two cols 
	#ax1 = fig.add_subplot(1,2,1)
	j = 0

	for model, label in zip(models, labels):
		avg_time_iters = np.arange(0, len(NM_list)).astype(float)
		rfc_time_iters = np.arange(0, len(NM_list)).astype(float)
		#median_time_iters = np.arange(0, len(NM_list)).astype(float)
		grid_sizes = np.arange(0, len(NM_list)).astype(float)
		
		for i,Nm in zip(range(len(NM_list)),NM_list):
			grid_label = f'{Nm}'
			avg_time_iters[i] = results[Nm][model][0]['average_time_iter']
			grid_sizes[i] = results[Nm][model][0]['grid_size']
			if model == 'RFC':
				rfc_time_iters[i] = results[Nm][model][0]['avg_time_RFC'] 
			#rfc_time_iters[i] = results[Nm][model][0]['avg_time_RFC']

		if model == 'RFC':
			ax1.plot(grid_sizes[1:], rfc_time_iters[1:], label='RFC', linestyle='dashed', color=palette[j])

		ax1.plot(grid_sizes[1:], avg_time_iters[1:], label=label, marker=markers[j], linestyle='-', color=palette[j])
		ax1.set_xlabel('Number of grid points')
		ax1.set_ylabel('Average time (min.)')
		ax1.set_title('No pension cap - 4 cons. regions')
		ax1.set_ylim(0, 25)
		ax1.set_yticks(np.arange(0, 25, 5))
		#ax1.legend()
		ax1.grid(True)

	
		j += 1
	
	j = 0 
	for model, label in zip(modelsRHS, labels_cons):
		avg_time_iters = np.arange(0, len(NM_list)).astype(float)
		#median_time_iters = np.arange(0, len(NM_list)).astype(float)
		grid_sizes = np.arange(0, len(NM_list)).astype(float)
		
		for i,Nm in zip(range(len(NM_list)),NM_list):
			grid_label = f'{Nm}'
			avg_time_iters[i] = results[Nm][model][0]['average_time_iter']
			grid_sizes[i] = results[Nm][model][0]['grid_size']
			if model == 'RFC_cons':
				rfc_time_iters[i] = results[Nm][model][0]['avg_time_RFC'] 

		if model == 'RFC_cons':
			ax2.plot(grid_sizes[1:], rfc_time_iters[1:], linestyle='dashed', color=palette[j])
		
		ax2.plot(grid_sizes[1:], avg_time_iters[1:], marker=markers[j], linestyle='-', color=palette[j])
		ax2.set_xlabel('Number of grid points')
		ax2.set_ylabel('Average time (min.)')
		ax2.set_title('Pension cap - 6 cons. regions')
		ax2.set_ylim(0, 25)
		ax2.set_yticks(np.arange(0, 25, 5))
		# set horizontal grid lines
		#ax.
		#ax2.legend()
		ax2.grid(True)
		j += 1
	#
	
	#fig.subplots_adjust(bottom=.05)  # adjust as needed
	fig.tight_layout()
	fig.subplots_adjust(bottom=0.2)
	fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.1), shadow=False, ncol=4, frameon=False)
	
	
	# save to plot path
	#fig.subplots_adjust(bottom=0.2)  # adjust the bottom parameter as needed
	#fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), shadow=False, ncol=4)
	
	plt.savefig(plot_path + 'timings.png')

	# Euler errors 
	fig = plt.figure(figsize=(8, 6))
	ax = fig.add_subplot(1,1,1)
	j = 0
	for model, label in zip(models, labels):
		avg_euler = np.arange(0, len(NM_list)).astype(float)
		grid_sizes = np.arange(0, len(NM_list)).astype(float)
		for i,Nm in zip(range(len(NM_list)),NM_list):
			grid_label = f'{Nm}'
			avg_euler[i] = results[Nm][model][0]['average_mean_euler']
			grid_sizes[i] = results[Nm][model][0]['grid_size']
			## access results by converting the grid size to a string
			#print(f'{model} with grid size {Nm}: {avg_time_iters[i]:.2f} secs')

		print(avg_euler)
		ax.plot(grid_sizes[1:], avg_euler[1:], label=label, marker=markers[j], linestyle='-', color=palette[j])
		ax.set_xlabel('Exogenous grid size')
		ax.set_ylabel('Avg. lg of relative Euler error')
		ax.legend()
		ax.grid(True) 
		j += 1
	
	# save to plot path
	plt.savefig(plot_path + 'euler_errors.png')


# Table

def generate_table(results, NmList):
	# a. models and grid sizes
	models = list(results.values())

	# b. euler errors and timings for each grid size
	for Nm in NmList:
		postfix = f'_G2EGM_vs_NEGM_{Nm}'

		# b.1 euler errors
		lines = []
		for stat in ['All (average)', '5th percentile', '95th percentile']:
			txt = stat
			for model in models:
				for model_dict in model:
					if model_dict['grid_size'] == Nm:
						if stat == 'All (average)':
							txt += f' & {model_dict["average_mean_euler"]:.2f}'
						elif stat == '5th percentile':
							txt += f' & {model_dict["5th percentile"]:.2f}'
						elif stat == '95th percentile':
							txt += f' & {model_dict["95th percentile"]:.2f}'
						elif stat == 'Kurtosis':
							txt += f' & {model_dict["average_kurtosis_euler"]:.2f}'
						elif stat == '1s percentile':
							txt += f' & {model_dict["1st percentile"]:.2f}'
						elif stat == '99th percentile':
							txt += f' & {model_dict["99th percentile"]:.2f}'
						
						elif stat == 'RMSE':
							txt += f' & {model_dict["average_rmse_euler"]:.2f}'
			txt += '\\\\ \n'
			lines.append(txt)

		with open(f'plots/tabs_euler_errors{postfix}.tex', 'w') as txtfile:
			txtfile.writelines(lines)

		# b.2 timings
		lines = []
		for stat in ['Total', 'Post-decision functions', 'EGM-step','RFC-step','VFI-step']:
			txt = stat
			for model in models:
				for model_dict in model:
					print(model_dict)
					if model_dict['grid_size'] == Nm:
						if stat == 'Total':
							txt += f' & {model_dict["average_time_iter"]:.2f}'
						if stat == 'RFC-step':
							txt += f' & {model_dict["avg_time_RFC"]:.2f}'
						if stat == 'EGM-step':
							txt += f' & {model_dict["avg_time_EGM"]:.2f}'
			txt += '\\\\ \n'
			lines.append(txt)

		with open(f'plots/tabs_timings{postfix}.tex', 'w') as txtfile:
			txtfile.writelines(lines)
		

#Timings
def timing(model,
		   rep=1, # set to 5 in the paper
		   do_print=True):
	
	name = model.name
	par = model.par
	
	time_best = np.inf
	for i in range(rep):
		
		model.solve()
		model.calculate_euler()
			
		tot_time = np.sum(model.par.time_work)
		if do_print:
			print(f'{i}: {tot_time:.2f} secs, euler: {np.nanmean(model.sim.euler):.3f}')
			print(f'RMSE: {np.nanmean((model.sim.euler)**2)}')
			print(f'50th percentile: {np.nanpercentile(model.sim.euler,50)}')
			print(f'95th percentile: {np.nanpercentile(model.sim.euler,99)}')
			print(f'5th percentile: {np.nanpercentile(model.sim.euler,5)}')
			print(f'75th percentile: {np.nanpercentile(model.sim.euler,75)}')
			print(f'0.1th percentile: {np.nanpercentile(model.sim.euler,0.1)}')
			print(f'Kurtosis of Euler Errors: {kurtosis(model.sim.euler[~np.isnan(model.sim.euler)],nan_policy="omit")}')
			
		if tot_time < time_best:
			time_best = tot_time
			model_best = model.copy('best')
			
	model_best.name = name
	return model_best        


if __name__ == '__main__':

	from mpi4py import MPI
	import dill as pickle
	import sys

	from G2EGM.G2EGMModel import G2EGMModelClass
	import G2EGM.figs as figs
	import G2EGM.simulate as simulate
	from scipy.stats import norm, kurtosis
	import numba as nb
	nb.set_num_threads(1)
	import numpy as np

	# MPI communicators 
	rank = MPI.COMM_WORLD.Get_rank()
	solve = False
	size = MPI.COMM_WORLD.Get_size()
	
	# calcuate equal grid spacing of rank siz e between 100 and 1200
	gridSizeMax = 1000
	gridSizeMin = 200
	NmList = np.linspace(gridSizeMin, gridSizeMax, size, dtype=int)
	NmList = np.round(NmList / 50) * 50
	NmList = NmList.astype(int)
	Nm = NmList[rank]


	if solve == False:
		size = 24
		NmList = np.linspace(gridSizeMin, gridSizeMax, size, dtype=int)
		NmList = np.round(NmList / 50) * 50
		NmList = NmList.astype(int)
		Nm = NmList[rank]

	#Settings
	T = 15
	Neta = 16
	var_eta = 0.1**2
	do_print = False
	nameres = sys.argv[1]
	rep = 1

	# RFC timings that depend on the grid size
	rad = 300/Nm
	k = 70
	if Nm < 500:
		k = 85
	J_bar = 1 + 1E-05
	p_L = 1
	max_iter_rfc = 10
	s = 0.045

	if solve:

		#RFC baseline 
		model_RFC = G2EGMModelClass(name='RFC',\
												par={'solmethod':'RFC','T':T,\
														'do_print':do_print,\
														'k': k, 'Nm':Nm,\
														'rad':rad, 'rad_I':rad,\
														'J_bar': J_bar,\
														'k1':40,\
														'k2':1,\
														'intersection': False,\
														'interp_intersect': False,\
														's':s,\
														'max_iter': max_iter_rfc,\
														'n_closest': 2,\
														'nearest_fill': False,\
														'correct_jumps': True})
		model_RFC.precompile_numba()
		model_RFC = timing(model_RFC, rep=rep)

		#RFC with added constraint  
		model_RFC_cons = G2EGMModelClass(name='RFC_cons',\
													par={'solmethod':'RFC','T':T,\
														'do_print':do_print,\
														'k': k, 'Nm':Nm,\
														'rad':rad, 'rad_I':rad,\
														'J_bar': J_bar,\
														'k1':40,\
														'k2':1,\
														'intersection': False,\
														'interp_intersect': False,\
														's':s,\
														'max_iter': max_iter_rfc,\
														'n_closest': 2,\
														'nearest_fill': False,\
														'correct_jumps': True, 
														 'p_L': p_L})
		model_RFC_cons.precompile_numba()
		model_RFC_cons = timing(model_RFC_cons, rep=rep)

		#NEGM

		model_NEGM = G2EGMModelClass(name='NEGM',par={'solmethod':'NEGM',\
												'T':T,'do_print':do_print, 'Nm':Nm})
		model_NEGM.precompile_numba()
		model_NEGM = timing(model_NEGM, rep=1)

		#G2EGM baseline 
		model_G2EGM = G2EGMModelClass(name='G2EGM',par={'solmethod':'G2EGM',\
												  'T':T,'do_print':do_print, 'Nm':Nm})
		model_G2EGM.precompile_numba()
		model_G2EGM = timing(model_G2EGM, rep=1)

		# G2EGM with added constraint
		model_G2EGM_cons = G2EGMModelClass(name='G2EGM_cons',par={'solmethod':'G2EGM',\
															'T':T,'do_print':do_print, 'Nm':Nm, 'p_L': p_L})
		model_G2EGM_cons.precompile_numba()
		model_G2EGM_cons= timing(model_G2EGM_cons, rep=1)


		# gather the models on master
		# format the result on each rank to be consistent with pliot inpits and gathering
		models_RES = {}
		models_RES['G2EGM'] = [{'grid_size': model_G2EGM.par.Nm, 'average_time_iter': np.mean(model_G2EGM.par.time_work),\
								 'avg_time_EGM': np.mean(model_G2EGM.par.time_egm),\
								'average_mean_euler':np.nanmean(model_G2EGM.sim.euler), \
								'average_rmse_euler':np.nanmean((model_G2EGM.sim.euler)**2),\
								'average_kurtosis_euler':kurtosis(model_G2EGM.sim.euler[~np.isnan(model_G2EGM.sim.euler)],nan_policy="omit"),\
								'1st percentile':np.nanpercentile(model_G2EGM.sim.euler,1),\
								'99th percentile':np.nanpercentile(model_G2EGM.sim.euler,99),\
								'5th percentile':np.nanpercentile(model_G2EGM.sim.euler,5),\
								'95th percentile':np.nanpercentile(model_G2EGM.sim.euler,95),\
								'99.9th percentile':np.nanpercentile(model_G2EGM.sim.euler,99.9),\
								'0.1th percentile':np.nanpercentile(model_G2EGM.sim.euler,0.1),\
								'median_euler':np.nanmedian(model_G2EGM.sim.euler)}]
		
		models_RES['G2EGM_cons'] = [{'grid_size': model_G2EGM_cons.par.Nm, 'average_time_iter': np.mean(model_G2EGM_cons.par.time_work),\
								 'avg_time_EGM': np.mean(model_G2EGM_cons.par.time_egm),\
								'average_mean_euler':np.nanmean(model_G2EGM_cons.sim.euler), \
								'average_rmse_euler':np.nanmean((model_G2EGM_cons.sim.euler)**2),\
								'average_kurtosis_euler':kurtosis(model_G2EGM_cons.sim.euler[~np.isnan(model_G2EGM_cons.sim.euler)],nan_policy="omit"),\
								'1st percentile':np.nanpercentile(model_G2EGM_cons.sim.euler,1),\
								'99th percentile':np.nanpercentile(model_G2EGM_cons.sim.euler,99),\
								'5th percentile':np.nanpercentile(model_G2EGM_cons.sim.euler,5),\
								'95th percentile':np.nanpercentile(model_G2EGM_cons.sim.euler,95),\
								'99.9th percentile':np.nanpercentile(model_G2EGM_cons.sim.euler,99.9),\
								'0.1th percentile':np.nanpercentile(model_G2EGM_cons.sim.euler,0.1),\
								'median_euler':np.nanmedian(model_G2EGM_cons.sim.euler)}]
		
		
		models_RES['NEGM'] = [{'grid_size': model_NEGM.par.Nm, 'average_time_iter': np.mean(model_NEGM.par.time_work),\
								'average_mean_euler':np.nanmean(model_NEGM.sim.euler), \
								'average_rmse_euler':np.nanmean((model_NEGM.sim.euler)**2),\
								'average_kurtosis_euler':kurtosis(model_NEGM.sim.euler[~np.isnan(model_NEGM.sim.euler)],nan_policy="omit"),\
								'1st percentile':np.nanpercentile(model_NEGM.sim.euler,1),\
								'99th percentile':np.nanpercentile(model_NEGM.sim.euler,99),\
								'5th percentile':np.nanpercentile(model_NEGM.sim.euler,5),\
								'95th percentile':np.nanpercentile(model_NEGM.sim.euler,95),\
								'99.9th percentile':np.nanpercentile(model_NEGM.sim.euler,99.9),\
								'median_euler':np.nanmedian(model_NEGM.sim.euler),\
								'0.1th percentile':np.nanpercentile(model_NEGM.sim.euler,0.1)}]
		
		models_RES['RFC'] = [{'grid_size': model_RFC.par.Nm, 'average_time_iter': np.mean(model_RFC.par.time_work),\
								'avg_time_RFC': np.mean(model_RFC.par.time_rfc),\
								'avg_time_inversion': np.mean(model_RFC_cons.par.time_invert),\
								'average_mean_euler':np.nanmean(model_RFC.sim.euler), \
								'average_rmse_euler':np.nanmean((model_RFC.sim.euler)**2),\
								'average_kurtosis_euler':kurtosis(model_RFC.sim.euler[~np.isnan(model_RFC.sim.euler)],nan_policy="omit"),\
								'1st percentile':np.nanpercentile(model_RFC.sim.euler,1),\
								'99th percentile':np.nanpercentile(model_RFC.sim.euler,99),\
								'5th percentile':np.nanpercentile(model_RFC.sim.euler,5),\
								'99.9th percentile':np.nanpercentile(model_RFC.sim.euler,99.9),\
								'0.1th percentile':np.nanpercentile(model_RFC.sim.euler,0.1),\
								'median_euler':np.nanmedian(model_RFC.sim.euler),\
								'95th percentile':np.nanpercentile(model_RFC.sim.euler,95)}]
		
		models_RES['RFC_cons'] = [{'grid_size': model_RFC_cons.par.Nm, 'average_time_iter': np.mean(model_RFC_cons.par.time_work),\
							      'avg_time_RFC': np.mean(model_RFC_cons.par.time_rfc),\
								  'avg_time_inversion': np.mean(model_RFC_cons.par.time_invert),\
									'average_mean_euler':np.nanmean(model_RFC_cons.sim.euler), \
									'average_rmse_euler':np.nanmean((model_RFC_cons.sim.euler)**2),\
									'average_kurtosis_euler':kurtosis(model_RFC_cons.sim.euler[~np.isnan(model_RFC_cons.sim.euler)],nan_policy="omit"),\
									'1st percentile':np.nanpercentile(model_RFC_cons.sim.euler,1),\
									'99th percentile':np.nanpercentile(model_RFC_cons.sim.euler,99),\
									'5th percentile':np.nanpercentile(model_RFC_cons.sim.euler,5),\
									'95th percentile':np.nanpercentile(model_RFC_cons.sim.euler,95),\
									'99.9th percentile':np.nanpercentile(model_RFC_cons.sim.euler,99.9),\
									'0.1th percentile':np.nanpercentile(model_RFC_cons.sim.euler,0.1),\
									'median_euler':np.nanmedian(model_RFC_cons.sim.euler)}]
		
		
		models_RES['grid_size'] = Nm

		# plot histogram of RFC vs G2EGM 
		palette = sns.color_palette("cubehelix", 3)
		color1 = palette[0] 
		color2 = palette[1]
		color3 = palette[2]
		fig = plt.figure()
		ax = fig.add_subplot(1,1,1)
		errors_g2egm = np.ravel(model_G2EGM.sim.euler)
		errors_rfc = np.ravel(model_RFC.sim.euler)
		ax.hist(errors_g2egm,bins=50, label = 'G2EGM', alpha = 0.75, color=color1,density=True)
		ax.hist(errors_rfc,bins=50, label = 'RFC', alpha = 0.75, color=color2,density=True)
		ax.set_xlabel('Log10 rel. Euler error')
		ax.legend()
		ax.grid(True)
		plt.savefig('plots/pensions/euler_hist_{}_timings.png'.format(Nm))


		MPI.COMM_WORLD.barrier()
		modelsAll = MPI.COMM_WORLD.gather(models_RES,root=0)

		# collect the timings data to plot 
		if rank == 0:
			# save results
			pickle.dump(modelsAll, open('plots/pensions/results_{}.pkl'.format(nameres), 'wb'))
			results = {}
			
			for rank, modelres in zip(range(len(modelsAll)),modelsAll):
					results[NmList[rank]] = modelres

			
			plot_timing_data(results, 'plots/pensions/', NmList, labels)
	else:
		if rank == 0:
			modelsAll = pickle.load(open('plots/pensions/results_{}.pkl'.format(nameres), 'rb'))
			results = {}
			labels = ['G2EGM', 'RFC w. Delaunay', 'NEGM']
			for j, modelres in zip(range(len(modelsAll)),modelsAll):
					results[NmList[j]] = modelres
			plot_timing_data(results, 'plots/pensions/', NmList, labels)