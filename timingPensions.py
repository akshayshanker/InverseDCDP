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
		"font.size": 11, "axes.titlesize": 11, "axes.labelsize": 11})
	

	palette = sns.color_palette("cubehelix", 3)
	palette1 = sns.color_palette("cubehelix", 5)
	color1 = palette[0] 
	color2 = palette[1]
	color3 = palette[2]
	palette[2] = palette1[0]
	markers = ['o', 'x', 'D']
	
	models = ['G2EGM', 'RFC', 'NEGM']
	# Plotting
	plt.figure(figsize=(8, 6))
	fig = plt.figure(figsize=(8, 6))
	ax = fig.add_subplot(1,1,1)
	j = 0
	for model, label in zip(models, labels):
		avg_time_iters = np.arange(0, len(NM_list)).astype(float)
		#median_time_iters = np.arange(0, len(NM_list)).astype(float)
		grid_sizes = np.arange(0, len(NM_list)).astype(float)
		
		for i,Nm in zip(range(len(NM_list)),NM_list):
			grid_label = f'{Nm}'
			avg_time_iters[i] = results[Nm][model][0]['average_time_iter']
			grid_sizes[i] = results[Nm][model][0]['grid_size']

		ax.plot(grid_sizes[1:], avg_time_iters[1:], label=label, marker=markers[j], linestyle='-', color=palette[j])
		ax.set_xlabel('Exogenous grid size')
		ax.set_ylabel('Average time (min.)')
		ax.legend()
		ax.grid(True)

		j += 1
	# save to plot path
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


	rank = MPI.COMM_WORLD.Get_rank()
	solve = False
	size = MPI.COMM_WORLD.Get_size()
	# calcuate equal grid spacing of rank siz e between 100 and 1200
	#gridSizeMax = 1200
	#gridSizeMin = 200
	#NmList = np.linspace(gridSizeMin, gridSizeMax, size, dtype=int)
	##NmList = np.round(NmList / 50) * 50
	#NmList = NmList.astype(int)
   
	NmList = np.arange(200, 900, 30)
	Nm = NmList[rank]

	#if solve == False:
		#size = 24
		#NmList = np.linspace(gridSizeMin, gridSizeMax, size, dtype=int)
		#NmList = np.round(NmList / 50) * 50
		#NmList = NmList.astype(int)
		#Nm = NmList[rank]

	#Settings
	T = 20
	Neta = 16
	var_eta = 0.1**2
	do_print = False
	nameres = sys.argv[1]

	# Build all the models 
	#RFC
	rad = 300/Nm

	if solve:
		model_RFC = G2EGMModelClass(name='RFC',\
												par={'solmethod':'RFC','T':T,\
														'do_print':do_print,\
														'k': 65, 'Nm':Nm,\
														'rad':rad, 'rad_I':rad,\
														'M_bar': 1.0001,\
														'k1':1,\
														'k2':1,\
														'intersection': False,\
														'interp_intersect': False,\
														's':0.045,\
														'max_iter': 5,\
														'n_closest': 3,\
														'nearest_fill': True,\
														'correct_jumps': False})
		model_RFC.precompile_numba()
		model_RFC = timing(model_RFC, rep=1)
		#figs.decision_functions(model_RFC,2,"RFC_{}".format(Nm))
		#figs.segments(model_RFC,2,"RFC_{}".format(Nm))


		#RFC with intersection 
		model_RFC_intersect = G2EGMModelClass(name='RFC_intersect',\
													par={'solmethod':'RFC','T':T,\
														'do_print':do_print,\
														'k': 60, 'Nm':Nm,\
														'rad':rad, 'rad_I':rad,\
														'M_bar': 1.0001,\
														'k1':40,\
														'k2':1,\
														'intersection': True,\
														'interp_intersect': False,\
														's':0.06,\
														'max_iter': 10,\
														'n_closest': 2,\
														'nearest_fill': True,\
														'correct_jumps': False})
		model_RFC_intersect.precompile_numba()
		model_RFC_intersect = timing(model_RFC_intersect, rep=1)
		#figs.decision_functions(model_RFC_intersect,2,"RFC_intersect_{:.0f}".format(Nm))
		#figs.segments(model_RFC_intersect,2,"RFC_intersect_{:.0f}".format(Nm))


		#NEGM

		model_NEGM = G2EGMModelClass(name='NEGM',par={'solmethod':'NEGM','T':T,'do_print':do_print, 'Nm':Nm})
		model_NEGM.precompile_numba()
		model_NEGM = timing(model_NEGM, rep=1)

		
		#G2EGM
		model_G2EGM = G2EGMModelClass(name='G2EGM',par={'solmethod':'G2EGM','T':T,'do_print':do_print, 'Nm':Nm})
		model_G2EGM.precompile_numba()
		model_G2EGM = timing(model_G2EGM, rep=1)
		#figs.decision_functions(model_G2EGM,2,"G2EGM_{:.0f}".format(Nm))
		#figs.segments(model_G2EGM,2,"G2EGM_{:.0f}".format(Nm))

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
		
		models_RES['grid_size'] = Nm

		# plot histogram of RFC vs G2EGM in the same plot to compare
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

		#collect the timings data to plot 
		if rank == 0:
			# save results
			pickle.dump(modelsAll, open('plots/pensions/results_{}.pkl'.format(nameres), 'wb'))
			results = {}
			
			for rank, modelres in zip(range(len(modelsAll)),modelsAll):
					results[NmList[rank]] = modelres

			
			plot_timing_data(results, 'plots/pensions/', NmList)
	else:
		if rank == 0:
			modelsAll = pickle.load(open('plots/pensions/results_{}.pkl'.format(nameres), 'rb'))
			results = {}
			labels = ['G2EGM', 'RFC w. Delaunay', 'NEGM']
			for j, modelres in zip(range(len(modelsAll)),modelsAll):
					results[NmList[j]] = modelres
			plot_timing_data(results, 'plots/pensions/', NmList, labels)

			#Generate Table
			#generate_table(results, NmList)
