#---------------------------------------------------------------------#
#                    Tensorflow Halo Finder (Kmeans)				  #
#                       Will Kinney, John Nord                        #
#                    Using updated API as of 5/3/18  	              #
#				 Iterates both training, num_clusters                 #
#---------------------------------------------------------------------#

#-------------#
#  CHANGELOG  #
#-------------#

# + Added total energy of particles to dataframe
# + Added exception handling to different sections and selectable data routine to code,
#		so don't have to manually flag/unflag multiple lines (and likely miss something)
# + Commented sections and reorganized to increase human-reading compatibility
# + Separated method of fitting normalized vs non-normalized data, so each has
#		independent ways of reaching optimal fitness (linear vs oscillatory)




import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import timeit


#---------------------------------------------------------------#
#  Set threshold values, max number of iterations for program.  #
#---------------------------------------------------------------#
TRAIN_THRESHOLD = 0.03			# Effects accuracy of centroid position
FIT_THRESHOLD = 0.03			# Effects number of centroids
CHI_THRESHOLD = 1.00			# Should be of order unity +/- small value when at optimal num_clusters 
MAX_ITERS = 1000				# Max iterations value to keep program from running infinitely


#-------------------------------------------------#
#  Program expects formatted input .txt file,     #
#  see accompanying 'Gadget2ASCII.c' to produce.  #
#-------------------------------------------------#
print '\nCorrect dataframe input format- please double check!'
print '[x    y    z    vx    vy    vz    V    m]\n'


#-------------------------------------------#
#  Routine to select which dataset to use.  #
#-------------------------------------------#
while True:
	try:
		which_file = int(input('\nChoose 64^3 or 128^3 simulation (Type 64 or 128): '))
		if which_file == 64:
			INPUTFILE = './TestPoints_all_param.txt'	# 64^3 test snapshot (64_24Mpc_snapshot)
			break
		elif which_file == 128:
			INPUTFILE = './Fullscale.txt'				# 128^3 test snapshot (128_50Mpc)
			break
		else:
			print '\nError: That number is not a valid option. Please choose 64 or 128.\n'
	except NameError:
		print '\nError: Unrecognized input. Please choose 64 or 128.\n' 


#---------------------------------------------------#
#  Convert raw data from .txt file into dataframe.  #
#---------------------------------------------------#
raw_frame = pd.read_csv(INPUTFILE, delim_whitespace=True, names=['x', 'y', 'z','vx','vy','vz','V','m'])


#---------------------------------------------#
#  Choose Normalized or Non-Normalized data.  #
#---------------------------------------------#
while True:
	normalization = raw_input('\nNormalize data? yes/no: ')
	if normalization in ['yes']:
		break
	elif normalization in ['no']:
		break
	else:
		print '\nError: That number is not a valid option. Please choose yes or no.\n'


if normalization in ['yes']:
	#-----------------------------------------------------------#
	#  Normalize values to aid in fitting                       #
	#  (decreases dynamic range, equally weighted parameters).  #
	#-----------------------------------------------------------#
	print '\nNormalizing dataframe...'
	print '------------------------\n'
	raw_frame.x = raw_frame.x / max(raw_frame.x)
	raw_frame.y = raw_frame.y / max(raw_frame.y)
	raw_frame.z = raw_frame.z / max(raw_frame.z)
	raw_frame.vx = raw_frame.vx / max(raw_frame.vx)
	raw_frame.vy = raw_frame.vy / max(raw_frame.vy)
	raw_frame.vz = raw_frame.vz / max(raw_frame.vz)
	raw_frame.V = raw_frame.V / max(raw_frame.V)


#-----------------------------------------#
#  Add sqrt of velocity**2 to dataframe.  #
#-----------------------------------------#
print '\nAdding sqrt(|v^2|) to dataframe...'
print '----------------------------------\n'
raw_frame.insert(loc=3, column='vv', value=raw_frame.vx)
raw_frame.vv = np.sqrt(raw_frame.vx * raw_frame.vx + raw_frame.vy * raw_frame.vy + raw_frame.vz * raw_frame.vz)
#print raw_frame			# debugging


#----------------------------------------------------#
#  Add total energy E = (1/2)mv^2 + V to dataframe.  #
#----------------------------------------------------#
print '\nAdding total energy E to dataframe...'
print '-------------------------------------\n'
raw_frame.insert(loc=3, column='E', value=raw_frame.m)
raw_frame.E = (0.5 * raw_frame.m * raw_frame.vv * raw_frame.vv) + raw_frame.V
raw_frame.E = raw_frame.E / max(raw_frame.E)


#--------------------------------------------------------------#
#  Select either just halo finding (outputs number of halos),  #
#  or halo finding + plot positions/centroids/mass function.   #
#--------------------------------------------------------------#
print '\nFile contains',len(raw_frame),'particles.'
while True:
	print '\nPlease select which type of analysis you wish to run:'
	print '\n\tBasic: Unsupervised KMeans run. Returns optimal number of clusters found.'
	print '\n\tFull: Unsupervised KMeans run, plus separation of particles into halos, plus'
	print '\t\tcalculation and plotting of particle positions and halo mass function.\n'
	GPU = raw_input('Basic or Full? ')
	if GPU in ['Basic']:
		break
	elif GPU in ['Full']:
		break
	else:
		print '\nError: invalid choice. Please select "basic" or "full" simulation.\n'


print '\n*******************************'
print '*  _________________________  *'
print '* /_____   _____/  ________/  *'
print '*      /  /    /  /___        *'
print '*     /  /    /  ____/        *'
print '*    /  /    /  /             *'
print '*   /  /    /  /              *'
print '*  /__/    /__/  Halo Finder  *'
print '*******************************\n'

start_time = timeit.default_timer()


#----------------------------------------------------------------------------------#
#  Run a K-means fit over the n-dimensional parameter space (given in dataframe):  #
#----------------------------------------------------------------------------------#
def points():
	return tf.train.limit_epochs(tf.convert_to_tensor(raw_frame.iloc[:,0:4].as_matrix(), dtype=tf.float32), num_epochs=1)


#---------------------------------------------------------------#
#  This function implements a K-Means fit to the input points.  #
#---------------------------------------------------------------#
def fit(num_clusters, training_iters=20):
	

	#------------------------------------------#
	#  Set up a TF KMeansClustering instance.  # 
	#  Only use x, y, z, E from dataframe.     #
	#------------------------------------------#
	tf.logging.set_verbosity(tf.logging.ERROR)
	kmeans = tf.contrib.factorization.KMeansClustering(num_clusters=num_clusters,relative_tolerance=0.0001)
	

	#------------------------------------------------------#
	#  Iterate network training. Includes exit criteria    #
	#  in the form of a score ratio (WRT previous score).  #
	#------------------------------------------------------#
	previous_centers = None
	last_score = None
	for _ in xrange(MAX_ITERS):
		kmeans.train(points)
		cluster_centers = kmeans.cluster_centers()
		if previous_centers is not None:
			previous_centers = cluster_centers
		score=kmeans.score(points)
		score_array.append(score)


		#--------------------------------#
		#  Evaluate training threshold.  #
		#--------------------------------#
		if last_score is not None:
			delta = (last_score - score) / score
			print 'Train delta =', delta
			if delta < TRAIN_THRESHOLD:
				break
		last_score = score


	#---------------------------------------------------------------------#
	#  Return cluster centroids, assignments, and overall score for fit.  #
	#---------------------------------------------------------------------#
	clusters = kmeans.cluster_centers()
	assignments = list(kmeans.predict_cluster_index(points))
	score=kmeans.score(points)

	return clusters, assignments, score


#---------------------------#
#  Fit points to clusters.  #
#  Two methods.             #
#---------------------------#
if normalization in ['no']:
	#----------------------------------------------------------#
	#  Uses a linear relationship to judge optimal fitness,	   #
	#  when current delta crosses threshold % of score delta.  #
	#----------------------------------------------------------#
	score_array = []
	final_score = []
	fit_deltas = []
	score = []
	num_clusters = 2 * len(raw_frame) / 1100
	last_score = None
	for _ in xrange (MAX_ITERS):
		print '\nBeginning fitting to', num_clusters, 'clusters...\n'
		clusters, assignments, score = fit(num_clusters=num_clusters)


		#---------------------------#
		#  Evaluate fit threshold.  #
		#---------------------------#
		if last_score is not None:
			delta = (last_score - score) / score
			print 'Fit delta =', delta
			fit_deltas.append(delta)
			if delta < FIT_THRESHOLD:
				print '\n***********************************************'
				print 'Optimal number of clusters achieved at:', num_clusters, '!'
				print '***********************************************\n'
				break


		#----------------------------------------------------#
		#  Print statistics of last num_clusters iteration,  #
		#  then move on to next iteration.				     #
		#----------------------------------------------------#
		final_score.append(score)
		print '\n---------------------------------------------------------------------------------------------'
		print '\n*******'
		print ' INFO:'
		print '*******\n'
		print 'Number of clusters =', num_clusters, ', fit score =', score, ', previous score =', last_score
		print '---------------------------------------------------------------------------------------------\n'
		last_score = score
		num_clusters += len(raw_frame) / 10000


if normalization in ['yes']:
	#----------------------------------------------------------#
	#  Uses oscillating relationship by taking previous score  #
	#  as new num_clusters to judge optimal fitness.           #												   #
	#----------------------------------------------------------#
	score_array = []
	final_score = []
	fit_deltas = []
	score = []
	num_clusters = 2 * len(raw_frame) / 1100
	last_score = None
	num_params = raw_frame.shape[1]
	for _ in xrange (MAX_ITERS):
		print '\nBeginning fitting to', num_clusters, 'clusters...\n'
		clusters, assignments, score = fit(num_clusters=num_clusters)


		#-------------------------------------------#
		#  Evaluate fit threshold.                  #
		#  Aim for chisq/DoF between 0.97 and 1.03  #
		#-------------------------------------------#
		if last_score is not None:
			chisq = score / (num_clusters - num_params)		# Calculate chi-squared per degree of freedom
			print 'Chi-squared =', chisq
			fit_deltas.append(chisq)
			if chisq < CHI_THRESHOLD * 1.03 and chisq > CHI_THRESHOLD * 0.97:
				print '\n***********************************************'
				print 'Optimal number of clusters achieved at:', num_clusters, '!'
				print '***********************************************\n'
				break


		#-------------------------------------------------------#
		#  Print statistics of current num_clusters iteration,  #
		#  then move on to next iteration.	   			        #
		#-------------------------------------------------------#
		final_score.append(score)
		print '\n---------------------------------------------------------------------------------------------'
		print '\n*******'
		print ' INFO:'
		print '*******\n'
		print 'Number of clusters =', num_clusters, ', fit score =', score, ', previous score =', last_score
		print '---------------------------------------------------------------------------------------------\n'
		last_score = score
		num_clusters = int(score)



#-----------------------#
#  Plot scores, deltas  #
#-----------------------#
print '\nFinished fit calculations. Begin score/fit plotting...'
print '------------------------------------------------------\n'

p1 = plt.subplot(2, 1, 1)
plt.plot(final_score)
plt.xlabel('Number of Iterations')
plt.ylabel('Score')
plt.title('Final Scores')

p2 = plt.subplot(2, 1, 2)
plt.plot(fit_deltas)
plt.xlabel('Number of Iterations')
if normalization in ['no']:
	plt.ylabel('Delta Score (% Max)')
if normalization in ['yes']:
	plt.ylabel('Chi-squared/DoF')
plt.title('Fit Deltas')

elapsed = timeit.default_timer() - start_time
print '\nTotal time elapsed:', elapsed, 'seconds.\n'

plt.show()


#------------------------------------------------#
#  Ends program if selection was for basic run,  #
#  only calculates optimal number of halos       #
#------------------------------------------------#
if GPU in ['Basic']:
	print '\nOmitting 3d plot to prevent freezeups/crashes.'
	print '\nIf you wish to do a visual plot, here is a one-liner:'
	print '\n\tpython halo_plotting.py',num_clusters,INPUTFILE,'\n'
	quit()


#---------------------------------------------------------#
#  Create an array of halos. Each element of the array    #
#  contains an array of points assigned to a given halo.  #
#---------------------------------------------------------#
print '\nBeginning halo mass and cluster potential calculations...'
print '---------------------------------------------------------\n'
Vav = 0.0
halos = [[] for i in range(num_clusters)]
for i in range(len(assignments)):
	halos[assignments[i]].append([raw_frame.x[i],raw_frame.y[i],raw_frame.z[i],raw_frame.vx[i],raw_frame.vy[i],raw_frame.vz[i],raw_frame.V[i],raw_frame.m[i]])
	Vav += raw_frame.V[i]


#---------------------------------------------------------#
#  Calculate average potential of all particles in box.   #
#  Compare with average potential of halos later.         #
#---------------------------------------------------------#
Vav /= len(assignments)


print '\nFinished cluster potential calculations. Begin data plotting...'
print '---------------------------------------------------------------\n'


#--------------------------------------------------------------------#
#  Get contrasting color scheme for plotting clusters of particles.  #
#--------------------------------------------------------------------#
def get_cmap(n, name='hsv'):
	return plt.cm.get_cmap(name, n)


#-----------------------------------------------------------------------------#
#  Test whether or not points within a halo lie in a given box for plotting.  #
#-----------------------------------------------------------------------------#
def isinbox(pt,box):
	if (pt[0]>box[0][0]) and (pt[0]<box[0][1]) and (pt[1]>box[1][0]) and (pt[1]<box[1][1]) and (pt[2]>box[2][0]) and (pt[2]<box[2][1]):
		return True
	else:
		return False


#-----------------------------------------------------------#
#  Plot halos in contrasting colors in position space.      #
#  Calculate and plot total mass of each halo.              #
#  If avg potential is lower than of all space, is a halo.  #
#-----------------------------------------------------------#
def plotbox(halos,clusters,box):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	cmap = get_cmap(len(halos))
	halo_mass = []
	for i in range(len(halos)):
		x = []
		y = []
		z = []
		V = 0.0
		Mtot = 0.0
		for j in range(len(halos[i])):
			if isinbox(clusters[i],box):
				x.append(halos[i][j][0])
				y.append(halos[i][j][1])
				z.append(halos[i][j][2])
				V += halos[i][j][6]
				Mtot += halos[i][j][7]
		
		if len(x) > 0:

			#--------------------------------------------------#
			#  Calculate avg potential of group of particles.  #
			#--------------------------------------------------#
			V /= len(x)
			
			if (V < Vav): 					# Overdense region, likely a halo
				halo_mass.append(Mtot)
				ax.scatter(x,y,z,s=0.002*(Vav/V),color=cmap(i))
				#ax.scatter(clusters[i][0],clusters[i][1],clusters[i][2],s=100,marker='+',color='black')
			else: 							# Underdense region, likely not a halo
				halo_mass.append(Mtot) 
				#ax.scatter(clusters[i][0],clusters[i][1],clusters[i][2],s=100,marker='+',linewidth=1,color='black')
				ax.scatter(x,y,z,s=0.008*(Vav/V),color=cmap(i),alpha=0.2)
	
	plt.title(num_clusters)
	plt.show()


	#-----------------------------------------------------------------------#
	# Plot halo mass function, histogram of halo masses.                    #
	# Plot both AHF halo mass function and TF halo mass function together.  #
	#-----------------------------------------------------------------------#
	print 'Loading: AHF Histogram- Halo Mass...'
	if which_file == 64:
		file = open('B24_64.z0.000.AHF_halos', 'r')		# _halos file, view halo mass
	else:
		file = open('B50_128.z0.000.AHF_halos', 'r')	# _halos file, view halo mass
	lines = file.readlines()
	AHF_halos = []
	for line in lines:
		if line[0] != '#':
			data = line.split()
			mass = float(data[3]) / 10e4		# Halo mass, _halo file
			AHF_halos.append(mass)
	plt.hist(AHF_halos, bins='auto', histtype='step', label='AHF Halo Mass')


	print '\nPlotting Halo Mass Function:\n'
	plt.hist(halo_mass, bins='auto', histtype='step', label='TF Halo Mass')
	plt.title('Halo Mass Function')
	if normalization in ['yes']:
		plt.xlabel('Halo Mass (Normalized)')
	else:
		plt.xlabel('Halo Mass (Non-Normalized)')
	plt.xscale('log')
	plt.yscale('log')
	plt.ylabel('Number of Halos')
	plt.legend(loc='upper right')
	plt.xlim(10e5, 10e8)

	plt.show()

	print '\nDone.\n'


#----------------------------------------------------------------------------#
#  Choose box size to plot.												     #
#  Can change to select different sub-regions and view halos more in depth.  #
#----------------------------------------------------------------------------#
box=[[0.0,50000.0],[0.0,50000.0],[0.0,50000.0]]
plotbox(halos,clusters,box)


