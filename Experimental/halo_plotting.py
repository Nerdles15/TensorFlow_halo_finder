# Simple single-run plotting use of tensorflow kmeans
# Doesn't invoke fit training, requires specific number of halos. Useful for testing halo mass function, parameter fits, etc

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import math
import timeit


if len(sys.argv) < 3:
    print '\nERROR- Correct input form is: python halo_plotting.py (num_clusters) (.txt particle file)\n'
    sys.exit(1)  # abort because of error

NUM_CLUSTERS = int(sys.argv[1])
TRAIN_THRESHOLD = 0.03
MAX_ITERS = 1000

INPUTFILE = sys.argv[2]

#
# Raw data:
#
raw_frame = pd.read_csv(INPUTFILE, delim_whitespace=True, names=['x', 'y', 'z','vx','vy','vz','V','m'])

print '\nFile contains',len(raw_frame),'particles.\n'
normalization = raw_input('Normalize data? yes/no: ')

start_time = timeit.default_timer()

if normalization in ['yes']:
	#
	# Normalize values to aid in fitting (decreases dynamic range)
	#
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


#
# Run a K-means fit over the n-dimensional parameter space: 
#
def points():
	#return tf.train.limit_epochs(tf.convert_to_tensor(pts_frame.as_matrix(), dtype=tf.float32), num_epochs=1)
	#return tf.train.limit_epochs(tf.convert_to_tensor(raw_frame.iloc[:,0:8].as_matrix(), dtype=tf.float32), num_epochs=1)
	return tf.train.limit_epochs(tf.convert_to_tensor(raw_frame.iloc[:,0:4].as_matrix(), dtype=tf.float32), num_epochs=1)

#
# This function implements a K-Means fit to the input points.
#
def fit(num_clusters, training_iters=20):
	
	#
	# Set up a KMeansClustering instance.
	#
	tf.logging.set_verbosity(tf.logging.ERROR)
	kmeans = tf.contrib.factorization.KMeansClustering(num_clusters=num_clusters,relative_tolerance=0.0001)
	
	#
	# Iterate network training. Includes exit criteria for
	# a score ratio (WRT previous score)
	#
	previous_centers = None
	last_score = None
	for _ in xrange(MAX_ITERS):
		kmeans.train(points)
		cluster_centers = kmeans.cluster_centers()
		if previous_centers is not None:
			previous_centers = cluster_centers
		score=kmeans.score(points)
		score_array.append(score)

		#
		# Evaluate training threshold.
		#
		if last_score is not None:
			delta = (last_score - score) / score
			print 'Train delta =', delta
			if delta < TRAIN_THRESHOLD:
				break
		last_score = score

	#
	# Return cluster centroids, assignments, and overall score for fit.
	#
	clusters = kmeans.cluster_centers()
	assignments = list(kmeans.predict_cluster_index(points))
	score=kmeans.score(points)

	return clusters, assignments, score

#
# Fit points to clusters
#
score_array = []
final_score = []
fit_deltas = []
score = []
clusters, assignments, score = fit(num_clusters=NUM_CLUSTERS)

elapsed = timeit.default_timer() - start_time
print '\nTotal time elapsed:',elapsed,'seconds.\n'

#
# Create an array of halos. Each element of the array contains an array of points assigned to a given halo.
#
print '\nBeginning mass and potential calculations...'
Vav = 0.0 	# Calculate an average potential for all the particles
halos = [[] for i in range(NUM_CLUSTERS)]
for i in range(len(assignments)):
	halos[assignments[i]].append([raw_frame.x[i],raw_frame.y[i],raw_frame.z[i],raw_frame.vx[i],raw_frame.vy[i],raw_frame.vz[i],raw_frame.V[i],raw_frame.m[i]])
	Vav += raw_frame.V[i]

Vav /= len(assignments)

print '\nFinished potential calculations. Begin data plotting...'

#
# Plot the clusters in contrasting colors
#
def get_cmap(n, name='hsv'):
	return plt.cm.get_cmap(name, n)

#
# Test whether or not a particular point lies in a given box
#
def isinbox(pt,box):
	if (pt[0]>box[0][0]) and (pt[0]<box[0][1]) and (pt[1]>box[1][0]) and (pt[1]<box[1][1]) and (pt[2]>box[2][0]) and (pt[2]<box[2][1]):
		return True
	else:
		return False

#
# Plot halos in contrasting colors in position space.
#
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
			V /= len(x) # Average potential in the halo
			
			if (V < Vav): # Overdense region
				halo_mass.append(Mtot)
				ax.scatter(x,y,z,s=0.003*(Vav/V),color=cmap(i))
				ax.scatter(clusters[i][0],clusters[i][1],clusters[i][2],s=100,marker='+',color='black')
			else: # Underdense region 
				halo_mass.append(Mtot)
				#ax.scatter(clusters[i][0],clusters[i][1],clusters[i][2],s=100,marker='+',linewidth=1,color='black')
				ax.scatter(x,y,z,s=0.008*(Vav/V),color=cmap(i),alpha=0.2)
				#print ' '		# Needed to fill else: call to prevent error

	plt.title(NUM_CLUSTERS)
	plt.show()
	
	#
	# Need separate call for HMF plotting
	#
	print '\nPlotting Halo Mass Function:'
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


box=[[0.0,50000.0],[0.0,50000.0],[0.0,50000.0]] # Box to plot
plotbox(halos,clusters,box)


