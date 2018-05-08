#                   TensorFlow Halo Finder (Kmeans)		      #
##                     Will Kinney, John Nord                          ##
####                   Using updated API as of 5/3/18    	              ####
<pre>
-------------------------------
-  _________________________  -
- /_____   _____/  ________/  -
-      /  /    /  /___        -
-     /  /    /  ____/        -
-    /  /    /  /             -
-   /  /    /  /              -
-  /__/    /__/  Halo Finder  -
-------------------------------
</pre>

## Foreward: ##

This is a halo finding program, designed to use an unsupervised KMeans algorithm in TensorFlow, to see if it's possible to create a competitive halo finder using these means.

Our goal was to create a functional halo finder, and then compare its results to those from existing halo finders (such as AMIGA Halo Finder, Rockstar, FoF_Special, MatchMaker, etc) to see if our output offers a viable alternative.

It can take any set of data fed to it (with any abstract number of parameters) and group together clusters accordingly.


# Sections: #
<pre>
I: 		Info
II: 		Installation
IIa.	 	Troubleshooting
III: 		Running The Code
IIIa:	 	Demo
IIIb:	 	Experimental
IV:		Partial Runs
V: 		Creating Datasets
</pre>



## I: Info- ##
<pre>
My system specs (Late-2015 Macbook Pro):
	-MacOS Sierra v.10.12.6
	-2.8 GHz quad-core i7
	-Intel Iris Pro graphics, 1536 MB VRAM
	-2x8 GB DDR3 1600 MHz RAM

Python 2.7.10 (System) info:
	-Python 2.7.10 (default, Feb  7 2017, 00:08:15) 
	-[GCC 4.2.1 Compatible Apple LLVM 8.0.0 (clang-800.0.34)] on darwin

TensorFlow info:
	DATA:
		-COMPILER_VERSION = '4.2.1 Compatible Apple LLVM 9.0.0 (clang-900.0.39....
		-CXX11_ABI_FLAG = 0
    		-GIT_VERSION = 'v1.5.0-0-g37aa430d84'
    		-GRAPH_DEF_VERSION = 24
    		-VERSION = '1.5.0'
    	PACKAGE CONTENTS:
    		-contrib (package), core (package), examples (package), libtensorflow_framework, python (package), tools (package)
    	SUBMODULES:
    		-app, bitwise, compat, data, distributions, errors, estimator, feature_column, flags, gfile, graph_util, image, initializers, keras, layers, linalg, logging, losses, metrics, nn, profiler, python_io, pywrap_tensorflow, resource_loader, saved_model, sets, spectral, summary, sysconfig, test, train, user_ops
</pre>


## II: Installation- ##

If necessary, install Python 2.7.x from the following link:

[https://www.python.org/downloads/](https://www.python.org/downloads/)


On MacOSX can also use homebrew (if installed):

	`brew install python2` (and associated dependencies as listed)
	
	
Install TensorFlow (compatible with Python2) from the following link:

[https://www.tensorflow.org/install/](https://www.tensorflow.org/install/)


### IIa: Installation Troubleshooting- ###

Following instructions from the provided links should be sufficient for most systems.


In case of difficulty, they offer some troubleshooting- however the issues I had and steps I followed are as follows:

-On MacOSX (v.10.12.6), system python (and modules) =/= manually installed python (and modules), and the OS prioritizes system python.

Need to set path for homebrew installed modules so system python will recognize them.

Navigate to bash profile `cd /Users/[USER]` `vim ~/.bash_profile`, then set appropriate PYTHONPATH: `PYTHONPATH=$PYTHONPATH:[/PATH/TO/DESIRED/PYTHON/MODULES]`

-If using homebrew (like I did) for missing modules, you may need to manually install TensorFlow dependencies (java8, bazel, protobuf, and any other error-causing missing modules/libs/packages)

-End result was me using system python (2.7.10) along with homebrew installed modules (linked with PYTHONPATH).

I'm sure there's an easier way (such as using Linux), but this is what worked for me in the end.


-If mpl_toolkits.mplot3d cannot be imported or errors out, the init file may be missing.

Navigate to appropriate module folder and add the missing initialization file with `echo > __init__.py`



## III: Running The Code- ##

Execute the program by navigating to the appropriate directory. Folders are as follows:


-Demo:

Program runs unsupervised KMeans clustering on supplied test files.

Can choose between 64^3 or 128^3 particle simulation. Can choose between normalized data or non-normalized.

Can choose between running just the unsupervised learning segment (iterating through varying numbers of clusters until best fit), or continuing on to calculate cluster positions, plot the clusters and particle positions, and plot the halo mass distribution histogram.	

-Experimental:

Includes all previously mentioned functions of demo version, but in addition can be run on user-supplied datasets (as explained in section IV), and includes capability of comparing halo mass function to that of existing halo finders (AMIGA Halo Finder, FoF_Special).

If you wish to run user-supplied data, be sure to add new code accordingly in TF_HALO_TEST.py as well as calculate halos using alternative halo finders for a direct comparison (currently only includes comparison for 64^3 and 128^3 tests).

### IIIa: Demo- ###

You will need to download the Fullscale.txt data file (since it's too large for github to host) into the `/Release_Version/Demo/` folder. Link is below:

[Fullscale.txt](https://buffalo.box.com/s/wnorlmtkepkmapgv1bcx7c2ex8ggjkbc)

Navigate to `cd /[YOUR/SYS/PATH/HERE]/Release_Version/Demo/`.


Contents will include the halo finding program (tf_halos.py), the standalone plotting program (halo_plotting.py), and the two test data files (TestPoints_all_param.txt (64^3) and Fullscale.txt (128^3)).


Execute the program with `python tf_halos.py` and follow the command line instructions to choose the parameters of the run.


Choices will include 64^3 vs 128^3 file, normalized vs non-normalized data, and Basic vs Full run.

-Basic will begin an unsupervised KMeans run, and returns the optimal number of clusters as determined by the program.	

-Full will begin an unsupervised KMeans run, find the optimal number of clusters, separate the individual particles into respective halos, then plot the particle + halo positions in a 3D plot, plus calculate the mass of each halo and plot the halo mass function in a histogram.

### IIIb: Experimental- ###

You will need to download the Fullscale.txt data file (since it's too large for github to host) into the `/Release_Version/Experimental/` folder. Link is below:

[Fullscale.txt](https://buffalo.box.com/s/wnorlmtkepkmapgv1bcx7c2ex8ggjkbc)

Navigate to `cd /[YOUR/SYS/PATH/HERE]/Release_Version/Experimental`.


Contents will include the above, plus the respective halo statistics files for the 64^3 and 128^3 simulations as determined by AMIGA Halo Finder (AHF) and FoF_Special.


In addition, included is the program used to read and interpret Gadget-2 snapshots (Gadget2ASCII.c), plus the test halo finding program (TF_HALO_TEST.py) which will run similar to the Demo version, but will plot comparison plots as well as allow for user-supplied data.


Feel free to edit the test program as you see fit!


A test snapshot containing 64^3 particles in a 24 Mpc box (64_24Mpc_snapshot) is available at the following link:

[64_24Mpc_snapshot](https://buffalo.box.com/s/9phuiagxkr64bs34pf83qadqgnn4ip7j)



## IV: Partial Runs- ##

Choosing the Basic run will end with a printout of the optimal number of clusters as determined by the program, and a one-liner for restarting the run at that "checkpoint" to continue the analysis (halo positions, visual plots, halo mass function).


Simply copy/paste the given line back into the terminal and it will begin assigning particles and analyzing the clusters.

It will result in a visual plot of the particle positions divided up into halos in addition to cluster centroids as +'s, and a histogram of the halo masses.


If you wish to begin a run from scratch or test a specific number of halos, the appropriate args are as follows:

	`python halo_plotting.py [NUM_CLUSTERS] [PARTICLE_FILE.txt]`



## V: Creating Datasets- ##

The experimental folder supplies the user with a program to read Gadget-2 snapshots (Gadget2ASCII.c).

Compatiblity should be universal, so compile with `gcc Gadget2ASCII.c -o read_snapshot`, and then execute with `./read_snapshot [YOUR_SNAPSHOT] [FILENAME.txt]`.

Your output .txt file should contain 8 (non-zero) columns in the following order:

<pre>
			x	y	z	vx	vy	vz	V	m
</pre>


-If the V or m columns return 0 or a significant number of negative values, check to see if your snapshot contains both gas *and* dark matter particles (Gadget-2 type 0 and type 1).

*Temporary solution, this will hopefully be fixed in future revisions.*





