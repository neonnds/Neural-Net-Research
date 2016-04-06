import numpy as np
import math
import random

TESTS = 12
EPOCHS = 100
ALPHA = 0.1
ALPHAMIN = 0.0
ALPHADECAY = 0.99

centers = []
widths = []
weights = []

addRBF = None
weightLabels = []

''' Give a rbf with a given width and center. What is its response given an input of x? '''
''' The gaussian is the rbf response curve '''
def gaussian(center, width, input_x):

	"""
	      Return gaussian radial function.

	      Args:
		radial: (num, num) of gaussian (base, width^2) pair
		x: num of input
	      Returns:
		num of gaussian output
	"""

	power = -1 / width / 2 * (input_x - center) ** 2
	#power = -(x - base) ** 2 / width2
	y = np.exp(power)

	return y

#def gaussian(x):
#	return math.exp((x ** 2) / -2) / (2*math.pi)	


''' Calculate the output of the network (y) for a given input (x) '''
def output(x):

	global radials, weights

	"""
		Return set of linearly combined gaussian functions.

		Args:
			radials: [(num, num) of (base, width^2) pairs
			weights: [num] of radial weights, |weights| -1 = |radials|
			x: num of input
		Returns:
			num of linear combination of radial functions.
	"""

	y = 0

	for center, width, weight in zip(centers, widths, weights):
		averagex, inputxset = center
		y += gaussian(averagex, width, x) * weight

	return y

def update_centers_average(x):

	global centers

	''' For a given x find what the most activated center is then add that x to it and recalculate the clusters center '''
	maxC = 0
	maxY = 0

	for c in range(0, len(centers)):
		averagex, inputxset = centers[c]
		width = widths[c]

		y = gaussian(averagex, width, x)

		if y > maxY:
			maxY = y
			maxC = c

	averagex, inputxset = centers[maxC]

	inputxset.append(x)
	averagex = sum(inputxset) / len(inputxset) 

	centers[maxC] = (averagex, inputxset)

	return centers


def update_centers(x):

	global ALPHA, ALPHAMIN, ALPHADECAY, centers

	distances = [(distance(center[0], x), index) for index, center in enumerate(centers)]

	d, index = min(distances)

	averagex, inputxset = centers[index]

	newAveragex = averagex + ALPHA * (x - averagex)

	#if ALPHA > ALPHAMIN:
	#	inputxset.append(x)
	#	newAveragex = sum(inputxset) / len(inputxset) 

	centers[index] = (newAveragex, inputxset)

	#print("INDEX: %i, ALPHA: %3f, BEFORE: %3f, AFTER: %3f" % (index, ALPHA, averagex, newAveragex))

	return centers


def update_weights(eta, x, d):

	global centers, widths, weights

	"""
		Update weight vector.

		Returns:
			[num] of updated weight vector, len = |weights|
	"""

	new_weights = []
	y = output(x)
	err = d-y

	#print("Error: %3f" % err)

	for center, width, weight in zip(centers, widths, weights):
		averagex, inputxset = center	
		w = weight + (eta * err * gaussian(averagex, width, x))
		new_weights.append(w)

	return new_weights


def distance(x1, x2):
	return abs(x1 - x2)


''' Create k nodes and calculate their center based on x input dataset '''
def k_means(input, k):
	"""
		Return n Gaussian centers computed by K-means algorithm from sample x.

		Args:
			input: [num] of input vector
			k: int number of bases, <= |set(input)|
		Returns:
			[(num, [num])] k-size list of (center, input cluster) pairs.
	"""

	# initialize k bases as randomly selected unique elements from input
	bases = random.sample(set(input), k)

	# place all inputs in the first cluster 0 to initialize
	clusters = [(x, 0) for x in input]

	updated = True

	while(updated):
		updated = False

		for i in range(0, len(clusters)):
			x, m = clusters[i]

			#Given the random value b and the desired input x, what is their distance apart and which cluster is it j 
			distances = [(distance(b, x), index) for index, b in enumerate(bases)]

			d, index = min(distances)

			# update to move x to a new base cluster other than the default of 0
			# That is, we want to ensure the input x is in the right cluster
			if m != index:

				#print("x: %3f, m: %i, d: %3f, j: %i" % (x, m, d, index))

				updated = True
				clusters[i] = (x, index)

		# update bases
		if updated:
			base_sums = [ [0,0] for s in range(k)]

			#For each input x how many times does it appear in a cluster
			for x, m in clusters:
				base_sums[m][0] += x
				base_sums[m][1] += 1

			# check for divide by zero errors
			new_bases = []

			for s, n in base_sums:
				# avoid rare edge case, <1% @ n=25
				# division by zero: select a new base from input
				if n == 0:
					base = random.sample(set(input), 1)[0]
				else:
					#Work out the average response given for the cumulative total of input x's
					base = s / n

				new_bases.append(base)

			bases = new_bases

	# generate returned value
	response = [ (b, []) for b in bases ]
	
	# For the new base (center) response provide the input x's that cause this response
	# That is, for the given x inputs we have relocated the centroid to the mean average between those x inputs
	for x, m in clusters:
		response[m][1].append(x)

	return response
      

def variance_width(k_meaned_x):
	"""
		Return mean, variance pairs computed from k_means(x, k).

		Args:
			k_meaned_x: [(num, [num])] of (base, input cluster) pairs
		Returns:
			[(num, num)] of (center, width^2) pairs.
	"""

	response = []

	for base, cluster in k_meaned_x:
		if len(cluster) > 1:
			var = sum([(base-x)**2 for x in cluster]) / len(cluster)
			# this actually produces excellent approximations
			# var = sum([(base-x)**2 for x in cluster])
		else:
			var = None

		response.append((base, var))

 	# set |cluster| widths to mean variance of other clusters
	vars = [v for b, v in response if v]
	
	if len(vars) == 0:
		raise Exception("No variance: cannot compute mean variance")
	else:
		var_mean = sum(vars) / len(vars)

	for i in range(len(response)):
		base, var = response[i]

		if not var:
			response[i] = (base, var_mean)

	return response


''' Give a set of rbfs with calculated centers, then calculate their default width based on the maximum distance between the centers '''
def shared_width(centers):
	"""
		Return shared gaussian widths computed from k_means(x, k).

		Args:
			k_meaned_x: [(num, [num])] of (base, input cluster) pairs
		Returns:
			[(num, num)] of (center, width^2) pairs.
	"""

	# ignore clusters
	bases = [b for b, cluster in centers]

	# compute distances between adjancent bases
	s_bases = bases[:]
	s_bases.sort()
	distances = [(lambda p: distance(p[0], p[1]))(x) for x in zip(s_bases, s_bases[1:])]
	max_d = max(distances)
	sigma_sq = (max_d / 2 ** 0.5) ** 2

	newWidths = []

	for b in bases:
		newWidths.append(sigma_sq)

	return newWidths

  
def error(actual, expected):
	"""
		Return error from actual to expected.

		Args
			actual: [num] of sampled output
			expected: [num] of expected ouput, ||expected|| = ||actual||
		Returns:
			num of average distance between actual and expected
	"""

	sum_d = 0
	
	for a, e in zip(actual, expected):
		sum_d += distance(a,e)

	err = sum_d / len(expected)

	return err


''' Create the NN by generating the radials and the weights '''
def init(input_x, k):

	global centers, widths, weights

	"""
		Run an RBF training test set; plot, return errors from results.

		Args:
			eta: training rate
			k: num of bases
			tests: num of sample set iterations
			f_width: function to generate radial widths

		Returns:
			{str: [num]} such that n = (tests*runs) and:
			"sample_err": [num] of n sampling errors
			"train_err": [num] of n training errors
			"gen_err": [num] of n estimation errors
	"""

	# initialize K radials
	centers = k_means(input_x, k)

	# give K radials their default width
	widths = shared_width(centers)

	# k+1 weights, last weight is bias
	weights = [random.uniform(-0.5, 0.5) for x in range(k)]


''' Train the network weights '''
#Need to start prunning the network in the advent of useless neurons
def train(input_x, measured_y, eta):

	global ALPHA, ALPHADECAY, ALPHAMIN, centers, widths, weights

	for i in range(0, 100):

		# train one epoch
		for x, d in zip(input_x, measured_y):

			#print("Traning epoch: %d" % epoch)

			weights = update_weights(eta, x, d)

			#Given the new x we need to update our base response
			centers = update_centers(x)

			widths = shared_width(centers)

			#epoch += 1

	for weightLabel, weight in zip(weightLabels, weights):
		weightLabel.setText(str(weight))

	if ALPHA > ALPHAMIN:
		ALPHA *= ALPHADECAY

