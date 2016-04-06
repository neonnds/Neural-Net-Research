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

weightLabels = []


''' 
	Give a rbf with a given width and center. What is its response given an input of x?
	The gaussian is the rbf response curve.

	Return gaussian radial function.

	Args:
		radial: (num, num) of gaussian (base, width^2) pair
		x: num of input
	Returns:
		num of gaussian output

'''
def gaussian(center, width, input_x):
	power = -1 / width / 2 * (input_x - center) ** 2
	y = np.exp(power)

	return y


''' 
	Calculate the output of the network (y) for a given input (x) 

	Return set of linearly combined gaussian functions.

	Args:
		radials: [(num, num) of (base, width^2) pairs
		weights: [num] of radial weights, |weights| -1 = |radials|
		x: num of input
	Returns:
		num of linear combination of radial functions.
'''
def output(x):

	global radials, weights

	y = 0

	for center, width, weight in zip(centers, widths, weights):
		averagex, inputxset = center
		y += gaussian(averagex, width, x) * weight

	return y

''' 
	Update the center positon of the rbf.
'''
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


'''
	Update the RBF network weights.

	Returns:
		[num] of updated weight vector, len = |weights|
	
'''
def update_weights(eta, x, d):

	global centers, widths, weights

	new_weights = []
	y = output(x)
	err = d-y

	#print("Error: %3f" % err)

	for center, width, weight in zip(centers, widths, weights):
		averagex, inputxset = center	
		w = weight + (eta * err * gaussian(averagex, width, x))
		new_weights.append(w)

	return new_weights


'''
	Calculate the absolute distance between two RBF x-input values.
'''
def distance(x1, x2):
	return abs(x1 - x2)


''' 
	Create k nodes and calculate their center based on x input dataset. 

	Return n Gaussian centers computed by K-means algorithm from sample x.

	Args:
		input: [num] of input vector
		k: int number of bases, <= |set(input)|
	Returns:
		[(num, [num])] k-size list of (center, input cluster) pairs.
'''
def k_means(input, k):
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
      

''' 
	Give a set of rbfs with calculated centers, then calculate their default width based on the maximum distance between the centers.

	Return shared gaussian widths computed from k_means(x, k).

	Args:
		k_meaned_x: [(num, [num])] of (base, input cluster) pairs
	Returns:
		[(num, num)] of (center, width^2) pairs.

'''
def shared_width(centers):
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

  
''' 
	Create the NN by generating the radials and the weights.
'''
def init(input_x, k):

	global centers, widths, weights

	# initialize K radials
	centers = k_means(input_x, k)

	# give K radials their default width
	widths = shared_width(centers)

	# k+1 weights, last weight is bias
	weights = [random.uniform(-0.5, 0.5) for x in range(k)]


''' 
	Train the network weights

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
	
'''
def train(input_x, noisy_y, eta):

	global ALPHA, ALPHADECAY, ALPHAMIN, centers, widths, weights

	for i in range(0, 100):

		# train one epoch
		for x, d in zip(input_x, noisy_y):

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

