import pyqtgraph as pg
import PyQt5
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
import time
import sys
import random
import math

import RBF00

SAMPLES = 75

# Enable antialiasing for prettier plots
pg.setConfigOptions(antialias=True)

win = pg.GraphicsWindow()
win.setWindowTitle('RBF - Response Curve')

'''
	Red Pen
'''
redPen = QtGui.QPen(QtGui.QBrush(QtGui.QColor(255, 0, 0)), 0.09)


'''
	Function to approximate: y = 0.5 + 0.4 * sin(2^x)
'''
def h(x):
	# note: pylab.sin can accept a numpy.ndarray, but math.sin cannot
	return 0.5 + 0.4 * np.sin(np.pi * 2 * x)


'''
	Add uniform noise in intervale [-0.1, 0.1].
'''
def noise(x):
	return x + random.uniform(-0.1, 0.1)


'''
	Return sample of n random points uniformly distributed in [0, 1].
'''
def sample(n):
	a = [random.random() for x in range(n)]
	a.sort()
	return a


'''
	Return error from actual to expected.

	Args
		actual: [num] of sampled output
		expected: [num] of expected ouput, ||expected|| = ||actual||
	Returns:
		num of average distance between actual and expected
'''
def error(actual, expected):
	sum_d = 0
	
	for a, e in zip(actual, expected):
		sum_d += RBF00.distance(a,e)

	err = sum_d / len(expected)

	return err


'''
	Curve Graph
'''
x = np.linspace(0, 1, 100, endpoint=True)
y = (0.5 + 0.4 * np.sin(np.pi * 2 * x))

p4 = win.addPlot(title="RBF - Desired Curve + Network Output Curve + Noisy Data")
p4.showGrid(x=True, y=True)
p4.plot(x, y)


'''
	Radial Width and Position 
'''
win3 = pg.GraphicsWindow()
win3.setWindowTitle('RBF - Radials Center and Width')

radialsPlot = win3.addPlot(title="RADIALS")
radialsPlot.showGrid(x=True, y=True)


'''
	Network Topology Layout
'''
win2 = pg.GraphicsWindow()
win2.setWindowTitle('RBF - Network Topology')

v = win2.addViewBox()
v.setAspectLocked()

g = pg.GraphItem()

v.addItem(g)

## Define positions of nodes
pos = np.array([[0,0], [15,0]])

## Define the set of connections in the graph
adj = np.array([0,0])


def drawRBFNetwork():

	global pos, adj, g

	totalRadials = len(RBF00.centers)

	k = (-totalRadials / 2) * 5

	for i in range(0, totalRadials):

		totalPlotted = len(pos)

		#Hidden node
		pos = np.vstack((pos, np.array([[5, k]])))

		#Line from input node
		adj = np.vstack((adj, np.array([[0, totalPlotted]])))

		#Line to output node
		adj = np.vstack((adj, np.array([[totalPlotted, 1]])))

		text2 = pg.TextItem("[10]", anchor=(0.5, -1.0))
		arrow2 = pg.ArrowItem(angle=90)
		arrow2.setParentItem(text2)
		v.addItem(text2)
		text2.setPos(6, k)

		RBF00.weightLabels.append(text2)

		#Flip point as we never want a radial at k=0
		if k == -5:
			k = 5
		else:
			k = k + 5 							

	## Update the graph
	g.setData(pos=pos, adj=adj, pen='r', size=1, pxMode=False)


def updateP4(item):
	if item[2] == 'o':
		p4.plot(item[0], item[1], pen='r', fillLevel=0, fillBrush=(255,0,0,30), name='red plot')
	elif item[2] == 'g':
		p4.plot(item[0], item[1], pen='g', fillLevel=0, fillBrush=(255,255,255,30), name='green plot')
	else:
		p4.plot(item[0], item[1], symbol=item[2], pen={'color': 0.8, 'width': 1})


def updateRadialsPlot(item):
	radialsPlot.plot(item[0], item[1], symbol=item[2], pen={'color': 0.8, 'width': 1})


class Thread(pg.QtCore.QThread):

	newData = pg.QtCore.Signal(object)
	radialsPlotData = pg.QtCore.Signal(object)
	clearPlots = pg.QtCore.Signal()

	def run(self):

		# compute input samples for x
		input_x = sample(SAMPLES)
	
		RBF00.init(input_x, k=10)

		# compute desired and ideal outputs for y
		ideal_y = [h(x) for x in input_x]

		drawRBFNetwork()

		#Train the network
		for i in range(1000):

			print("Traning iteration: %d" % i)

			#Give some new data to be learnt
			noisy_y = [noise(y) for y in ideal_y]

			RBF00.train(input_x, noisy_y, eta=0.02) 

			# examine results
			trained_y = [(lambda y: RBF00.output(y))(x) for x in input_x]

			print("AVERAGE ERROR: %3f" % (error(ideal_y, trained_y)))

			self.clearPlots.emit()
			self.newData.emit([input_x, ideal_y, 'o'])
			self.newData.emit([input_x, noisy_y, 't'])
			self.newData.emit([input_x, trained_y, 'g'])

			#See how the radials are responding
			for center,width in zip(RBF00.centers, RBF00.widths):
				averagex,inputxset = center
				y_responses = [(lambda inputx: RBF00.gaussian(averagex, width, x))(x) for x in input_x]
				self.radialsPlotData.emit([input_x, y_responses, 'x'])

			time.sleep(1)

		self.clearPlots.emit()
		self.newData.emit([input_x, ideal_y, 'o'])
		self.newData.emit([input_x, trained_y, '+'])


if __name__ == "__main__":

	thread = Thread()
	thread.newData.connect(updateP4)
	thread.radialsPlotData.connect(updateRadialsPlot)
	thread.clearPlots.connect(p4.clear)
	thread.clearPlots.connect(radialsPlot.clear)
	thread.start()

	QtGui.QApplication.instance().exec_()

	random.seed()
