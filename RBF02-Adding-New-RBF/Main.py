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

# Red Pen
redPen = QtGui.QPen(QtGui.QBrush(QtGui.QColor(255, 0, 0)), 0.09)



#0.5 + 0.4 * pylab.sin(pylab.pi * 2 * x)
x = np.linspace(0, 1, 100, endpoint=True)
y = (0.5 + 0.4 * np.sin(np.pi * 2 * x))

p4 = win.addPlot(title="RBF - Desired Curve")
p4.showGrid(x=True, y=True)
p4.plot(x, y)

#power = -1 / 0.05 / 2 * (x - 0.5) ** 2
#power = -(x - 0) ** 2 / 0.05
#y = np.exp(power)
#p4.plot(x, y) 


'''
	Network Topology Layout
'''
win2 = pg.GraphicsWindow()
win2.setWindowTitle('RBF - Network Topology')

v = win2.addViewBox()
v.setAspectLocked()

g = pg.GraphItem()

v.addItem(g)

'''
inputNode = QtGui.QGraphicsEllipseItem(-0.5, -0.5, 1, 1)
inputNode.setPen(redPen)

outputNode = QtGui.QGraphicsEllipseItem(14.5, -0.5, 1, 1)
outputNode.setPen(redPen)

v.addItem(inputNode)
v.addItem(outputNode)
'''

## Define positions of nodes
pos = np.array([[0,0], [15,0]])

## Define the set of connections in the graph
adj = np.array([0,0])


def addRBFs():

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



'''
	Radial Width and Position 
'''
win3 = pg.GraphicsWindow()
win3.setWindowTitle('RBF - Radials Width and Position')

radialsPlot = win3.addPlot(title="RADIALS")
radialsPlot.showGrid(x=True, y=True)



def h(x):
	"""Function to approximate: y = 0.5 + 0.4 * sin(2^x)"""
	# note: pylab.sin can accept a numpy.ndarray, but math.sin cannot
	return 0.5 + 0.4 * np.sin(np.pi * 2 * x)


def noise(x):
	"""Add uniform noise in intervale [-0.1, 0.1]."""
	return x + random.uniform(-0.1, 0.1)


def sample(n):
	"""Return sample of n random points uniformly distributed in [0, 1]."""
	a = [random.random() for x in range(n)]
	a.sort()
	return a


def stats(values):
	"""
		Return tuple of common statistical measures.

		Returns: (num, num, num, num) as (mean, std, min, max)
	"""

	mean = sum(values) / len(values)

	sum_sqs = sum([(lambda y: y*y)(x) for x in values])
	var = sum([(mean-x)**2 for x in values]) / len(values)
	var = (sum_sqs - len(values)*mean**2) / len(values)
	std = var**0.5
	min_var, max_var = min(values), max(values)
	
	return (mean, std, min_var, max_var)



#y = (0.2 + 0.6 * np.sin(np.pi * 2 * x))
#top = np.linspace(0.1, 0.1, 100)
#bottom = np.linspace(0.1, 0.1, 100)
#top[5] = 0
#bottom[5] = 0
#err = pg.ErrorBarItem(x=x, y=y, top=top, bottom=bottom, beam=0.01)
#p4.addItem(err)


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
	
		RBF00.init(input_x, k=20)

		# compute desired and ideal outputs for y
		ideal_y = [h(x) for x in input_x]

		addRBFs()

		#Train the network
		for i in range(1000):

			print("Traning iteration: %d" % i)

			#Give some new data to be learnt
			measured_y = [noise(y) for y in ideal_y]

			RBF00.train(input_x, measured_y, eta=0.02) 

			# examine results
			trained_y = [(lambda y: RBF00.output(y))(x) for x in input_x]

			self.clearPlots.emit()
			self.newData.emit([input_x, ideal_y, 'o'])
			self.newData.emit([input_x, measured_y, 't'])
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

	#timer = pg.QtCore.QTimer()
	#timer.timeout.connect(update3)
	#timer.start(50)

	thread = Thread()
	thread.newData.connect(updateP4)
	thread.radialsPlotData.connect(updateRadialsPlot)
	thread.clearPlots.connect(p4.clear)
	thread.clearPlots.connect(radialsPlot.clear)
	thread.start()

	QtGui.QApplication.instance().exec_()

	random.seed()


