from sklearn import tree
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris
#from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_regression 
#from sklearn.feature_selection import mutual_info_regression
#from sklearn.feature_selection import VarianceThreshold
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D
import pylab as pl
import numpy as np
import math
import pprint
import csv


def cutoff(X, Y, thresh):
	
	X0 = []
	X1 = []
	Y0 = []
	Y1 = []
	for i in range(n):
		if( Y[i] > thresh ):
	#		Y[i] = 1
			X1.append( np.array([float(it2) for it2 in X[i,0:m]]) )
			Y1.append(Y[i])
		else:
	#		Y[i] = 0
			X0.append( np.array([float(it2) for it2 in X[i,0:m]]) )
			Y0.append(Y[i])
	
	X0 = np.array(X0)
	X1 = np.array(X1)
	Y0 = np.array(Y0)
	Y1 = np.array(Y1)
		
	return X1, Y1, X0, Y0
	
# end


########################
########################
########################
########################


with open('GoodData.txt', 'r') as f:
	reader = csv.reader(f)
	data = list(reader)

n = len(data)
mMax = len(data[0])
print n, mMax

# only get first params 2-16
m = 12
off = 2

X = []
Y = []
for it in data:
	X.append( np.array([float(it2) for it2 in it[off:m+off]]) )
	Y.append( float(it[mMax-1]) )
X = np.array(X)
Y = np.array(Y)

convertMass = 1
if( convertMass == 1):
	for i in range(n):
		m1 = X[i,4]
		m2 = X[i,5]
		
		X[i,4] = m1/m2
		X[i,5] = m1+m2
	# end
# end


thresh = 0.55



plotAll = 1
if( plotAll == 1 ):
	
	fig, axes = plt.subplots(nrows=3, ncols=4)
	
	minThresh = 0.0
	maxThresh = max(Y)
	nStep = 500
	dt = (maxThresh - minThresh)/nStep
	curThresh = 0.0
	
	X1, Y1, X0, Y0 = cutoff(X, Y, curThresh)
	
	titles = ['z', 'v_x', 'v_y', 'v_z', 'm_p/m_s', 'm_p+m_s', 'eps_1','eps_2', 'phi_p', 'phi_s', 'theta_p', 'theta_s' ]
	
	ind = 0
	for ax in axes.flat:
		ax.plot(X[:,ind],Y,'r.')
		ax.set_title(titles[ind])
		ind += 1
	# end
	
	plt.tight_layout()
	
#	pl.savefig( 'HeartGalaxy_MeanStdPlot.png' )
	
# end


plt.show()





