# Graham West
from copy import deepcopy
import random
import numpy as np
import math
import pandas as pd
from subprocess import call
from scipy import optimize
from matplotlib import pyplot as plt
from matplotlib import image as img



##############
#    MAIN    #
##############

def main():
	
	ind = 5
	data = np.loadtxt("ResultsConv_01.txt")
	
	# plot param runs
	fig, axes = plt.subplots( nrows=int(2), ncols=int(3) )
	
	ind2 = 0
	for ax in axes.flat:
		if( ind2 == 0 ):
			ax.plot(data[:,ind],data[:,-2], 'b-')
			ax.set_title('error vs. param')
		elif( ind2 == 1 ):
			ax.plot(data[:,ind],data[:,-1], 'b-')
			ax.set_title('conv vs. param')
		elif( ind2 == 2 ):
			ax.plot(data[:,-2],data[:,-1], 'b-')
			ax.set_title('conv vs. param')
		
		ind2 += 1 
	# end

	
	plt.tight_layout(w_pad=-1, h_pad=-0.5)
	
	plt.show()
	
	
##############
#    MAIN    #
##############

def solve(paramList, nBin, bound, alpha, fitToPlot):
	
	p = deepcopy(paramList)
	
	a = p[6]
	b = p[7]
	
	p[7] = b/(a+1.0)
	p[6] = a*p[7]
	
	paramStr = ','.join( map(str, p[0:-2]) )
	call("./basic_run " + paramStr + " > SolveMetro.out", shell=True)
	RV = np.loadtxt("a.101")
	binC, binV = BinField(nBin, RV, bound, alpha, fitToPlot)
	
	call("./basic_run_unpreturbed " + paramStr + " > SolveMetro.out", shell=True)
	RV_u = np.loadtxt("a.000")
	binC_u, binV_u = BinField(nBin, RV_u, bound, alpha, fitToPlot)
	
	return binC, binV, binC_u, binV_u, RV, RV_u

def GetWeights( nBin, binC, binCu, weights ):
	
	if( weights == 1 ):
		
		W = np.zeros((nBin,nBin))
		
		eps = 0.000001
		
		binC  = binC*1.0
		binCu = binCu*1.0
		
		maxC  = np.max(np.max(binC ))
		maxCu = np.max(np.max(binCu))
		
		for i in range(nBin):
			for j in range(nBin):
				vv = (binC[j,i]-binCu[j,i])/(binC[j,i]+binCu[j,i]+eps)
				nw = binC[j,i]/maxC + binCu[j,i]/maxCu
				W[j,i] = nw*vv**2
			# end
		# end
		
		W = W/np.sum(np.sum(W))*nBin**2
	else:
		W = np.ones((nBin,nBin))
	# end
	
	return W



def BinField(nBin, RV, bound, alpha, fitToPlot):

	nPts = np.size(RV,0)
	
	binCnt  = np.zeros((nBin,nBin,nBin))
	binVel  = np.zeros((nBin,nBin,nBin))
	
	binCnt2 = np.zeros((nBin,nBin))
	binVel2 = np.zeros((nBin,nBin))

	if( fitToPlot == 1 ):
		rmax = np.max(np.max(np.max( np.abs(RV[:,0:2]) ) ) )
		
		xmin = np.min( RV[:,0] )
		xmax = np.max( RV[:,0] )
		ymin = np.min( RV[:,1] )
		ymax = np.max( RV[:,1] )
		zmin = np.min( RV[:,2] )
		zmax = np.max( RV[:,2] )
	else:
		xmin = bound[0,0]
		xmax = bound[0,1]
		ymin = bound[1,0]
		ymax = bound[1,1]
		zmin = bound[2,0]
		zmax = bound[2,1]
	# end
	
	dx = (xmax-xmin)/nBin
	dy = (ymax-ymin)/nBin
	dz = (zmax-zmin)/nBin

	for i in range(nPts):		
		x = float(RV[i,0])
		y = float(RV[i,1])
		z = float(RV[i,2])
		vz = float(RV[i,5])
		
		ii = (x - xmin) / (xmax - xmin) * nBin
		jj = (y - ymin) / (ymax - ymin) * nBin
		kk = (z - zmin) / (zmax - zmin) * nBin
		
		if( ii > 0 and ii < nBin and jj > 0 and jj < nBin and kk > 0 and kk < nBin ):
		        binCnt[jj,ii,kk] = binCnt[jj,ii,kk] + 1
		        binVel[jj,ii,kk] = binVel[jj,ii,kk] + vz
	
	
	for i in range(nBin):
		for j in range(nBin):
			for k in range(nBin):
				if( binCnt[i,j,k] > 1 ):
					binVel[i,j,k] = binVel[i,j,k]/binCnt[i,j,k]
			# end
		# end
	# end
	
	
	for i in range(nBin):
		for j in range(nBin):
			sumV = 0
			sumW = 0
			
			for k in range(nBin):
				if( binCnt[i,j,k] > 0 ):
					w = np.exp(-alpha * np.sum(binCnt[i,j,nBin-k-1:nBin])*dz/(dx*dy*dz)/(nPts-1) )
					
					sumV += binCnt[i,j,k]*w*binVel[i,j,k]
					sumW += binCnt[i,j,k]*w
				# end
			# end
			
			if( sumW > 0 ):
				binVel2[i,j] = sumV/sumW
			else:
				binVel2[i,j] = 0
			# end
			binCnt2[i,j] = np.sum(binCnt[i,j,:])
		# end
	# end
	
	
	
	
	
	
	return binCnt2, binVel2

# end

def ErrorFunction( nBin, Vi, V1 ):
	
	As = 0
	At = 0
	Ovr = 0
	MSE = 0
	morph = 0
	error = 0
	
	isOvr = 0
	
	W = np.zeros((nBin,nBin))
	
	for i in range(nBin):
		for j in range(nBin):
			
			if( V1[j,i] > 0 or V1[j,i] < 0 ):
				At = At + 1
				if( Vi[j,i] > 0 or Vi[j,i] < 0 ):
					As = As + 1
					Ovr = Ovr + 1
					isOvr = 1
			elif( Vi[j,i] > 0 or Vi[j,i] < 0 ):
				As = As + 1
			# end if
			
			if( isOvr == 1 ):
				MSE = MSE + ( Vi[j,i] - V1[j,i] )**2
			# end if
			
			isOvr = 0
		
		# end for
	# end for
	
	if( Ovr > 1 ):
		MSE = MSE/Ovr
	
	RMSE = math.sqrt(MSE)
	
	OvrFrac = Ovr/(As+At-Ovr*1.0)
#	OvrFrac = Ovr/(As)
#	error = RMSE*(1-OvrFrac)
#	error = (1-OvrFrac**2)
#	error = RMSE
	
	Bi = np.piecewise(Vi, [Vi > 0, Vi < 0, Vi == 0], [1, 1, 0])
	B1 = np.piecewise(V1, [V1 > 0, V1 < 0, V1 == 0], [1, 1, 0])
	corr = np.corrcoef( np.ndarray.flatten(Bi), np.ndarray.flatten(B1) )[1,0]
	
#	error = RMSE*math.sqrt((1-OvrFrac)*(1-corr))
	
	error = RMSE**(1-corr)**0.5
	
	return error, RMSE, OvrFrac, corr

main()
