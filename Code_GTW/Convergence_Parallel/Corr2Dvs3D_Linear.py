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
from mpl_toolkits.mplot3d import Axes3D



##############
#    MAIN    #
##############

def main():
	
	n = 2914
	m = 12
	
	with open('StatsFile.txt', 'r') as f:
		lines = f.readlines()
	
	num = map( float, lines[0].split(', ') )
	
	min_   = []
	max_   = []
	median = []
	mean   = []
	std    = []
	
	for i in range(m):
		min_.append(   map( float, lines[1+5*i].split(', ') ))
		max_.append(   map( float, lines[2+5*i].split(', ') ))
		median.append( map( float, lines[3+5*i].split(', ') ))
		mean.append(   map( float, lines[4+5*i].split(', ') ))
		std.append(    map( float, lines[5+5*i].split(', ') ))
	
	num    = np.array( num )
	min_   = np.array( min_ )
	max_   = np.array( max_ )
	median = np.array( median )
	mean   = np.array( mean )
	std    = np.array( std )
	
	threshInd = 80
	
	
	
	fitToPlot = 0
	nGen = 26
	nBin = 50
	thresh = 0.0
	lineNum = 100
	alpha = 1.0
	bound = np.array([
		[-0.8,0.8],
		[-0.8,0.8],
		[-1.0,4.0]])
	
	paramFile = open('587729227151704160_combined.txt')
	for i in range(lineNum):
		paramStr = paramFile.readline().split('\t')[1]
	# end
#	print paramStr
	param_real = [float(it) for it in paramStr.split(',')]
	
	a = param_real[6]/param_real[7]
	param_real[7] = param_real[6]+param_real[7]
	param_real[6] = a
	
	num_param = len(param_real)
	
	binC_real, binV_real, binC2_real, binV2_real = solve(param_real, nBin, bound, fitToPlot, alpha)
	
	data2 = []
	data3 = []
	
	for j in range(nGen):
		
		param_start = deepcopy(param_real)
		
		"""
		# generate params
		for i in range(num_param):
			if( i < 3 ):
				param_start[i] = param_real[i]
#			elif( i < m-2 ):
#				param_start[i] = param_real[i] + np.random.normal(0,mult*std[i-2,threshInd])
#			elif( i < m ):
#				param_start[i] = param_real[i] + np.random.normal(0,0.15*mult*std[i-2,threshInd])
#			elif( i < m+1 ):
#				param_start[i] = param_real[i] + np.random.normal(0,0.01*mult*std[i-2,threshInd])
#			elif( i < m+2 ):
#				param_start[i] = param_real[i] + np.random.normal(0,0.05*mult*std[i-2,threshInd])
#			else:
#				param_start[i] = param_real[i]
			elif( i < 3+3 ):
				param_start[i] = param_real[i] + np.random.normal(0,mult*std[i-2,threshInd])
			else:
				param_start[i] = param_real[i]
		# end
		"""
		
		mult = 0.0
		ind = 3
#		param_start[ind] = param_real[ind] + mult*std[ind-2,threshInd]
		param_start[ind] = param_real[ind] + (j*2.0 - nGen)/nGen*mult*std[ind-2,threshInd]
		
		mult = 5.0000
		ind = 5
		param_start[ind] = param_real[ind] + (j*2.0 - nGen)/nGen*mult*std[ind-2,threshInd]
		
		binC, binV, binC2, binV2 = solve(param_start, nBin, bound, fitToPlot, alpha)
		
		error2, RMSE2, OvrFrac2, corr2 = ErrorFunction2D( nBin, binV2, binV2_real )
		error3, RMSE3, OvrFrac3, corr3 = ErrorFunction3D( nBin, binV, binV_real )
		
		
		data2.append([])
		data2[j] = param_start[0:m+2]
		data2[j].append(1-corr2)
		data2[j].append(1-OvrFrac2)
		data2[j].append(RMSE2)
		data2[j].append(error2)
		
		data3.append([])
		data3[j] = param_start[0:m+2]
		data3[j].append(1-corr3)
		data3[j].append(1-OvrFrac3)
		data3[j].append(RMSE3)
		data3[j].append(error3)	
		print str(data2[j][0:m+2])
		print str(data2[j][-4:]) + " " + str(j/(nGen*1.0))
		print str(data3[j][-4:]) + " " + str(j/(nGen*1.0))
		
	# end
	
	data2 = np.array(data2)
	data3 = np.array(data3)
	
	print " "
	
#	fig = plt.figure()
	
	# plot param runs
	fig, axes = plt.subplots( nrows=int(2), ncols=int(3) )
	
	ind2 = 0
	for ax in axes.flat:
		if( ind2 == 0 ):
			ax.plot(data2[:,ind],data2[:,-3], 'r-')
			ax.plot(data3[:,ind],data3[:,-3], 'b-')
			ax.set_title('ind vs. ovrfrac')
		elif( ind2 == 1 ):
			ax.plot(data2[:,ind],data2[:,-2], 'r-')
			ax.plot(data3[:,ind],data3[:,-2], 'b-')
			ax.set_title('ind vs. rmse')
		elif( ind2 == 2 ):
			ax.plot(data2[:,ind],data2[:,-1], 'r-')
			ax.plot(data3[:,ind],data3[:,-1], 'b-')
			ax.set_title('ind vs. error')
		elif( ind2 == 3 ):
			ax.plot(data2[:,-3],data3[:,-3], 'b-')
			ax.set_title('ovfrac')
		elif( ind2 == 4 ):
			ax.plot(data2[:,-2],data3[:,-2], 'b-')
			ax.set_title('rmse')
		elif( ind2 == 5 ):
			ax.plot(data2[:,-1],data3[:,-1], 'b-')
			ax.set_title('error')

		
		ind2 += 1 
	# end

	
	
	
	
	
	
	plt.show()
	
	
	
	


def solve(paramList, nBin, bound, fitToPlot, alpha):
	
	p = deepcopy(paramList)
	
	a = p[6]
	b = p[7]
	
	p[7] = b/(a+1)
	p[6] = a*p[7]
	
	#print paramList
	
	paramStr = ','.join( map(str, p[0:-2]) )
#	print paramStr
#	call("./basic_run " + paramStr + "", shell=True)
	call("./basic_run " + paramStr + " > SolveMetro.out", shell=True)
	RV = np.loadtxt("a.101")
	print len(RV)
	binC, binV, binC2, binV2 = BinField(nBin, bound, RV, fitToPlot, alpha)
	
	return binC, binV, binC2, binV2

# end

def BinField(nBin, bound, RV, fitToPlot, alpha):

	nPts = np.size(RV,0)
	
	binCnt  = np.zeros((nBin,nBin,nBin))
	binVel  = np.zeros((nBin,nBin,nBin))
	
	binCnt2 = np.zeros((nBin,nBin))
	binVel2 = np.zeros((nBin,nBin))

	if( fitToPlot == 1 ):
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
		x  = float(RV[i,0])
		y  = float(RV[i,1])
		z  = float(RV[i,2])
		vz = float(RV[i,5])
		
		ii = (x - xmin) / (xmax - xmin) * nBin
		jj = (y - ymin) / (ymax - ymin) * nBin
		kk = (z - zmin) / (zmax - zmin) * nBin
		
		if( ii > 0 and ii < nBin and jj > 0 and jj < nBin and kk > 0 and kk < nBin ):
		        binCnt[jj,ii,kk] = binCnt[jj,ii,kk] + 1
		        binVel[jj,ii,kk] = binVel[jj,ii,kk] + vz
		# end
	# end
	
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
				#	w = np.exp(-alpha * np.sum(binCnt[i,j,nBin-k-1:nBin])*dz/(dx*dy*dz)/(nPts-1) )
					w = np.exp(-alpha * np.sum(binCnt[i,j,0:k])*dz/(dx*dy*dz)/(nPts-1) )
					
					sumV += binCnt[i,j,k]*w*binVel[i,j,k]
					sumW += binCnt[i,j,k]*w
				# end
			# end
#			print sumW
			
			if( sumW > 0 ):
				binVel2[i,j] = sumV/sumW
			else:
				binVel2[i,j] = 0
			# end
		# end
	# end
	
	
	
	
	
	
	return binCnt, binVel, binCnt2, binVel2

# end

def ErrorFunction2D( nBin, Vi, V1 ):
	
	
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
	
	"""
	print "Vi"
	print Vi
	print "V1"
	print V1
	"""
	
	if( Ovr > 1 ):
		MSE = MSE/Ovr
	
	RMSE = math.sqrt(MSE)
	
	if( As+At-Ovr*1.0 == 0.0 ):
		OvrFrac = 0.0
	else:
		OvrFrac = Ovr/(As+At-Ovr*1.0)
#	error = RMSE*(1-OvrFrac)
#	error = (1-OvrFrac**2)
#	error = RMSE
	
	Bi = np.piecewise(Vi, [Vi > 0, Vi < 0, Vi == 0], [1, 1, 0])
	B1 = np.piecewise(V1, [V1 > 0, V1 < 0, V1 == 0], [1, 1, 0])
	corr = np.corrcoef( np.ndarray.flatten(Bi), np.ndarray.flatten(B1) )[1,0]
	
#	error = RMSE*math.sqrt((1-OvrFrac)*(1-corr))
	
#	error = RMSE**(1-OvrFrac)**0.5
	
	"""	
	fftVi = np.abs( np.fft.fft2(Vi) )
	fftV1 = np.abs( np.fft.fft2(V1) )
	
#	fftC = np.log(1 + fftC)
#	fftV = np.log(1 + fftV)
	
	gamma = 0.8
	
	fftVi = np.power(fftVi, gamma)
	fftV1 = np.power(fftV1, gamma)
	
	error = np.sum(np.sum( np.power(fftVi - fftV1, 2) ) )
	"""	
	
	error = RMSE*(1-OvrFrac)
	
	return error, RMSE, OvrFrac, corr

# end

def ErrorFunction3D( nBin, Vi, V1 ):
	
	
	As = 0
	At = 0
	Ovr = 0
	MSE = 0
	morph = 0
	error = 0
	
	isOvr = 0
	
	for i in range(nBin):
		for j in range(nBin):
			for k in range(nBin):
				
				if( V1[k,j,i] > 0 or V1[k,j,i] < 0 ):
					At = At + 1
					if( Vi[k,j,i] > 0 or Vi[k,j,i] < 0 ):
						As = As + 1
						Ovr = Ovr + 1
						isOvr = 1
				elif( Vi[k,j,i] > 0 or Vi[k,j,i] < 0 ):
					As = As + 1
				# end if
				
				if( isOvr == 1 ):
					MSE = MSE + ( Vi[k,j,i] - V1[k,j,i] )**2
				# end if
				
				isOvr = 0
			# end
		# end
	# end
	
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
	
#	error = RMSE**(1-OvrFrac)**0.5
	
	"""	
	fftVi = np.abs( np.fft.fft2(Vi) )
	fftV1 = np.abs( np.fft.fft2(V1) )
	
#	fftC = np.log(1 + fftC)
#	fftV = np.log(1 + fftV)
	
	gamma = 0.8
	
	fftVi = np.power(fftVi, gamma)
	fftV1 = np.power(fftV1, gamma)
	
	error = np.sum(np.sum( np.power(fftVi - fftV1, 2) ) )
	"""	
	
	error = RMSE*(1-OvrFrac)
	
	return error, RMSE, OvrFrac, corr




main()
