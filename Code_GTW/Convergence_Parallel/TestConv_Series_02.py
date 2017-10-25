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
	
	# set constants
	nGen = 26
	num_param = 14
	width = 0.000
	
	targNum = 100
	ind = 2
	nBin = 10
	nStep = 300
	weights = 0
	alpha = 0.0
	multStep = 1.0
	measErr = 0.1
	bound = np.array([
		[-0.8,0.8],
		[-0.8,0.8],
		[-1.0,4.0]])
	
	constParams = {
		"targNum" : targNum,
		"ind" : ind,
		"nBin" : nBin,
		"nStep" : nStep,
		"weights" : weights,
		"alpha" : alpha,
		"multStep" : multStep,
		"measErr" : measErr,
		"bound" : bound}
	
	wFile = open("ConstantParams.txt", 'w')
	for it in constParams.keys():
		if( not it == "bound" ):
			wFile.write(it + " " + str(constParams[it]) + "\n")
		else:
			wFile.write(it + " " + str(bound[0,0]) + "," + str(bound[0,1]) + "," + str(bound[1,0]) + "," + str(bound[1,1]) + "," + str(bound[2,0]) + "," + str(bound[2,1]) + "\n")
		# end
	# end
	wFile.close()
	
	
	# get target data
	paramFile = open('587729227151704160_combined.txt')
	for i in range(targNum):
		paramTarg_str = paramFile.readline().split('\t')[1]
	# end
	paramTarg = [float(it) for it in paramTarg_str.split(',')]
	a = paramTarg[6]/paramTarg[7]
	paramTarg[7] = paramTarg[6]+paramTarg[7]
	paramTarg[6] = a
	binC_real, binV_real, binCu_real, binVu_real = solve(paramTarg, nBin, bound, alpha)
	
	# all data
	data = []
	
	# run test
	for j in range(nGen):
		
		paramStart = deepcopy(paramTarg)
		paramStart[ind] = paramTarg[ind] + (j*2.0 - nGen)/nGen/2.0*width
		
		
		binC, binV, binCu, binVu = solve(paramStart, nBin, bound, alpha)
		W = GetWeights( nBin, binC, binCu, weights )
		error, RMSE, OvrFrac = ErrorFunction( nBin, binV, binV_real, W )
		
		paramTarg_str = ','.join( map( str, paramTarg) )
		paramStart_str = ','.join( map( str, paramStart) )
		
		cmnd = "python Metropolis_Read.py out.txt 0 " + paramTarg_str.strip() + " " + paramStart_str.strip()
		call(cmnd, shell=True)
		
		data.append([])
		data[j] = param_start[0:num_param]
		data[j].append(1-OvrFrac)
		data[j].append(RMSE)
		data[j].append(error)
		
		print str(data[j][0:num_param])
		print str(data[j][-3:]) + " " + str(j/(nGen*1.0))
		
	# end
	data = np.array(data)
	print " "
	
	"""
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
	"""
	
	
	
	
	
	
	plt.show()
	
	
	
	


def solve(paramList, nBin, bound, alpha):
	
	p = deepcopy(paramList)
	
	a = p[6]
	b = p[7]
	p[7] = b/(a+1.0)
	p[6] = a*p[7]
	
	paramStr = ','.join( map(str, p[0:-2]) )
	
	call("./basic_run_unpreturbed " + paramStr + " > SolveMetro.out", shell=True)
	
	RV_u = np.loadtxt("a.000")
	RV = np.loadtxt("a.101")
	
	dr = RV[-1,0:3]-RV_u[-1,0:3]
	for i in range(len(RV)/2+1):
		j = i + len(RV)/2
		RV_u[j,0:3] = RV_u[j,0:3] + dr
		RV_u[j,3:] = 0
	# end
	
	binC_u, binV_u = BinField(nBin, RV_u, bound, alpha)
	binC, binV = BinField(nBin, RV, bound, alpha)
	
	return binC, binV, binC_u, binV_u
	
# end

def BinField(nBin, RV, bound, alpha):

	nPts = np.size(RV,0)
	
	binCnt  = np.zeros((nBin,nBin,nBin))
	binVel  = np.zeros((nBin,nBin,nBin))
	
	binCnt2 = np.zeros((nBin,nBin))
	binVel2 = np.zeros((nBin,nBin))

	xmin = bound[0,0]
	xmax = bound[0,1]
	ymin = bound[1,0]
	ymax = bound[1,1]
	zmin = bound[2,0]
	zmax = bound[2,1]
#	zmin = np.min( RV[:,2] )
#	zmax = np.max( RV[:,2] )
	
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
			binCnt2[i,j] = np.sum(binCnt[i,j,:])
		# end
	# end
	
	return binCnt2, binVel2

# end

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
			#	print vv, nw
			# end
		# end
		
		W = W/np.sum(np.sum(W))*nBin**2
	else:
		W = np.ones((nBin,nBin))
	# end
	
	return W

def ErrorFunction( nBin, Vi, V1, W ):
	
	
	As = 0
	At = 0
	Ovr = 0
	MSE = 0
	morph = 0
	error = 0
	isOvr = 0
	
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
				MSE = MSE + (( Vi[j,i] - V1[j,i] )**2)*W[j,i]
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
	# end
	
	RMSE = math.sqrt(MSE)
	
	if( As+At-Ovr*1.0 == 0.0 ):
		OvrFrac = 0.0
	else:
		OvrFrac = Ovr/(As+At-Ovr*1.0)
	# end
	
	error = RMSE*(1-OvrFrac)
	
	return error, RMSE, OvrFrac

# end

def paramDist(curr, targ, n):
	
	eps = 0.00000001
	dist = 0
	
	for i in range(n):
		dist += abs(curr[i]-targ[i])/(abs(targ[i])+eps)
	# end
	
	return dist

main()
