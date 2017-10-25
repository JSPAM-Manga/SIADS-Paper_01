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
	
	thresh = 0.005
	plotBin = 0
	direct = "Runs_p03_06/"
	direct = "Runs_p05_02/"
	direct = "Runs_p06_01/"
#	direct = ""
	
	# read constants
	constFile = direct + "ConstantParams.txt"
	targNum, ind, temp, nStep, nGen, redos, weights, alpha, multStep, measErr, bound = ReadConstants(constFile, 11)
	
	nBin = nStep+1
	
	paramFile = open("587729227151704160_combined.txt")
	for i in range(targNum):
		paramTarg_str = paramFile.readline().split('\t')[1]
	# end
	paramTarg = [float(it) for it in paramTarg_str.split(',')]
	
	conv  = np.zeros(nGen)
	conv2 = np.zeros((nGen,nStep+1))
	conv3  = np.zeros(nStep+1)
	ovrf  = np.zeros(nGen)
	rmse  = np.zeros(nGen)
	error = np.zeros(nGen)
	param = np.zeros(nGen)
	chains = []
	
	metOutFileBase = direct + "Metropolis_Output_p"
	if( ind < 10 ):
		metOutFileBase += "0" + str(ind) + "_"
	else:
		metOutFileBase += str(ind) + "_"
	# end
	
	for j in range(nGen):
		
		if( j < 10 ):
			metOutFile = metOutFileBase + "g0" + str(j) + "_r"
		else:
			metOutFile = metOutFileBase + "g" + str(j) + "_r"
		# end
		
		for k in range(redos):
			if( k < 10 ):
				metOutFile_2 = metOutFile + "0" + str(k) + ".txt"
			else:
				metOutFile_2 = metOutFile + str(k) + ".txt"
			# end
#			print metOutFile_2
			
			a, b, c, d = GetConv(metOutFile_2, nStep, ind, paramTarg[ind], thresh)
			conv2[j,:] += a
			if( k == 0 ):
				ovrf[j]  = 1.0 - b
				rmse[j]  = c
				error[j] = (1.0 - b)*c
				param[j] = d[0]
#				chains.append(d)
			# end
			chains.append(d)
		# end
		conv2[j,:] /= (redos*1.0)
		conv[j] = np.sum(conv2[j,:])/(nStep+1.0)
	# end
	trueP = paramTarg[ind]*np.ones(nStep+1)
	chains = np.array(chains)
	
	binChain = pow(BinField(nBin, chains), 0.5)
	binChain = np.log(BinField(nBin, chains)+1.0)
	
	halfStep  = []
	wholeStep = []
	for i in range(nGen):
		halfStep.append(-1)
		wholeStep.append(-1)
		
		for j in range(nStep+1):
			if( conv2[i,j] >= 0.5 and halfStep[i] == -1 ):
				halfStep[i] = j
			# end
			if( conv2[i,j] == 1.0 and wholeStep[i] == -1 ):
				wholeStep[i] = j
			# end
		# end
	# end
	
	
	
	
	# plot param runs
	fig, axes = plt.subplots( nrows=int(2), ncols=int(3) )
	
	ind2 = 0
	for ax in axes.flat:
		if( ind2 == 0 ):
			ax.plot(param, error, 'b-')
			ax.set_title('error vs. param')
		elif( ind2 == 1 ):
			if( plotBin == 1 ):
				ax.imshow(conv2,interpolation='none')
			else:
				for i in range(nGen):
					ax.plot(conv2[i,:], 'b-')
				# end
			# end
			ax.set_title('conv vs. step')
		elif( ind2 == 2 ):
			ax.plot(param, halfStep, 'b-')
			ax.plot(param, wholeStep, 'r-')
			ax.set_title('half(b)/whole(r) vs. param')
		elif( ind2 == 3 ):
			ax.plot(param, conv, 'b-')
			ax.set_title('conv vs. param')
		elif( ind2 == 4 ):
			ax.plot(error, conv, 'b.')
			ax.set_title('conv vs. error')
		elif( ind2 == 5 ):
			if( plotBin == 1 ):
				ax.imshow(binChain, interpolation='none')
			else:
				for it in chains:
					ax.plot(it, 'b-')
				# end
				ax.plot(trueP, 'r-')
				ax.set_title('chains')
			# end
		
		ind2 += 1 
	# end

	
	plt.tight_layout(w_pad=-1, h_pad=-0.5)
	
	plt.show()
	
	
##############
#    MAIN    #
##############

def GetConv(filename, nStep, ind, trueP, thresh):
	
	isConv = 0
	conv = []
	
	stuff  = np.loadtxt(filename)
	params = deepcopy(stuff[:,0:14])
	output = deepcopy(stuff[:,-5:])
	
	accepted = []
	
	init = output[0,:]
	for i in range(nStep+1):
		if( output[i,1] == 1 ):
			w = params[i,ind]
			x = output[i,:]
		# end
		accepted.append(w)
		
		if( (1.0-x[2])*x[3] < thresh ):
			isConv = 1
		# end
		
		if( isConv == 1 ):
			conv.append(1.0)
		else:
			conv.append(0.0)
		# end
	# end
	conv = np.array(conv)
	
	return conv, init[2], init[3], accepted
	
# end

def BinField(nBin, chains):
	
	nChain = len(chains)
	nStep  = len(chains[0])
	
	binCnt  = np.zeros((nBin,nBin))
	
	xmin = 0
	xmax = nStep-1
	ymin = min(chains[:,0])
	ymax = max(chains[:,0])
	
	dx = (xmax-xmin)/nBin
	dy = (ymax-ymin)/nBin
	
	for it in chains:
		for i in range(nStep):
			x = 0.9999999*i
			y = it[i]
			
			ii = (x - xmin) / (xmax - xmin) * nBin
			jj = (y - ymin) / (ymax - ymin) * nBin
			
			if( ii >= 0 and ii < nBin and jj >= 0 and jj < nBin ):
			        binCnt[jj,ii] = binCnt[jj,ii] + 1
			# end
		# end
	# end
		
	return binCnt

# end

def ReadConstants(filename, num):
	
	rFile = open(filename)
	for i in range(num):
		string = rFile.readline().split(' ')
		if( string[0] == "targNum" ):
			targNum = int(string[1])
		elif( string[0] == "ind" ):
			ind = int(string[1])
		elif( string[0] == "nBin" ):
			nBin = int(string[1])
		elif( string[0] == "nStep" ):
			nStep = int(string[1])
		elif( string[0] == "nGen" ):
			nGen = int(string[1])
		elif( string[0] == "redos" ):
			redos = int(string[1])
		elif( string[0] == "weights" ):
			weights = int(string[1])
		elif( string[0] == "alpha" ):
			alpha = float(string[1])
		elif( string[0] == "multStep" ):
			multStep = float(string[1])
		elif( string[0] == "measErr" ):
			measErr = float(string[1])
		elif( string[0] == "bound" ):
			stuff = map( float, string[1].split(',') )
			bound = []
			bound.append( [stuff[0], stuff[1] ] )
			bound.append( [stuff[2], stuff[3] ] )
			bound.append( [stuff[4], stuff[5] ] )
			bound = np.array(bound)
		# end
	# end
	
	return targNum, ind, nBin, nStep, nGen, redos, weights, alpha, multStep, measErr, bound

# end






main()
