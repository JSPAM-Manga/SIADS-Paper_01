# Graham West
from copy import deepcopy
import sys 
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
	
	num = 11
	m = 14
	
	# read constants
	constFile = "ConstantParams.txt"
	targNum, ind, nBin, nStep, weights, alpha, multStep, measErr, bound = ReadConstants(constFile, num)
	
	# get cmnd line args, target params
	outFile = sys.argv[1]
	tid     = int(sys.argv[2])
	paramTarg_str  = sys.argv[3]
	paramStart_str = sys.argv[4]
	
	paramTarg = [float(it) for it in paramTarg_str.split(',')]
	paramTarg.append(measErr)
	
	paramStart = [float(it) for it in paramStart_str.split(',')]
	paramStart.append(measErr)
	
	num_param = len(paramStart)
	
	binC_real, binV_real, binCu_real, binVu_real = solve(paramTarg, nBin, bound, alpha, tid)
	
	# params to fit
	param_toFit = [ ind ]
	
	# sigma for MCMC steps
	jump_sigma = np.zeros([num_param])
	pLim       = np.zeros([num_param,2])
	
	stepDec = 0.05
	stepInc = 0.05
	limDec  = 1.0/10.0
	limInc  = 10.0
	
	jump_sigma[0]  = 0.0
	jump_sigma[1]  = 0.0
	jump_sigma[2]  = 0.1
	
#	jump_sigma[3]  = 0.02
	jump_sigma[3]  = 0.1
#	jump_sigma[4]  = 0.02
	jump_sigma[4]  = 0.1
#	jump_sigma[5]  = 0.02
	jump_sigma[5]  = 0.1
	
	jump_sigma[6]  = 0.03
	jump_sigma[7]  = 0.1
	
	jump_sigma[8]  = 0.00001
	jump_sigma[9]  = 0.00001
	
	jump_sigma[10] = 0.7
	jump_sigma[11] = 0.7
	jump_sigma[12] = 0.7
	jump_sigma[13] = 0.7
	
	for i in range(num_param):
		if( i not in param_toFit ):
			jump_sigma[i] = 0
		# end
	# end
	
	for i in range(num_param):
		pLim[i][0]    = -np.inf
		pLim[i][1]    =  np.inf
	# end
	pLim[-1,0] = 0.0
	jump_sigma[-1] = 0.0005
	
	for i in range(num_param):
		jump_sigma[i] = multStep*jump_sigma[i]
	# end
	
	print " "
	for i in range(m):
		print paramStart[i], paramTarg[i]
	# end
	print " "
	
	chain, acc, max_ps, max_lp = metropolis(paramStart, binV_real, nBin, bound, alpha, weights, pLim, jump_sigma, nStep, param_toFit, outFile, tid, stepDec, stepInc, limDec, limInc)
	
	
##############
#    MAIN    #
##############

def solve(paramList, nBin, bound, alpha, tid):
	
	p = deepcopy(paramList)
	
	a = p[6]
	b = p[7]
	p[7] = b/(a+1.0)
	p[6] = a*p[7]
	
	paramStr = ','.join( map(str, p[0:-2]) )
	
	if( tid < 10 ):
		basicOutFile = "0" + str(tid)
		basicUnpFile = "0" + str(tid)
	else:
		basicOutFile = str(tid)
		basicUnpFile = str(tid)
	
	call("./basic_run_unpreturbed -o " + basicOutFile + " " + paramStr + " > SolveMetro.out", shell=True)
	
	RV_u = np.loadtxt("basic_unp_"+basicOutFile+".out")
	RV = np.loadtxt("basic_"+basicOutFile+".out")
	
	dr = RV[-1,0:3]-RV_u[-1,0:3]
	for i in range(len(RV)/2+1):
		j = i + len(RV)/2
		RV_u[j,0:3] = RV_u[j,0:3] + dr
		RV_u[j,3:] = 0
	# end
	
	binC_u, binV_u = BinField(nBin, RV_u, bound, alpha)
	binC, binV = BinField(nBin, RV, bound, alpha)
	
	return binC, binV, binC_u, binV_u

def BinField(nBin, RV, bound, alpha):

	nPts = np.size(RV,0)-1
	
	binCnt  = np.zeros((nBin,nBin,nBin))
	binVel  = np.zeros((nBin,nBin,nBin))
	
	binCnt2 = np.zeros((nBin,nBin))
	binVel2 = np.zeros((nBin,nBin))

	xmin = bound[0,0]
	xmax = bound[0,1]
	ymin = bound[1,0]
	ymax = bound[1,1]
	zmin = np.min( RV[:,2] )
	zmax = np.max( RV[:,2] )
	
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
					w = np.exp(-alpha * np.sum(binCnt[i,j,nBin-k-1:nBin])*dz/(dx*dy*dz)/(nPts) )
					
					sumV += binCnt[i,j,k]*w*binVel[i,j,k]
					sumW += binCnt[i,j,k]*w
				# end
			# end
			if( np.sum(binCnt[i,j]) > 0 ):
				binVel2[i,j] = sumV/sumW
			else:
				binVel2[i,j] = 0
			# end
			binCnt2[i,j] = np.sum(binCnt[i,j,:])
		# end
	# end
	
#	return binCnt, binVel, binCnt2, binVel2
	return binCnt2, binVel2

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
	
	if( Ovr > 1 ):
		MSE = MSE/Ovr
	
	RMSE = math.sqrt(MSE)
	
	OvrFrac = Ovr/(As+At-Ovr*1.0)
#	OvrFrac = Ovr/(As)
#	error = RMSE**1.5*(1-OvrFrac**2)**0.5
	corr = np.corrcoef( np.ndarray.flatten(Vi), np.ndarray.flatten(V1) )[1,0]
#	error = RMSE*math.sqrt((1-corr)*(1-OvrFrac))
	error = RMSE*(1-OvrFrac)
	
	print "OvrF:\t" + str(OvrFrac)
	print "RMSE:\t" + str(RMSE)
	print "Corr:\t" + str(corr)
	
	return error, OvrFrac, RMSE

def log_likelihood(param, binV_real, nBin, bound, alpha, weights, tid):
	
	sig = param[-1]
	
	binC_sim, binV_sim, binC_sim_u, binV_sim_u = solve(param, nBin, bound, alpha, tid)
	
	W = GetWeights( nBin, binC_sim, binC_sim_u, weights )
	
	error, OvrFrac, RMSE = ErrorFunction( nBin, binV_sim, binV_real, W )
	
#	ll = -Ovr*np.log(2*np.pi*sig**2)/2 - (Ovr*(RMSE**2)/(2*sig**2))
#	ll = -(Ovr*(RMSE**2)/(2*sig**2))
#	ll = -(nBin**2*(RMSE**2))
	
	# sinkala's
#	ll = -nBin**2*( np.log(1+2*np.pi*sig**2)/2 + error**2/(2*sig**2) )
	ll = -( np.log(2*np.pi*sig**2)/2 + error**2/(2*sig**2) )
	
	return ll, OvrFrac, RMSE

def log_prior(param, pLim):
	
	
	inRange = 1
	# num-1: don't worry about sigma range
	for i in range(len(param)):
		if( not ( pLim[i,0] <= param[i] <= pLim[i,1] ) ):
			inRange = inRange*0
#		print i, pLim[i,0] <= param[i] <= pLim[i,1], pLim[i,0], param[i], pLim[i,1]
#	print inRange
	if( inRange ):
		return 0
	else:
		return -np.inf

def log_posterior(param, data_real, nBin, bound, alpha, weights, pLim, tid):
	
	"""
	log_pri = log_prior(param, pLim)
#	print log_pri
	if np.isfinite(log_pri):
		log_like, Ovr, RMSE = log_likelihood(param, data_real, nBin, bound, alpha, weights, tid)
		return log_pri + log_like
	else:
		return -np.inf
	"""
	log_like, OvrFrac, RMSE = log_likelihood(param, data_real, nBin, bound, alpha, weights, tid)
	
	return log_like, OvrFrac, RMSE
	
# end

def metropolis(start, data_real, nBin, bound, alpha, weights, pLim, jump_sigma, n, param_toFit, filename, tid, stepDec, stepInc, limDec, limInc):
	
	cov = np.diag(jump_sigma**2)
	zero = np.zeros(len(jump_sigma))
	
	# Counter for number of accepted jumps
	n_accept = 0.
	
	
	# Create a chain and add start position to it
	chain = np.array([start,])
	#chain = [ start ]
	
	# Draw n random samples from Gaussian proposal distribution to use for 
	# determining step size
	jumps = np.random.multivariate_normal(mean=zero,
	                                      cov=cov, 
	                                      size=n)
	
	for i in range(n):
		for j in range(len(start)):
			if( jump_sigma[j] == 0.0 ):
				jumps[i][j] = 0.0
	
	# Draw n random samples from U(0, 1) for determining acceptance
	y_i = np.random.uniform(low=0, high=1, size=n)
	
	max_lp = -np.inf
	max_ps = start
	
	cur_lp, cur_OvrFrac, cur_RMSE = log_posterior(start, data_real, nBin, bound, alpha, weights, pLim, tid)
	
	# Get current position of chain
	cur = chain[-1]
	
#	print "start",  cur[:15]
	
	isAcc = 1
	
	wFile = open(filename, 'w')
	p = deepcopy(cur)
	a = p[6]
	b = p[7]
	p[7] = b/(a+1)
	p[6] = a*p[7]
	wFile.write(" ".join(map(str,p))+" "+str(0)+" "+str(isAcc)+" "+str(cur_OvrFrac)+" "+str(cur_RMSE)+" "+str(cur_lp)+"\n")
	
	rejects  = 0
	currInc = 1.0
	currDec = 1.0
	
	modRate  = 0.5
	minMod   = 0.8
	# Random walk
	for step in range(n):
		print "step: ", step
		
		"""
		print "cand", cand[:15]
		print "cur",  cur[:15]
		print "jump", jumps[step,:15]
		"""
		
		# Get current position of chain
		cur = chain[-1]
			
		if( rejects == 0 ):
			currDec = 1.0
			currInc = 1.0
			cand = cur + jumps[step]
		else:
			# decrease step
			if( rejects % 3 == 1 ):
				if( currDec*(1.0-stepDec) < limDec ):
					currDec = limDec
				else:
					currDec = currDec*(1-stepDec)
				# end
				cand = cur + currDec*jumps[step]
			# increase step
			if( rejects % 3 == 2 ):
				if( currInc*(1.0+stepInc) > limInc ):
					currInc = limInc
				else:
					currInc = currInc*(1+stepInc)
				# end
				cand = cur + currInc*jumps[step]
			# control
			else:
				cand = cur + jumps[step]
			# end
		
#		print cand
		cand_lp, cand_OvrFrac, cand_RMSE = log_posterior(cand, data_real, nBin, bound, alpha, weights, pLim, tid)
#		print cand
		
		# Calculate acccept prob
		acc_prob = np.exp(cand_lp - cur_lp)
		accMod = (minMod + (1-minMod)*math.exp(-modRate*rejects))
		
		# Accept candidate if y_i <= alpha
		if( y_i[step]*accMod <= acc_prob ):
			rejects = 0
			n_accept += 1
			cur = cand
			cur_lp = cand_lp
			print "log-p:\t" + str(cur_lp)
			chain = np.append(chain, [cur,], axis=0)
			isAcc = 1
			#chain.append(cand)
			print "      accepted " + str(tid)
		else:
			rejects += 1
			chain = np.append(chain, [cur,], axis=0)
			isAcc = 0
			print "log-p:\t" + str(cand_lp)
			print "rejected " + str(tid)
		# end
		
#		print "modUp: " + str(modUp) + " modDown: " + str(modDown) + " accMod: " + str(accMod)
		
		p = deepcopy(cand)
		a = p[6]
		b = p[7]
		p[7] = b/(a+1)
		p[6] = a*p[7]
		wFile.write(" ".join(map(str,p))+" "+str(step+1)+" "+str(isAcc)+" "+str(cand_OvrFrac)+" "+str(cand_RMSE)+" "+str(cur_lp)+"\n")
		
		if( cur_lp > max_lp ):
			max_lp = cur_lp
			max_ps = cur
		# end
	# end
	
	wFile.close()
	
	# Acceptance rate
	acc_rate = (100*n_accept/n)
		
	return [chain, acc_rate, max_ps, max_lp]

def neg_log_posterior(param, data_sim, data_real, nBin, pLim):
    
    return -log_posterior(param, data_sim, data_real, nBin, pLim)

def equals(a, b):
	
	n = len(a)
	
	test = 1
	for i in range(n):
		if( a[i] != b[i] ):
			test = test*0
	
	return test

def print2D(A):
	
	print('\n'.join([''.join(['{:4}'.format(item) for item in row]) for row in A]))

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
	
	return targNum, ind, nBin, nStep, weights, alpha, multStep, measErr, bound

# end

main()
