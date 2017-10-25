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
	
	m = 12
	
	filename = sys.argv[1]
	tid      = int(sys.argv[2])
	print filename
#	filename = 'Metropolis_ParamsOut.txt'
	
	
	weights = 0
	nStep = 300
	nBin  = 40
	dist  = 1.0
	measErr = 0.1
	alpha = 10.0
	bound = np.array([
		[-0.6,0.6],
		[-0.4,0.8]])
	targNum = 1
	
	paramFile = open('587729227151704160_combined.txt')
	
	for i in range(targNum):
		paramStr = paramFile.readline().split('\t')[1]
	# end
	param_real = [float(it) for it in paramStr.split(',')]
	a = param_real[6]/param_real[7]
	param_real[7] = param_real[6]+param_real[7]
	param_real[6] = a
	param_real.append(measErr)
	#param_real = np.array(param_real)
	
	"""
	# get real data
	# get real data
	# get real data
#	RV_real = np.loadtxt("HeartTarget_30000p.txt")
	
	targFile = "Runs30000p_587729227151704160/HeartTarget_30000p_"
	if( targNum < 10 ):
		targFile = targFile + "00" + str(targNum) + ".txt"
	elif( targNum < 100 ):
		targFile = targFile + "0"  + str(targNum) + ".txt"
	# end
	RV_real = np.loadtxt(targFile)
	binC_real, binV_real = BinField(nBin, RV_real, bound, alpha)
	"""
	
	binC_real, binV_real, binCu_real, binVu_real = solve(param_real, nBin, bound, alpha, tid)
	
	
	threshInd = 80
	sigFac    = 0.1
	
	multInit = 15.0
	multStep = 1.0 
	
	initSeed = 2648428829
	randSeed = int(10000000*np.random.uniform(0,1))
	np.random.seed(initSeed)
	
	# generate initial params
	num_param = len(param_real)
	param_start = np.zeros([num_param])
	
	# params to fit
	param_toFit = [ 2, 3, 4, 5, 6, 7, 10, 11, 12, 13 ]
#	param_toFit = [ 3, 4, 5 ]
#	param_toFit = [ 2, 6, 7, 10, 11 ]
#	param_toFit = range(2,14)
	
	# sigma for MCMC steps
	jump_sigma = np.zeros([num_param])
	pLim       = np.zeros([num_param,2])
	
	jump_sigma[0]  = 0.0
	jump_sigma[1]  = 0.0
	jump_sigma[2]  = 0.1
	
	jump_sigma[3]  = 0.02
	jump_sigma[4]  = 0.02
	jump_sigma[5]  = 0.02
	
	jump_sigma[6]  = 0.03
	jump_sigma[7]  = 0.1
	
	jump_sigma[8]  = 0.00001
	jump_sigma[9]  = 0.00001
	
	jump_sigma[10] = 0.7
	jump_sigma[11] = 0.7
	jump_sigma[12] = 0.7
	jump_sigma[13] = 0.7
	
	for i in range(m+2):
		if( i not in param_toFit ):
			jump_sigma[i] = 0
		# end
	# end
	
	print jump_sigma
	
	for i in range(num_param):
		pLim[i][0]    = -np.inf
		pLim[i][1]    =  np.inf
	# end
	pLim[-1,0] = 0.0
	jump_sigma[-1] = 0.0005
	
	for i in range(num_param):
		if( i in param_toFit ):
			param_start[i] = param_real[i] + np.random.normal(0,multInit*jump_sigma[i])
		else:
			param_start[i] = param_real[i]
	# end
	
	for i in range(num_param):
		jump_sigma[i] = multStep*jump_sigma[i]
	# end
	
	np.random.seed(randSeed)
	
	print ' ' 
	for i in range(m+2):
#		print pLim[i][0], pLim[i][1]
		print param_start[i], param_real[i]
	
	print " "
	
	
	chain, acc, max_ps, max_lp = metropolis(param_start, binV_real, nBin, bound, alpha, weights, pLim, jump_sigma, nStep, filename, tid)
	
	print " "
	print "Acceptance probability: " + str(acc)
	
	chain = np.array(chain)
	
	"""
	# plot param runs
	fig, axes = plt.subplots(nrows=3, ncols=4)
	
	ind = 0
	for ax in axes.flat:
		ax.plot(chain[:,ind+2])
		ind += 1
	
	plt.show()
	"""
	
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
	
	return error, Ovr

def log_likelihood(param, binV_real, nBin, bound, alpha, weights, tid):
	
	sig = param[-1]
	
	binC_sim, binV_sim, binC_sim_u, binV_sim_u = solve(param, nBin, bound, alpha, tid)
	
	W = GetWeights( nBin, binC_sim, binC_sim_u, weights )
	
	error, Ovr = ErrorFunction( nBin, binV_sim, binV_real, W )
	
#	ll = -Ovr*np.log(2*np.pi*sig**2)/2 - (Ovr*(RMSE**2)/(2*sig**2))
#	ll = -(Ovr*(RMSE**2)/(2*sig**2))
#	ll = -(nBin**2*(RMSE**2))
	
	# sinkala's
	ll = -nBin**2*( np.log(1+2*np.pi*sig**2)/2 + error**2/(2*sig**2) )
	
	return ll

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
	
	log_pri = log_prior(param, pLim)
#	print log_pri
	if np.isfinite(log_pri):
		log_like = log_likelihood(param, data_real, nBin, bound, alpha, weights, tid)
		return log_pri + log_like
	else:
		return -np.inf

def metropolis(start, data_real, nBin, bound, alpha, weights, pLim, jump_sigma, n, filename, tid):
	
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
	
	cur_lp = log_posterior(start, data_real, nBin, bound, alpha, weights, pLim, tid)
	
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
	wFile.write(','.join(map(str,p))+' '+str(0)+' '+str(isAcc)+' '+str(cur_lp)+'\n')
	
	rejects  = 0
	adaptDec = 0.99
	adaptInc = 0.5
	modUp    = 1
	modDown  = 1
	
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
			modUp    = 1
			modDown  = 1
			cand = cur + jumps[step]
		else:
			# decrease step
			if( rejects % 2 == 0 ):
				modDown = adaptDec**rejects
				cand = cur + modDown*jumps[step]
			# increase step
			else:
				modUp += adaptInc**rejects
				cand = cur + modUp*jumps[step]
		
#		print cand
		cand_lp = log_posterior(cand, data_real, nBin, bound, alpha, weights, pLim, tid)
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
		
		print "modUp: " + str(modUp) + " modDown: " + str(modDown) + " accMod: " + str(accMod)
		
		p = deepcopy(cand)
		a = p[6]
		b = p[7]
		p[7] = b/(a+1)
		p[6] = a*p[7]
		wFile.write(','.join(map(str,p))+' '+str(step+1)+' '+str(isAcc)+' '+str(cand_lp)+'\n')
		
		if( cur_lp > max_lp ):
			max_lp = cur_lp
			max_ps = cur
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

main()
