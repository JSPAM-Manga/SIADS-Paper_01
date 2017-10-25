import numpy as np

rFile = open("ConstantParams.txt")
for i in range(8):
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
	elif( string[0] == "measErr" ):
		measErr = float(string[1])
	elif( string[0] == "bound" ):
		bound = []
		stuff = map( float, string[1].split(',') )
		bound.append( [stuff[0], stuff[1] ] )
		bound.append( [stuff[2], stuff[3] ] )
		bound.append( [stuff[4], stuff[5] ] )
		bound = np.array(bound)
	# end
# end

print bound



