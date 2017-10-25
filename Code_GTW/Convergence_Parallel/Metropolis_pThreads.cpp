#include <iostream>
#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <string>
#include <cstdlib>
#include <unistd.h>
#include <math.h>
#include <pthread.h>

using namespace std;

// %%%%%%%%%%%%%%%%%%%%
// CODE BEGINS
// %%%%%%%%%%%%%%%%%%%%

void* solve(void *arg);

int *tids;
int rc;
int nThrd;
int shift;
int iter;

string targ, start, fileBase;
pthread_t *thrdid;


int main(int argc, char *argv[])
{
	fileBase = argv[1];
	nThrd = atoi(argv[2]);
	targ  = argv[3];
	start = argv[4];
	shift = atoi(argv[5]);
	
	// QUIT: TOO MANY THREADS
	if( nThrd > 10 )
	{
		cout << "Aborted run!!! Too many threads requested." << endl;
		return 0;
	}
	
	thrdid  = new pthread_t[nThrd];
	tids    = new int[nThrd];
	
	/*************
	CREATE THREADS
	*************/
	
	for ( int i = 0; i < nThrd; i++)
	{
		tids[i] = i;
		rc = pthread_create(&thrdid[i],NULL,solve,(void *)&tids[i]);
	}
	
	/***********
	JOIN THREADS
	***********/
	
	for ( int i = 0; i < nThrd; i++)
	{
		pthread_join(thrdid[i],NULL);
	}
	
	cout << "DONE" << endl;
	
}

void* solve(void *arg)
{
	int id = *((int *)arg); // integral thrd id
	int run_ = id + iter*nThrd + shift;
	string run;
	stringstream temp;
	temp << run_;
	run = temp.str();
	
	int err;
	string filename, cmnd;
	
//	filename = "Metropolis_ParamsOut_";
	
	if( run_ < 10 )
	{
		filename = fileBase + "0" + run + ".txt";
	}
	else
	{
		filename = fileBase + run + ".txt";
	}
	cout << filename << endl;
	
//	cmnd = "python Metropolis_Read.py " + filename + " " + run + " " + targ + " " + start + " > out.txt";
	cmnd = "python Metropolis_Read.py " + filename + " " + run + " " + targ + " " + start;
	
//	cout << tid << endl;
//	cout << cmnd << endl;
	
	err = system(cmnd.c_str());
	
	cout << "Done thrd: " << run << endl;
	
}


