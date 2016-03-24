My implementation of bucket sort using the openmpi library.  I was able to sort
1 billion doubles in about 24 seconds with openmpi and 32 processors.  The naive solution with no openmpi took about 400
seconds (or 6.6 minutes).  This is incerdibly significant and useful in computer
science. 

I later used my knowledge of openmp to create a hybrid program that used both
openmp and openmpi.  I was able to sort 1 billion doubles on 32 processors in
about 20 seconds.  
