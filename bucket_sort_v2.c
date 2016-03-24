#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include <math.h>

#define EPSILON 0.0000000000001

// Comparison function used by qsort
int compare_floats(const void* arg1, const void* arg2) {
    float a1 = *(float *) arg1;
    float a2 = *(float *) arg2;
    if (a1 < a2) return -1;
    else if (a1 == a2) return 0;
    else return 1;
}

void qsort_floats(float *array, int array_len) {
    qsort(array, (size_t)array_len, sizeof(float), compare_floats);
}

int main(int argc, char *argv[]) {
    const int root = 0;
    int comm_sz, my_rank, N, littleN;
    double t1, t2, total=0;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    if (argc != 2) {
        printf("Please include only the relevant data, # of processes and N");
        return 1;
    }
    N = atoi(argv[1]);
    littleN = N/comm_sz;  // assuming (N/p) isn't truncated
    srand(time(NULL) + my_rank);

    if (my_rank == root) {
        printf("# of processes is %d, # of elements in N is %d\n", comm_sz, N);
    }

/************************************************************************************************ 
 *   generate 
 ************************************************************************************************/
    t1 = MPI_Wtime();
    float* localA = malloc(littleN*sizeof(float));
    int i;
    for (i = 0; i<littleN; i++) {
        float r = (float)rand() / RAND_MAX; 
        localA[i] = r;
    } 
    t2 = MPI_Wtime();
    if (my_rank == root) {
        total += t2-t1;
        printf("Generating took %f seconds.\n", t2-t1);
    }

/************************************************************************************************ 
 *   bin
 ************************************************************************************************/
    t1 = MPI_Wtime();
    float* localBuckets = malloc(littleN*sizeof(float));
    int* localBucketsCount = malloc(comm_sz*sizeof(int));
    int j, count = 0, index=0;
    for (i=0; i<comm_sz; i++) {
        for (j=0; j<littleN; j++) {
            if (((float)i / (float)comm_sz) <= localA[j] && localA[j] < ((float)(i+1) / (float)comm_sz)) {
                localBuckets[index] = localA[j];
                count += 1;
                index += 1;
            } else if ((i+1) == comm_sz && localA[j] == 1) {
                localBuckets[index] = localA[j];
                count += 1;
                index += 1;
            }
        }
        localBucketsCount[i] = count; 
        count = 0;
    }
    free(localA);
    int* localBucketsDisp = malloc(comm_sz*sizeof(int));
    localBucketsDisp[0] = 0;
    for (i=1; i<comm_sz; i++)
        localBucketsDisp[i] = localBucketsCount[i-1] + localBucketsDisp[i-1]; 
    
    t2 = MPI_Wtime();
    if (my_rank == root) {
        total += t2-t1;
        printf("Binning took %f seconds.\n", t2-t1);
    }

    MPI_Barrier(MPI_COMM_WORLD);

/************************************************************************************************ 
 *   distribute
 ************************************************************************************************/
    t1 = MPI_Wtime();

    // distribute the number of elements each process should expect from all the other processes
    int* recvBucketsCount = malloc(comm_sz*sizeof(int));
    int* recvBucketsDisp = malloc(comm_sz*sizeof(int));
    MPI_Alltoall(localBucketsCount, 1, MPI_INT, recvBucketsCount, 1, MPI_INT, MPI_COMM_WORLD);

    // add all the counts from the process up
    int recvCount[1] = {0};
    for(i=0; i<comm_sz; i++) {
        recvCount[0] += recvBucketsCount[i];
        if (i==0) {recvBucketsDisp[i] = 0;}
        else {recvBucketsDisp[i] = recvBucketsDisp[i-1]+recvBucketsCount[i-1];}
    }

    float* recvBuckets = malloc(recvCount[0]*sizeof(float));
    // send floats so each process builds their respected bucket
    MPI_Alltoallv(localBuckets, localBucketsCount, localBucketsDisp, MPI_FLOAT, recvBuckets,
                  recvBucketsCount, recvBucketsDisp, MPI_FLOAT, MPI_COMM_WORLD);

    free(localBuckets);
    free(localBucketsCount);
    free(localBucketsDisp);
    free(recvBucketsCount);
    free(recvBucketsDisp);

    t2 = MPI_Wtime();
    if (my_rank == root) {
        total += t2-t1;
        printf("Distributing took %f seconds.\n", t2-t1);
    }

/************************************************************************************************ 
 *   local sort 
 ************************************************************************************************/
    t1 = MPI_Wtime();
    qsort_floats(recvBuckets, recvCount[0]); 
    t2 = MPI_Wtime();
    if (my_rank == root) {
        total += t2-t1;
        printf("Local sort took %f seconds.\n", t2-t1);
    }
 

/************************************************************************************************ 
 *   gather 
 ************************************************************************************************/
    t1 = MPI_Wtime();
    float* out = malloc(N*sizeof(float));
    int* outCount = malloc(comm_sz*sizeof(int));
    int* outDisp = malloc(comm_sz*sizeof(int));
    

    // gather each process' count
    MPI_Gather(recvCount, 1, MPI_INT, outCount, 1, MPI_INT, 
               root, MPI_COMM_WORLD);

    // figure out disp, given count
    outDisp[0] = 0;
    for (i=1; i<comm_sz; i++)
        outDisp[i] = outCount[i-1] + outDisp[i-1]; 

    // gather each process' bucket 
    MPI_Gatherv(recvBuckets, recvCount[0], MPI_FLOAT, out, outCount, 
                outDisp, MPI_FLOAT, root, MPI_COMM_WORLD);
    free(recvBuckets);
    free(outCount);
    free(outDisp);
    t2 = MPI_Wtime();
    if (my_rank == root) {
        total += t2-t1;
        printf("Gathering took %f seconds.\n", t2-t1);
    }
    

/************************************************************************************************ 
 *   check if my solution is correct 
 ************************************************************************************************/
    if (my_rank == root) {
        printf("Total time is %f seconds.\n", total);
        int j, error = 0, i=0;
        for (j=1; j<N; j++) {
            if (out[j] < out[j-1] && fabs(out[j]-out[j-1]) > EPSILON) {
                error = 1;
                // printf("****************ERROR******* %f >= %f\n", out[j], out[j-1]); 
            }
        }
        if (error == 0) {printf("Correct: all elements of the solution are in ascending order.\n");}
        else {printf("Incorrect: not all elements of the solution are in ascending order.\n");}
        printf("\n\n");
    }
    free(out);
    MPI_Finalize();
    return 0;
}
