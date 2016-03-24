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
    int comm_sz, my_rank, N;
    double t1, t2, total=0;
    float* A = NULL;
    float* buckets = NULL;
    int* bucketsCount = NULL;
    int* bucketsDisp = NULL;
    int* recvBucketsIndex;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    if (argc != 2) {
        printf("Please include only the relevant data, # of processes and N");
        return 1;
    }
    N = atoi(argv[1]);
    srand(time(NULL) + my_rank);

/************************************************************************************************ 
 *   generate
 ************************************************************************************************/

    t1 = MPI_Wtime();
    if (my_rank == root) {
        printf("# of processes is %d, # of elements in N is %d\n", comm_sz, N);
        A = malloc(N*sizeof(float));        
        int i;
        for (i = 0; i<N; i++) {
            float r = (float)rand() / RAND_MAX; 
            A[i] = r;
        } 
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
    if (my_rank == root) {
        bucketsCount = malloc(comm_sz*sizeof(int));
        buckets = malloc(N*sizeof(float));
        int i, j, count = 0, index=0;
        for (i=0; i<comm_sz; i++) {
            for (j=0; j<N; j++) {
                if (((float)i / (float)comm_sz) <= A[j] && A[j] < ((float)(i+1) / (float)comm_sz)) {
                    buckets[index] = A[j];
                    count += 1;
                    index += 1;
                } else if ((i+1) == comm_sz && A[j] == 1) {
                    buckets[index] = A[j];
                    count += 1;
                    index += 1;
                }
            }
            bucketsCount[i] = count; 
            count = 0;
        }
        free(A);
        bucketsDisp = malloc(comm_sz*sizeof(int));
        bucketsDisp[0] = 0;
        for (i=1; i<comm_sz; i++)
            bucketsDisp[i] = bucketsCount[i-1] + bucketsDisp[i-1]; 
    }
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
    // distribute the number of elements each process should expect
    recvBucketsIndex = malloc(1*sizeof(int));
    MPI_Scatter(bucketsCount, 1, MPI_INT, recvBucketsIndex, 1, MPI_INT, root, MPI_COMM_WORLD);
    // make an array to recieve each process' floats
    float* localBucket = malloc(recvBucketsIndex[0]*sizeof(float));
    // distribute the elements for each process
    MPI_Scatterv(buckets, bucketsCount, bucketsDisp, MPI_FLOAT, localBucket, 
                 recvBucketsIndex[0], MPI_FLOAT, root, MPI_COMM_WORLD);
    
    t2 = MPI_Wtime();
    if (my_rank == root) {
        total += t2-t1;
        printf("Distributing took %f seconds.\n", t2-t1);
    }

    // local sort
    t1 = MPI_Wtime();
    qsort_floats(localBucket, recvBucketsIndex[0]);
    t2 = MPI_Wtime();
    if (my_rank == root) {
        total += t2-t1;
        printf("Local sort took %f seconds.\n", t2-t1);
    }
 
    MPI_Barrier(MPI_COMM_WORLD);


/************************************************************************************************ 
 *   gather 
 ************************************************************************************************/
    t1 = MPI_Wtime();

    float* out = malloc(N*sizeof(float));

    MPI_Gatherv(localBucket, recvBucketsIndex[0], MPI_INT, out, bucketsCount, 
                bucketsDisp, MPI_INT, root, MPI_COMM_WORLD);
    free(localBucket);
    free(recvBucketsIndex);
    t2 = MPI_Wtime();
    if (my_rank == root) {
        total += t2-t1;
        printf("Gathering took %f seconds.\n", t2-t1);
    }
    

/************************************************************************************************ 
 *   check if solution is correct 
 ************************************************************************************************/
    if (my_rank == root) {
        free(bucketsCount);
        free(bucketsDisp);
        printf("Total time is %f seconds.\n", total);
        int j, error = 0, i=0;
        for (j=1; j<N; j++) {
            if (out[j] < out[j-1] && fabs(out[j]-out[j-1]) > EPSILON) {
                error = 1;
                // printf("****************ERROR******* %f != %f\n", out[j], out[j-1]); 
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
