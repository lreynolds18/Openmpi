#include <omp.h>
#include <stdio.h>


int main(int argc, char* argv[]) {
    int tc = atoi(argv[0]);
    int* a = malloc(10*sizeof(int));
    int b[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    int i;

    # pragma omp parallel for num_threads(tc) default(none) \
        private(i) shared(a)
    for (i=0; i<10; i++) {
        a[i]++; 
    }


    for (i=0; i<10; i++) {
        printf("a = %d\n", a[i]);
    }
    return 0;
}
