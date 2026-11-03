#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

#define N_VECTOR 175      
#define REPS 2000         

double Serial_dot(double x[], double y[], int n) {
    int i;
    double sum = 0.0;
    for (i = 0; i < n; i++) sum += x[i] * y[i];
    return sum;
}

int main(int argc, char** argv) {
    int my_rank, p, n = 0, i, q, r;
    double* x = NULL, * y = NULL;              
    double* local_x, * local_y;
    int* counts, * displs;
    double local_dot, dot = 0.0, serial = 0.0, t_start = 0.0, t;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    if (my_rank == 0) n = N_VECTOR;
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);      

   
    counts = (int*)malloc(p * sizeof(int));
    displs = (int*)malloc(p * sizeof(int));
    for (q = 0, i = 0; q < p; q++) {
        counts[q] = n / p + (q < n % p ? 1 : 0);
        displs[q] = i;
        i += counts[q];
    }
    local_x = (double*)malloc(counts[my_rank] * sizeof(double));
    local_y = (double*)malloc(counts[my_rank] * sizeof(double));

   
    if (my_rank == 0) {
        x = (double*)malloc(n * sizeof(double));
        y = (double*)malloc(n * sizeof(double));
        srand(12345);                                   
        for (i = 0; i < n; i++) {
            x[i] = rand() % 1001 - 500;                  
            y[i] = rand() % 1001 - 500;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (my_rank == 0) t_start = MPI_Wtime();

    for (r = 0; r < REPS; r++) {
        
        if (my_rank == 0) {
            for (i = 0; i < counts[0]; i++) { local_x[i] = x[i]; local_y[i] = y[i]; }
            for (q = 1; q < p; q++) {
                MPI_Send(x + displs[q], counts[q], MPI_DOUBLE, q, 0, MPI_COMM_WORLD);
                MPI_Send(y + displs[q], counts[q], MPI_DOUBLE, q, 1, MPI_COMM_WORLD);
            }
        }
        else {
            MPI_Recv(local_x, counts[my_rank], MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
            MPI_Recv(local_y, counts[my_rank], MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &status);
        }
        
        local_dot = Serial_dot(local_x, local_y, counts[my_rank]);
        MPI_Reduce(&local_dot, &dot, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    }

    if (my_rank == 0) {
        t = (MPI_Wtime() - t_start) / REPS;             
        serial = Serial_dot(x, y, n);
        printf("Processes: %d, vector order N = %d\n", p, n);
        printf("The dot product is %.0f\n", dot);
        printf("Serial check     : %.0f\n", serial);
        printf("Time of calculating is %1.5f millisecond(s)\n\n", t * 1000.0);

        /* vectors for checking in MATLAB */
        printf("%% MATLAB check:\nX1 = [");
        for (i = 0; i < n; i++) printf("%.0f ", x[i]);
        printf("];\nX2 = [");
        for (i = 0; i < n; i++) printf("%.0f ", y[i]);
        printf("];\nRes = sum(X1.*X2)\n");
    }

    MPI_Finalize();
    return 0;
}