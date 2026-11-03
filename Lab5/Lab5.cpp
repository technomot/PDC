
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mpi.h"

#define N 17          
#define REPS 500      

int main(int argc, char** argv) {
    int my_rank, p, i, j, k, q, r;
    static double A[N * N], B[N * N], C[N * N], D[N * N];
    int rows[64], first[64];
    double* local_A, * local_C;
    double t_start = 0.0, t, maxdiff = 0.0;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    
    for (q = 0, i = 0; q < p; q++) {
        rows[q] = N / p + (q < N % p ? 1 : 0);
        first[q] = i;
        i += rows[q];
    }
    local_A = (double*)malloc(rows[my_rank] * N * sizeof(double) + 8);
    local_C = (double*)malloc(rows[my_rank] * N * sizeof(double) + 8);

   
    if (my_rank == 0) {
        srand(12345);
        for (i = 0; i < N * N; i++) A[i] = rand() % 10;
        for (i = 0; i < N * N; i++) B[i] = rand() % 10;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (my_rank == 0) t_start = MPI_Wtime();

    for (r = 0; r < REPS; r++) {
        
        MPI_Bcast(B, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        
        if (my_rank == 0) {
            for (i = 0; i < rows[0] * N; i++) local_A[i] = A[i];
            for (q = 1; q < p; q++)
                MPI_Send(A + first[q] * N, rows[q] * N, MPI_DOUBLE, q, 0, MPI_COMM_WORLD);
        }
        else {
            MPI_Recv(local_A, rows[my_rank] * N, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
        }

        
        for (i = 0; i < rows[my_rank]; i++)
            for (j = 0; j < N; j++) {
                double s = 0.0;
                for (k = 0; k < N; k++) s += local_A[i * N + k] * B[k * N + j];
                local_C[i * N + j] = s;
            }

        if (my_rank == 0) {
            for (i = 0; i < rows[0] * N; i++) C[i] = local_C[i];
            for (q = 1; q < p; q++)
                MPI_Recv(C + first[q] * N, rows[q] * N, MPI_DOUBLE, q, 1, MPI_COMM_WORLD, &status);
        }
        else {
            MPI_Send(local_C, rows[my_rank] * N, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
        }
    }

    if (my_rank == 0) {
        t = (MPI_Wtime() - t_start) / REPS;               

        for (i = 0; i < N; i++)
            for (j = 0; j < N; j++) {
                double s = 0.0;
                for (k = 0; k < N; k++) s += A[i * N + k] * B[k * N + j];
                D[i * N + j] = s;
                if (fabs(s - C[i * N + j]) > maxdiff) maxdiff = fabs(s - C[i * N + j]);
            }

        printf("Processes: %d, matrices %d x %d\n", p, N, N);
        printf("Max difference with serial multiplication: %g\n", maxdiff);
        printf("Time of calculating is %1.5f millisecond(s)\n\n", t * 1000.0);

        printf("The 1st matrix A:\n");
        for (i = 0; i < N; i++) { for (j = 0; j < N; j++) printf("%4.0f", A[i * N + j]); printf("\n"); }
        printf("The 2nd matrix B:\n");
        for (i = 0; i < N; i++) { for (j = 0; j < N; j++) printf("%4.0f", B[i * N + j]); printf("\n"); }
        printf("The product C = A*B:\n");
        for (i = 0; i < N; i++) { for (j = 0; j < N; j++) printf("%6.0f", C[i * N + j]); printf("\n"); }

        
        printf("\n%% MATLAB check:\nA = [");
        for (i = 0; i < N; i++) { for (j = 0; j < N; j++) printf("%.0f ", A[i * N + j]); printf(i < N - 1 ? "; " : ""); }
        printf("];\nB = [");
        for (i = 0; i < N; i++) { for (j = 0; j < N; j++) printf("%.0f ", B[i * N + j]); printf(i < N - 1 ? "; " : ""); }
        printf("];\nC = [");
        for (i = 0; i < N; i++) { for (j = 0; j < N; j++) printf("%.0f ", C[i * N + j]); printf(i < N - 1 ? "; " : ""); }
        printf("];\nD = A*B;\nmax(max(abs(C-D)))\nC*inv(D)\n");
    }

    MPI_Finalize();
    return 0;
}