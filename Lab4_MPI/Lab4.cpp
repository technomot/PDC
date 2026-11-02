#define _CRT_SECURE_NO_WARNINGS

#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int p, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const int m = 80;
    const int n = 40;
    const int Osn = 4;
    const int REPS = 10000;

    if (m % p != 0 || n % p != 0) {
        if (rank == 0)
            printf("Error: p=%d must divide m=%d and n=%d. Valid p: 1,2,4,5,8,10,20,40\n",
                p, m, n);
        MPI_Finalize();
        return 1;
    }

    const int local_m = m / p;
    const int local_n = n / p;

    std::vector<float> A, x, y, y_ref;
    if (rank == 0) {
        A.resize(m * n);
        x.resize(n);
        y.resize(m);
        y_ref.resize(m);

        srand(1);
        for (int i = 0; i < m * n; i++) A[i] = (float)(rand() % Osn);
        for (int j = 0; j < n; j++)     x[j] = (float)(rand() % Osn);

        FILE* fa = fopen("A.txt", "w");
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) fprintf(fa, "%.0f ", A[i * n + j]);
            fprintf(fa, "\n");
        }
        fclose(fa);

        FILE* fx = fopen("x.txt", "w");
        for (int j = 0; j < n; j++) fprintf(fx, "%.0f ", x[j]);
        fclose(fx);

        for (int i = 0; i < m; i++) {
            float s = 0;
            for (int j = 0; j < n; j++) s += A[i * n + j] * x[j];
            y_ref[i] = s;
        }
    }

    std::vector<float> local_A(local_m * n);
    std::vector<float> local_x(local_n);
    std::vector<float> global_x(n);
    std::vector<float> local_y(local_m);

    MPI_Scatter(A.data(), local_m * n, MPI_FLOAT,
        local_A.data(), local_m * n, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Scatter(x.data(), local_n, MPI_FLOAT,
        local_x.data(), local_n, MPI_FLOAT, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();
    for (int r = 0; r < REPS; r++) {
        MPI_Allgather(local_x.data(), local_n, MPI_FLOAT,
            global_x.data(), local_n, MPI_FLOAT, MPI_COMM_WORLD);

        for (int i = 0; i < local_m; i++) {
            float s = 0;
            for (int j = 0; j < n; j++)
                s += local_A[i * n + j] * global_x[j];
            local_y[i] = s;
        }
    }
    double t_local = (MPI_Wtime() - t1) / REPS;

    double t_max;
    MPI_Reduce(&t_local, &t_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    MPI_Gather(local_y.data(), local_m, MPI_FLOAT,
        y.data(), local_m, MPI_FLOAT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("The product is equal:\n");
        for (int i = 0; i < m; i++) {
            printf("%4.0f ", y[i]);
            if ((i + 1) % 20 == 0) printf("\n");
        }
        printf("\n");

        FILE* fr = fopen("result.txt", "w");
        for (int i = 0; i < m; i++) fprintf(fr, "%.0f ", y[i]);
        fclose(fr);

        int errors = 0;
        for (int i = 0; i < m; i++)
            if (fabsf(y[i] - y_ref[i]) > 1e-3f) errors++;
        printf("Check vs serial: %s (errors: %d)\n", errors == 0 ? "OK" : "FAIL", errors);

        printf("p = %d, execution time = %.3f microsecond(s)\n", p, t_max * 1e6);
    }

    MPI_Finalize();
    return 0;
}