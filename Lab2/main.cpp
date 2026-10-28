
#include <stdio.h>
#include <math.h>
#include "mpi.h"

double f(double x) { return tan(x); }                
double exact(double a, double b) { return -log(cos(b)) + log(cos(a)); }

double Trap(double local_a, double local_b, int local_n, double h) {
    double integral = (f(local_a) + f(local_b)) / 2.0;
    double x = local_a;
    int i;
    for (i = 1; i <= local_n - 1; i++) {
        x = x + h;
        integral += f(x);
    }
    return integral * h;
}

int main(int argc, char** argv) {
    int my_rank, p, source, dest = 0, tag = 0, k;
    double a = 0.0, b = 1.0;                          
    int n_list[5] = { 10, 20, 30, 40, 50 };
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    if (my_rank == 0) {
        printf("Function: tg(x), [%.1f; %.1f], processes: %d\n", a, b, p);
        printf("Exact value = %.10f\n\n", exact(a, b));
        printf("%6s %16s %16s %14s\n", "n", "integral", "rel.error,%", "time,ms");
    }

    for (k = 0; k < 5; k++) {
        int n = n_list[k];
        double h, local_a, local_b, integral, total, t_start = 0, t;
        int local_n;

        
        if (my_rank == 0) {
            for (dest = 1; dest < p; dest++)
                MPI_Send(&n, 1, MPI_INT, dest, tag, MPI_COMM_WORLD);
        }
        else {
            MPI_Recv(&n, 1, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        if (my_rank == 0) t_start = MPI_Wtime();

        h = (b - a) / n;
        local_n = n / p;
        local_a = a + my_rank * local_n * h;
        if (my_rank == p - 1) local_n += n % p;      
        local_b = local_a + local_n * h;

        integral = Trap(local_a, local_b, local_n, h);

        if (my_rank == 0) {
            total = integral;
            for (source = 1; source < p; source++) {
                MPI_Recv(&integral, 1, MPI_DOUBLE, source, tag, MPI_COMM_WORLD, &status);
                total += integral;
            }
            t = MPI_Wtime() - t_start;
            {
                double ex = exact(a, b);
                printf("%6d %16.10f %16.8f %14.6f\n", n, total,
                    fabs(ex - total) / ex * 100.0, t * 1000.0);
            }
        }
        else {
            MPI_Send(&integral, 1, MPI_DOUBLE, 0, tag, MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();
    return 0;
}