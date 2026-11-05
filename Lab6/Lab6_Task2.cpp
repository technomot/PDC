#define _CRT_SECURE_NO_WARNINGS

#include "mpi.h"
#include <stdio.h>
#include <string.h>

int main(int argc, char* argv[])
{
    int numprocs;
    char message[32];
    int myrank;
    MPI_Status status;
    int TAG = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

    if (numprocs < 2)
    {
        if (myrank == 0)
            printf("Run the program with at least 2 processes\n");
        MPI_Finalize();
        return 1;
    }

    if (myrank == 0)
    {
        strcpy(message, "Hi, Second Processor!");
        MPI_Send(message, (int)strlen(message) + 1, MPI_CHAR, 1, TAG, MPI_COMM_WORLD);
    }
    else if (myrank == 1)
    {
        MPI_Recv(message, 32, MPI_CHAR, 0, TAG, MPI_COMM_WORLD, &status);
        printf("received %s\n", message);
    }

    MPI_Finalize();
    return 0;
}