#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <math.h>
#include <sys/time.h>
#include <mpi.h>


#include "../inc/argument_utils.h"

// Convert 'struct timeval' into seconds in double prec. floating point
#define WALLTIME(t) ((double)(t).tv_sec + 1e-6 * (double)(t).tv_usec)

typedef int64_t int_t;
typedef double real_t;

int_t
    N,
    M,
    max_iteration,
    snapshot_frequency;

real_t
    *temp[2] = { NULL, NULL },
    *thermal_diffusivity,
    dx,
    dt;

int rank, size;

#define T(i,j)                      temp[0][(i) * (M + 2) + (j)]
#define T_next(i,j)                 temp[1][((i) * (M + 2) + (j))]
#define THERMAL_DIFFUSIVITY(i,j)    thermal_diffusivity[(i) * (M + 2) + (j)]

void time_step ( void );
void boundary_condition( void );
void border_exchange( void );
void domain_init ( void );
void domain_save ( int_t iteration );
void domain_finalize ( void );


void
swap ( real_t** m1, real_t** m2 )
{
    real_t* tmp;
    tmp = *m1;
    *m1 = *m2;
    *m2 = tmp;
}


int
main ( int argc, char **argv )
{
    
    // TODO 1: Initialize MPI
    int_t arr[4];
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // TODO 2: Parse arguments in the rank 0 processes
    // and broadcast to other processes
    if (rank == 0){
    OPTIONS *options = parse_args( argc, argv );
    if ( !options )
    {
        fprintf( stderr, "Argument parsing failed\n" );
        exit(1);
    }

    N = options->N;
    M = options->M;
    max_iteration = options->max_iteration;
    snapshot_frequency = options->snapshot_frequency;
    
    arr[0] = N;
    arr[1] = M;
    arr[2] = max_iteration;
    arr[3] = snapshot_frequency;
    }

    // MPI_Bcast(&N, 1, MPI_INT64_T, 0, MPI_COMM_WORLD);
    // MPI_Bcast(&M, 1, MPI_INT64_T, 0, MPI_COMM_WORLD); 
    // MPI_Bcast(&max_iteration, 1, MPI_INT64_T, 0, MPI_COMM_WORLD); 
    // MPI_Bcast(&snapshot_frequency, 1, MPI_INT64_T, 0, MPI_COMM_WORLD); 
    MPI_Bcast(&arr, 4, MPI_INT64_T, 0, MPI_COMM_WORLD); 
    if (rank > 0){
        N = arr[0];
        M = arr[1];
        max_iteration = arr[2];
        snapshot_frequency = arr[3];
    }
    // else{
    //     MPI_Recv(&arr, 4, MPI_INT64_T, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    //     N = arr[0];
    //     M = arr[1];
    //     max_iteration = arr[2];
    //     snapshot_frequency = arr[3];
    // }
    

    // TODO 3: Allocate space for each process' sub-grids
    // and initialize data for the sub-grids
    domain_init();

    struct timeval t_start, t_end;
    gettimeofday ( &t_start, NULL );

    for ( int_t iteration = 0; iteration <= max_iteration; iteration++ )
    {
        // TODO 7: Communicate border values
        border_exchange();

        // TODO 5: Boundary conditions
        boundary_condition();

        // TODO 4: Time step calculations
        time_step();

        if ( iteration % snapshot_frequency == 0 )
        {
            printf (
                "Iteration %ld of %ld (%.2lf%% complete)\n",
                iteration,
                max_iteration,
                100.0 * (real_t) iteration / (real_t) max_iteration
            );

            // TODO 6 MPI I/O
            domain_save ( iteration );
        }

        swap( &temp[0], &temp[1] );
    }
    gettimeofday ( &t_end, NULL );
    printf ( "Total elapsed time: %lf seconds\n",
            WALLTIME(t_end) - WALLTIME(t_start)
            );


    domain_finalize();

    // TODO 1: Finalize MPI
    MPI_Finalize();
    exit ( EXIT_SUCCESS );
}


void
time_step ( void )
{
    // TODO 4: Time step calculations
    real_t c, t, b, l, r, K, new_value;
    for ( int_t x = 1; x <= N/size; x++ )
    {
        for ( int_t y = 1; y <= M; y++ )
        {
            c = T(x, y);

            t = T(x - 1, y);
            b = T(x + 1, y);
            l = T(x, y - 1);
            r = T(x, y + 1);
            K = THERMAL_DIFFUSIVITY(x, y);

            new_value = c + K * dt * ((l - 2 * c + r) + (b - 2 * c + t));

            T_next(x, y) = new_value;
        }
    }
}


void
boundary_condition ( void )
{
    // TODO 5: Boundary conditions
    int_t k = N/size;
    for ( int_t x = 1; x <= k; x++ )
    {
        T(x + rank*k, 0) = T(x, 2);
        T(x + rank*k, M+1) = T(x, M-1);
    }

    if (rank == 0){
        for (int_t y = 1; y <= M; y++){
            T(0, y) = T(2, y);
        }
    }
    if (rank == size-1){
        for ( int_t y = 1; y <= M; y++ ){
            T(k+1, y) = T(k-1, y);
        }
    }

}


void
border_exchange ( void )
{
    // TODO 7: Communicate border values
    int_t k = N/size;
    real_t second_last_M[M+2], second_M[M+2], recv_up[M+2], recv_down[M+2];
    for (int_t i = 0; i <= M+1; i++){
        second_M[i] = temp[0][M+2 + i];
        second_last_M[i] = temp[0][k*(M+2) + i];
    }
    if (rank == 0){
        MPI_Sendrecv(&second_last_M, M+2, MPI_DOUBLE, 1, 0, &recv_up, M+2, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // MPI_Send ( &second_last_M , M+2, MPI_DOUBLE , 1 , 0 , MPI_COMM_WORLD ); 
        // MPI_Recv ( &recv_up , M+2, MPI_DOUBLE , 1 , 0 , MPI_COMM_WORLD, MPI_STATUS_IGNORE ) ;
        for (int_t i = 0; i <= M+1; i++){
            temp[0][(k+1)*(M+2) + i] = recv_up[i];
        }
    }
    else if (1 <= rank && rank < size-1){
        MPI_Sendrecv(&second_M, M+2, MPI_DOUBLE, rank-1, 0, &recv_down, M+2, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(&second_last_M, M+2, MPI_DOUBLE, rank+1, 0, &recv_up, M+2, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (int_t i = 0; i <= M+1; i++){
            temp[0][(k+1)*(M+2) + i] = recv_up[i];
            temp[0][i] = recv_down[i];
        }
    }
    else{
        MPI_Sendrecv(&second_M, M+2, MPI_DOUBLE, rank-1, 0, &recv_down, M+2, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // MPI_Send ( &second_last_M , M+2, MPI_DOUBLE , 0 , 0 , MPI_COMM_WORLD ); 
        // MPI_Recv ( &recv_up , M+2, MPI_DOUBLE , 0 , 0 , MPI_COMM_WORLD, MPI_STATUS_IGNORE ) ;
        for (int_t i = 0; i <= M+1; i++){
            temp[0][i] = recv_down[i];
        }
    }
}


void
domain_init ( void )
{
    // TODO 3: Allocate space for each process' sub-grids
    // and initialize data for the sub-grids
    real_t
        temperature,
        diffusivity;

    // Create subgrid by row
    int_t k = N/size;
    temp[0] = malloc ( (k+2)*(M+2) * sizeof(real_t) );
    temp[1] = malloc ( (k+2)*(M+2) * sizeof(real_t) );
    thermal_diffusivity = malloc ( (k+2)*(M+2) * sizeof(real_t) );

    dt = 0.1;
    dx = 0.1;

    for ( int_t x = 1; x <= k; x++ )
    {
        for ( int_t y = 1; y <= M; y++ )
        {
            temperature = 30 + 30 * sin(((x + rank*k) + y) / 20.0);
            diffusivity = 0.05 + (30 + 30 * sin((k*size - (x + rank*k) + y) / 20.0)) / 605.0; // k*size = N
            T(x,y) = temperature;
            T_next(x,y) = temperature;

            THERMAL_DIFFUSIVITY(x,y) = diffusivity;
        }
    }
}


void
domain_save ( int_t iteration )
{
    // int_t k=N/size;
    // int_t index = iteration / snapshot_frequency;
    // char filename[256];
    // memset ( filename, 0, 256*sizeof(char) );
    // sprintf ( filename, "data/%.5ld.bin", index );
    
    // MPI_File out;
    // MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_CREATE | MPI_MODE_RDWR, MPI_INFO_NULL, &out);
    // MPI_Offset offset = rank*(k+1)*(M+2)*sizeof(real_t);
    // MPI_File_write_at_all(out, offset, &temp[0], (k+2)*(M+2), MPI_DOUBLE, MPI_STATUS_IGNORE);

    // MPI_File_close(&out); 

    // FILE *out = fopen ( filename, "wb" );
    // if ( ! out ) {
    //     fprintf(stderr, "Failed to open file: %s\n", filename);
    //     exit(1);
    // }
    // fwrite( temp[0], sizeof(real_t), (N+2)*(M+2), out );
    // fclose ( out );
}


void
domain_finalize ( void )
{
    free ( temp[0] );
    free ( temp[1] );
    free ( thermal_diffusivity );
}
