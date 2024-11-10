#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <stdbool.h>
#include <math.h>
#include <sys/time.h>
#include <mpi.h>

#include "../inc/argument_utils.h"

// Convert 'struct timeval' into seconds in double prec. floating point
#define WALLTIME(t) ((double)(t).tv_sec + 1e-6 * (double)(t).tv_usec)

typedef int64_t int_t;
typedef double real_t;

int_t
    M,
    N,
    max_iteration,
    snapshot_frequency,
    num_rows, 
    num_colums;

real_t
    *temp[2] = { NULL, NULL },
    *thermal_diffusivity,
    dt;

int rank, size;
int dims[2] = {0, 0};
int my_coords[2];

MPI_Comm COORD_COMMUNICATOR;

#define T(x,y)                      temp[0][(y) * (num_colums + 2) + (x)]
#define T_next(x,y)                 temp[1][((y) * (num_colums + 2) + (x))]
#define THERMAL_DIFFUSIVITY(x,y)    thermal_diffusivity[(y) * (num_colums + 2) + (x)]

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
    // TODO 1:
    // - Initialize and finalize MPI.
    // - Create a cartesian communicator.
    // - Parse arguments in the rank 0 processes
    //   and broadcast to other processes

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Total freedom of decomposing processes in a 2D grid
    MPI_Dims_create(size,2,dims);

    // X,Y-dimensions do not need to be periodic
    int periods[2] = {false, false};

    // Without reordering assigments of ranks
    int reorder = false;

    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &COORD_COMMUNICATOR);

    // Get new ranks <- same as in MPI_COMM_WORLD
    MPI_Comm_rank(COORD_COMMUNICATOR, &rank);

    // Get rank position
    MPI_Cart_coords(COORD_COMMUNICATOR, rank, 2, my_coords);
 
    // Print my location in the 2D torus.
    int neighbours_ranks[4];

    // X - direction
    MPI_Cart_shift(COORD_COMMUNICATOR, 0, 1, &neighbours_ranks[0], &neighbours_ranks[1]);

    // Y - direction
    MPI_Cart_shift(COORD_COMMUNICATOR, 1, 1, &neighbours_ranks[2], &neighbours_ranks[3]);

    printf("[MPI process %d] I am located at (%d, %d)... left: %d, right: %d, down: %d, up: %d.\n", rank, my_coords[0],my_coords[1], neighbours_ranks[0], neighbours_ranks[1], neighbours_ranks[2], neighbours_ranks[3]);


    if (rank == 0){

        OPTIONS *options = parse_args( argc, argv );
        if ( !options )
        {
            fprintf( stderr, "Argument parsing failed\n" );
            exit(1);
        }

        M = options->M;
        N = options->N;
        max_iteration = options->max_iteration;
        snapshot_frequency = options->snapshot_frequency;
    }

    // Broadcast values
    MPI_Bcast( &N, 1, MPI_INT64_T, 0, MPI_COMM_WORLD );
    MPI_Bcast( &M, 1, MPI_INT64_T, 0, MPI_COMM_WORLD );
    MPI_Bcast( &max_iteration, 1, MPI_INT64_T, 0, MPI_COMM_WORLD );
    MPI_Bcast( &snapshot_frequency, 1, MPI_INT64_T, 0, MPI_COMM_WORLD );


    // calculate # of rows and columns in each subgrid
    num_rows = M/dims[1];
    num_colums = N/dims[0];

    domain_init();

    struct timeval t_start, t_end;
    gettimeofday ( &t_start, NULL );

    for ( int_t iteration = 0; iteration <= max_iteration; iteration++ )
    {
        // TODO 6: Implement border exchange.
        // Hint: Creating MPI datatypes for rows and columns might be useful.
        border_exchange();

        boundary_condition();

        time_step();

        if ( iteration % snapshot_frequency == 0 )
        {
            printf (
                "Iteration %ld of %ld (%.2lf%% complete)\n",
                iteration,
                max_iteration,
                100.0 * (real_t) iteration / (real_t) max_iteration
            );

            domain_save ( iteration );
        }

        swap( &temp[0], &temp[1] );
    }

    gettimeofday ( &t_end, NULL );
    printf ( "Total elapsed time: %lf seconds\n",
            WALLTIME(t_end) - WALLTIME(t_start)
            );


    domain_finalize();

    MPI_Finalize();
    exit ( EXIT_SUCCESS );
}


void border_exchange(void){
    // TODO 6: Implement border exchange.
    int neighbours_ranks[4];
    
    real_t push_vec_column[2][num_rows], push_vec_row[2][num_colums], pull_vec_column[2][num_rows], pull_vec_row[2][num_colums];

    // Take first and last row in the interior of (nom_columns+2)x(nom_rows+2) grid
    for (int_t i = 1; i <= num_colums; i++){
        push_vec_row[0][i-1] = T(i, 1);
        push_vec_row[1][i-1] = T(i, num_rows);
    }

    // Take first and last column in the interior of (nom_columns+2)x(nom_rows+2) grid
    for (int_t i = 1; i <= num_rows; i++){
        push_vec_column[0][i-1] = T(1, i);
        push_vec_column[1][i-1] = T(-2,i+1);
    }

     // X - direction
     MPI_Cart_shift(COORD_COMMUNICATOR, 0, 1, &neighbours_ranks[0], &neighbours_ranks[1]);
     
     // Y - direction
     MPI_Cart_shift(COORD_COMMUNICATOR, 1, 1, &neighbours_ranks[2], &neighbours_ranks[3]);

    // 0 - left neighbor, 1 - right neighbor, 2 - neighbor below, 3 - neighbor above
    if (neighbours_ranks[0] != -2){
        MPI_Sendrecv(&push_vec_column[0], num_rows, MPI_DOUBLE, neighbours_ranks[0], 0, &pull_vec_column[0], num_rows, MPI_DOUBLE, neighbours_ranks[0], 0, COORD_COMMUNICATOR, MPI_STATUS_IGNORE);
        for(int_t i = 1; i <= num_rows; i++){
            T(0,i) = pull_vec_column[0][i-1]; // leftmost column in (nom_columns+2)x(nom_rows+2) grid

        }
    }
    if (neighbours_ranks[1] != -2){
        MPI_Sendrecv(&push_vec_column[1], num_rows, MPI_DOUBLE, neighbours_ranks[1], 0, &pull_vec_column[1], num_rows, MPI_DOUBLE, neighbours_ranks[1], 0, COORD_COMMUNICATOR, MPI_STATUS_IGNORE);
        for(int_t i = 1; i <= num_rows; i++){
           T(-1,i+1) = pull_vec_column[1][i-1]; // rightmost column in (nom_columns+2)x(nom_rows+2) grid
        }
        
    }
    if (neighbours_ranks[2] != -2){
        MPI_Sendrecv(&push_vec_row[0], num_colums, MPI_DOUBLE, neighbours_ranks[2], 0, &pull_vec_row[0], num_colums, MPI_DOUBLE, neighbours_ranks[2], 0, COORD_COMMUNICATOR, MPI_STATUS_IGNORE);
        for(int_t i = 1; i <= num_colums; i++){
            T(i,0) = pull_vec_row[0][i-1]; // bottommost row in (nom_columns+2)x(nom_rows+2) grid
        }

    }
    if (neighbours_ranks[3] != -2){
        MPI_Sendrecv(&push_vec_row[1], num_colums, MPI_DOUBLE, neighbours_ranks[3], 0, &pull_vec_row[1], num_colums, MPI_DOUBLE, neighbours_ranks[3], 0, COORD_COMMUNICATOR, MPI_STATUS_IGNORE);
        for(int_t i = 1; i <= num_colums; i++){
            T(i,num_rows+1) = pull_vec_row[1][i-1]; // uppermost row in (nom_columns+2)x(nom_rows+2) grid
        }

    }
}


void
time_step ( void )
{
    real_t c, t, b, l, r, K, new_value;

    // TODO 3: Update the area of iteration so that each
    // process only iterates over its own subgrid.

    for ( int_t y = 1; y <= num_rows; y++ )
    {
        for ( int_t x = 1; x <= num_colums; x++ )
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
    // TODO 4: Change the application of boundary conditions
    // to match the cartesian topology.
    int my_coords[2];
    MPI_Cart_coords(COORD_COMMUNICATOR, rank, 2, my_coords);

    // lower boundary
    if (my_coords[1] == 0){
        for (int_t x = 1; x <= num_colums; x++){
            T(x, 0) = T(x, 2);
        }
    }

    // left boundary
    if (my_coords[0] == 0){
        for (int_t y = 1; y <= num_rows; y++){
            T(0, y) = T(2, y);
        }
    }

    // upper boundary
    if (my_coords[1] == dims[1]-1){
        for (int_t x = 1; x <= num_colums; x++){
            T(x, num_rows + 1) = T(x, num_rows - 1);
        }
    }

    // right boundary
    if (my_coords[0] == dims[0]-1){
        for (int_t y = 1; y <= num_rows; y++){
            T(num_colums + 1, y) = T(num_colums - 1, y);
        }
    }
}


void
domain_init ( void )
{
    // TODO 2:
    // - Find the number of columns and rows in each process' subgrid.
    // - Allocate memory for each process' subgrid.
    // - Find each process' offset to calculate the correct initial values.
    // Hint: you can get useful information from the cartesian communicator.
    // Note: you are allowed to assume that the grid size is divisible by
    // the number of processes.

    temp[0] = malloc ( (num_rows+2)*(num_colums+2) * sizeof(real_t) );
    temp[1] = malloc ( (num_rows+2)*(num_colums+2) * sizeof(real_t) );
    thermal_diffusivity = malloc ( (num_rows+2)*(num_colums+2) * sizeof(real_t) );

    dt = 0.1;

    for ( int_t y = 1; y <= num_rows; y++ )
    {
        for ( int_t x = 1; x <= num_colums; x++ )
        {
            real_t temperature = 30 + 30 * sin(((x + my_coords[0]*num_colums) + (y + my_coords[1]*num_rows)) / 20.0);
            real_t diffusivity = 0.05 + (30 + 30 * sin((N - (x + my_coords[0]*num_colums) + (y + my_coords[1]*num_rows)) / 20.0)) / 605.0;

            T(x,y) = temperature;
            T_next(x,y) = temperature;
            THERMAL_DIFFUSIVITY(x,y) = diffusivity;
        }
    }
}


void
domain_save ( int_t iteration )
{
    // TODO 5: Use MPI I/O to save the state of the domain to file.
    // Hint: Creating MPI datatypes might be useful.

    int_t index = iteration / snapshot_frequency;
    char filename[256];
    memset ( filename, 0, 256*sizeof(char) );
    sprintf ( filename, "data/%.5ld.bin", index );

    // calculating the starting position of each subarray in the whole 2D grid
    int position[2] = {num_colums*my_coords[0], num_rows*my_coords[1]};

    MPI_Datatype sub_grid_type, grid_type;
    // Take the interior of (num_columns+2)x(num_rows+2) grid
    MPI_Type_create_subarray(2, (int[2]){num_colums+2,num_rows+2}, (int[2]){num_colums,num_rows}, (int[2]){1,1},  MPI_ORDER_FORTRAN, MPI_DOUBLE, &sub_grid_type);
    MPI_Type_commit(&sub_grid_type);

    // place (num_columns)x(num_rows) subgrid to the proper position in NxM grid
    MPI_Type_create_subarray(2, (int[2]){N,M}, (int[2]){num_colums,num_rows}, (int[2]){position[0],position[1]},  MPI_ORDER_FORTRAN, MPI_DOUBLE, &grid_type);
    MPI_Type_commit(&grid_type);

    // Write to a file
    MPI_File out;
    MPI_File_open ( COORD_COMMUNICATOR, filename, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL , &out );
    MPI_File_set_view(out, 0, MPI_DOUBLE, grid_type, "native", MPI_INFO_NULL);
    MPI_File_write_all(out, temp[0], 1, sub_grid_type, MPI_STATUS_IGNORE);

    MPI_File_close(&out);

    MPI_Type_free(&sub_grid_type);
    MPI_Type_free(&grid_type);
}


void
domain_finalize ( void )
{
    free ( temp[0] );
    free ( temp[1] );
    free ( thermal_diffusivity );
}
