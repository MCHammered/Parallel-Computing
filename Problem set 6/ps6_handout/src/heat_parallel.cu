#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <math.h>
#include <sys/time.h>
#include <cooperative_groups.h>

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
    size;

real_t
    dt,
    *temp,
    *thermal_diffusivity,
    // Declare device side pointers to store host-side data.
    *d_temp,
    *d_thermal_diffusivity;

namespace cg = cooperative_groups;

#define T(x,y)                      temp[(y) * (N + 2) + (x)]
#define THERMAL_DIFFUSIVITY(x,y)    thermal_diffusivity[(y) * (N + 2) + (x)]

#define cudaErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__global__ void time_step (const int_t N, const int_t M, real_t *d_temp, const real_t *d_thermal_diffusivity, const real_t dt);
__device__ void boundary_condition (const int x, const int y, const int_t N, const int_t M, real_t *d_temp);
void domain_init ( void );
void domain_save ( int_t iteration );
void domain_finalize ( void );


int
main ( int argc, char **argv )
{
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

    domain_init();

    struct timeval t_start, t_end;
    gettimeofday ( &t_start, NULL );

    // Initialise parameters needed for cudaLaunchCooperativeKernel call
    void *args[] = {(void*) &N, (void*) &M, (void*) &d_temp, (void*) &d_thermal_diffusivity, (void*) &dt};
    dim3 threadBlockDims(16,16);
    dim3 gridDims((N+2)/16+1, (M+2)/16+1);


    for ( int_t iteration = 0; iteration <= max_iteration; iteration++ )
    {
        // Launch the cooperative time_step-kernel.
        cudaLaunchCooperativeKernel((void*) time_step, gridDims, threadBlockDims, args);


        if ( iteration % snapshot_frequency == 0 )
        {
            printf (
                "Iteration %ld of %ld (%.2lf%% complete)\n",
                iteration,
                max_iteration,
                100.0 * (real_t) iteration / (real_t) max_iteration
            );

            // Copy data from device to host.
            cudaMemcpy(temp, d_temp, size, cudaMemcpyDeviceToHost);
            domain_save ( iteration );
        }
    }

    gettimeofday ( &t_end, NULL );
    printf ( "Total elapsed time: %lf seconds\n",
            WALLTIME(t_end) - WALLTIME(t_start)
            );


    domain_finalize();

    exit ( EXIT_SUCCESS );
}


// Make time_step() a cooperative CUDA kernel
// where one thread is responsible for one grid point.
__global__ void time_step(const int_t N, const int_t M, real_t *d_temp, const real_t *d_thermal_diffusivity, const real_t dt)
{
    cg::grid_group grid = cg::this_grid();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    int row = idy % 2;

    boundary_condition(idx, idy, N, M, d_temp);

    grid.sync();

    // compute first color
    if ((1 <= idx && idx <= N) && (1 <= idy && idy <= M) && (idx + N*idy + row) % 2 == 1){
        real_t c, t, b, l, r, K, A, D, new_value;
        //(y) * (N + 2) + (x)
        c = d_temp[idy * (N + 2) + idx];

        t = d_temp[idy * (N + 2) + idx - 1];
        b = d_temp[idy * (N + 2) + idx + 1];
        l = d_temp[(idy-1) * (N + 2) + idx];
        r = d_temp[(idy+1) * (N + 2) + idx];
        K = d_thermal_diffusivity[idy * (N + 2) + idx];

        A = - K * dt;
        D = 1.0f + 4.0f * K * dt;

        new_value = (c - A * (t + b + l + r)) / D;

        d_temp[idy * (N + 2) + idx] = new_value;   
    }

    grid.sync();

    // compute second color
    if ((1 <= idx && idx <= N) && (1 <= idy && idy <= M) && (idx + N*idy + row) % 2 == 0){
        real_t c, t, b, l, r, K, A, D, new_value;
        //(y) * (N + 2) + (x)
        c = d_temp[idy * (N + 2) + idx];

        t = d_temp[idy * (N + 2) + idx - 1];
        b = d_temp[idy * (N + 2) + idx + 1];
        l = d_temp[(idy-1) * (N + 2) + idx];
        r = d_temp[(idy+1) * (N + 2) + idx];
        K = d_thermal_diffusivity[idy * (N + 2) + idx];

        A = - K * dt;
        D = 1.0f + 4.0f * K * dt;

        new_value = (c - A * (t + b + l + r)) / D;

        d_temp[idy * (N + 2) + idx] = new_value;    
    }


}

// Make boundary_condition() a device function and
// call it from the time_step-kernel.
__device__ void boundary_condition (const int x, const int y, const int_t N, const int_t M, real_t *d_temp)
{
    // set boundary columns, i.e (0,y) and (N+1, y), y ∈ [1, 2, ..., M]
    if(x == 0 && (1 <= y && y <= M)){d_temp[y * (N + 2)] = d_temp[y * (N + 2) + 2];}
    else if(x == N+1 && (1 <= y && y <= M)){d_temp[y * (N + 2) + N+1] = d_temp[y * (N + 2) + N-1];}
    // set boundary rows, i.e (x,0) and (x, M+1), x ∈ [1, 2, ..., N]
    else if(y == 0 && (1 <= x && x <= N)){d_temp[x] = d_temp[2 * (N + 2) + x];}
    else if(y == M+1 && (1 <= x && x <= N)){d_temp[(M+1) * (N + 2) + x] = d_temp[(M-1) * (N + 2) + x];}
}

void
domain_init ( void )
{
    size = (M+2)*(N+2) * sizeof(real_t);
    temp = (real_t*) malloc (size);
    thermal_diffusivity = (real_t*) malloc (size);

    // Allocate device memory.
    cudaMalloc (&d_temp, size);
    cudaMalloc (&d_thermal_diffusivity, size);

    dt = 0.1;

    for ( int_t y = 1; y <= M; y++ )
    {
        for ( int_t x = 1; x <= N; x++ )
        {
            real_t temperature = 30 + 30 * sin((x + y) / 20.0);
            real_t diffusivity = 0.05 + (30 + 30 * sin((N - x + y) / 20.0)) / 605.0;

            T(x,y) = temperature;
            THERMAL_DIFFUSIVITY(x,y) = diffusivity;
        }
    }

    // Copy data from host to device.
    cudaMemcpy (d_temp, temp, size, cudaMemcpyHostToDevice);
    cudaMemcpy (d_thermal_diffusivity, thermal_diffusivity, size, cudaMemcpyHostToDevice);
}


void
domain_save ( int_t iteration )
{
    int_t index = iteration / snapshot_frequency;
    char filename[256];
    memset ( filename, 0, 256*sizeof(char) );
    sprintf ( filename, "data/%.5ld.bin", index );

    FILE *out = fopen ( filename, "wb" );
    if ( ! out )
    {
        fprintf(stderr, "Failed to open file: %s\n", filename);
        exit(1);
    }
    fwrite( temp, sizeof(real_t), (N + 2) * (M + 2), out );
    fclose ( out );
}


void
domain_finalize ( void )
{
    free ( temp );
    free ( thermal_diffusivity );

    // Free device memory.
    cudaFree (d_temp);
    cudaFree (d_thermal_diffusivity);
}
