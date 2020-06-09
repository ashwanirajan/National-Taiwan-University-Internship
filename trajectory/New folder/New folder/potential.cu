/////////////////////////
/////////////////////////

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <cufft.h>
#include <cutil_inline.h>
#include <complex.h>
#include <time.h>
#include <omp.h>

// CUDA error check 
#define CUDA_CHECK_ERROR( Call )   CUDA_Check_Error( Call, __FILE__, __LINE__, __FUNCTION__ )

inline void CUDA_Check_Error( cudaError Return, const char *File, const int Line, const char *Func )
{
   if ( Return != cudaSuccess )
      fprintf( stderr, "CUDA ERROR : %s !!\n", cudaGetErrorString( Return ) );
}

#define ERROR_INFO          File, Line, Func
#define CUFFT_CHECK_ERROR( Call )   CUFFT_Check_Error( Call, __FILE__, __LINE__, __FUNCTION__ )


#ifdef FLOAT8
#include <drfftw.h>
#else
#include <srfftw.h>
#endif

#ifdef FLOAT8
 typedef double             real;
 typedef cufftDoubleReal    CUFFT_Real;
 typedef cufftDoubleComplex CUFFT_Complex;
#else
 typedef float           real;
 typedef cufftReal       CUFFT_Real;
 typedef cufftComplex    CUFFT_Complex;
#endif

#define N 512
// should be the multiple of 16, due to Potential Evolution.

#define Nc (N/2+1)
#define N_thread  (int)N/2
#define N_block  (int)N/2
#define Batch     (int)1

#define N_dump_3D 10
#define N_revol 10000

#define NT 6
/////////////////////
#define N_S 10

/////////////////////
#define d_mass 1e-4
#define M_BH_Final 1.
#define N_BH_start 50000



#define x_bh 227
#define y_bh 276
#define z_bh 240
//////////////////////
#define N_part 1
// Mass of soliton :2.41520982994032671e+01 
//#define PartMassMax (0.1*2.41520982994032671e+01/N_part)
//#define d_PartMass (PartMassMax/100.)

/////////////////////

#define PeriodicBondary
#define Part_Evo


//////////////////////
int i,j,k;
int ll,mm;
int n_i;

FILE* file;

//const double d_time0 = 0.5*M_1_PI; 

int Nr;



unsigned int Timer;
  real total_time       = (real)0.0;
  real average_time     = (real)0.0;
  real operation_time   = (real)0.0;


/*
__global__ void Force_x( CUFFT_Real phi[][N][N], CUFFT_Real force[N][N], int i_layer);
__global__ void Force_y( CUFFT_Real phi[][N][N], CUFFT_Real force[N][N], int i_layer);
__global__ void Force_z( CUFFT_Real phi[][N][N], CUFFT_Real force[N][N], int i_layer);
void Part_Pos(double Pos_t[][3], double Vel_t[][3] ,CUFFT_Real force_h_x[][N][N], CUFFT_Real force_h_y[][N][N], CUFFT_Real force_h_z[][N][N],
                  double *d_time);
*/
void Ext_potential(CUFFT_Real delta_p[][N][N], double *d_time_h, double t_rot);



int main()
{

CUFFT_Real    (*delta_p)[N][N]   = new CUFFT_Real    [N][N][N];
/*
CUFFT_Real    (*force_h_x)[N][N  ]   = new CUFFT_Real    [N][N][N  ];
CUFFT_Real    (*force_h_y)[N][N  ]   = new CUFFT_Real    [N][N][N  ];
CUFFT_Real    (*force_h_z)[N][N  ]   = new CUFFT_Real    [N][N][N  ];

CUFFT_Real    (*delta_2d)  [N];
CUFFT_Real    (*device_delta_p) [N][N];

double *mass_t;
double *mass_t0;

double *M_BH;
*/
double *d_time;
double *d_time_h = new double [1];
double t_rot;
double *PotenMax;

size_t size_real   =N*N*(N)    *sizeof(CUFFT_Real   );
size_t size2d      =  N*(N)    *sizeof(real         );
size_t size_res    =50*N*(N)   *sizeof(CUFFT_Real   );
size_t size2d_r    =  N*N        *sizeof(CUFFT_Real   );
/*
double (*Pos_t  )[3]  = new double [N_part][3]        ;
double (*Vel_t  )[3]  = new double [N_part][3]        ;


float *Pos_tmp = new float [3];
float *Vel_tmp = new float [3];
*/
int i_layer;

FILE* file_output;
char filename[200];

/*

CUDA_CHECK_ERROR( cudaMalloc((void**)&device_delta_p  ,size_real          ));
CUDA_CHECK_ERROR( cudaMalloc((void**)&delta_2d ,size2d_r           ));

CUDA_CHECK_ERROR( cudaMalloc((void**)&mass_t       ,sizeof(double)    ));
CUDA_CHECK_ERROR( cudaMalloc((void**)&mass_t0      ,sizeof(double)    ));
CUDA_CHECK_ERROR( cudaMalloc((void**)&d_time       ,sizeof(double)    ));
CUDA_CHECK_ERROR( cudaMalloc((void**)&PotenMax     ,sizeof(double)    ));


CUDA_CHECK_ERROR( cudaMalloc((void**)&M_BH         ,sizeof(double)    ));
*/

/////////////////////Initialising Data////////////
/*
Pos_t[0][0] = (double)N/2 ;
Pos_t[0][1] = (double)N/2 ;
Pos_t[0][2] = (double)N/2 ;

Vel_t[0][0] = 0. ;
Vel_t[0][1] = 0. ;
Vel_t[0][2] = 0. ;


*/


Nr = 1;
//d_time_h[0] = 2.*M_1_PI*(N/512.)*(N/512.)*0.5;


d_time_h[0] = 16.*M_1_PI*(N/512.)*(N/512.)*0.5;

t_rot = 0.;

do{

sprintf(filename, "potential_t_%d",Nr);
file_output = fopen(filename,"wb");


//////////////////////////////////////////////////////////// Added External potential \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\


Ext_potential( delta_p, d_time_h, t_rot);
//CUDA_CHECK_ERROR( cudaMemcpy(device_delta_p,delta_p,size_real,cudaMemcpyHostToDevice));
//printf(" %24.17e %24.17e %24.17e\n", delta_p[0][0][0], delta_p[0][0][1], delta_p[0][0][2]);

t_rot += d_time_h[0];

///////////////////////////////////////////////////////// Added External potential \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

//////////////////////////////////////////////////////// write ext potential\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\




for ( j = 0 ; i < N ; i++ )
	for ( i = 0 ; j< N ; j++  )
	{
  fwrite(&delta_p[i][j][256],sizeof(double),1,file_output);
    }


//////////////////////////////////////////////////////// write ext potential\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

/*
////////// Particle Evolution ///////////

/////ã~@~@Force //////
for ( i_layer = 0; i_layer < N ; i_layer++ )
{
Force_x<<<N_block,N_thread>>>( device_delta_p, delta_2d, i_layer);
CUDA_CHECK_ERROR( cudaThreadSynchronize() );
CUDA_CHECK_ERROR( cudaMemcpy(&force_h_x[i_layer][0][0],delta_2d,size2d_r,cudaMemcpyDeviceToHost));
}
for ( i_layer = 0; i_layer < N ; i_layer++ )
{
Force_y<<<N_block,N_thread>>>( device_delta_p, delta_2d, i_layer);
CUDA_CHECK_ERROR( cudaThreadSynchronize() );
CUDA_CHECK_ERROR( cudaMemcpy(&force_h_y[i_layer][0][0],delta_2d,size2d_r,cudaMemcpyDeviceToHost));
}
for ( i_layer = 0; i_layer < N ; i_layer++ )
{
Force_z<<<N_block,N_thread>>>( device_delta_p, delta_2d, i_layer);
CUDA_CHECK_ERROR( cudaThreadSynchronize() );
CUDA_CHECK_ERROR( cudaMemcpy(&force_h_z[i_layer][0][0],delta_2d,size2d_r,cudaMemcpyDeviceToHost));
}

Part_Pos(Pos_t, Vel_t, force_h_x, force_h_y, force_h_z, d_time_h);
printf("position %lf %lf %lf\n", Pos_t[0][0], Pos_t[0][1], Pos_t[0][2]);
//printf("velocity %24.17e %lf %lf\n", Vel_t[0][0], Vel_t[0][1], Vel_t[0][2]);
//printf("force  %24.17e %24.17e %24.17e\n", force_h_x[256][256][256], force_h_y[257][256][256], force_h_z[255][256][256]);
////////// Particle Evolution ////////////////

*/



//if( Nr %N_dump_3D ==  0 )
//{

#ifdef Part_Evo
  //  fwrite(Pos_t,sizeof(double),N_part*3,file_output);
//  fwrite(Vel_t,sizeof(double),N_part*3,file_output);
//  for ( ID_part = 0 ; ID_part < N_part ; ID_part++ ){
//  fwrite(&Pos_t[ID_part][0],sizeof(real),3,file_output);
//  fwrite(&Vel_t[ID_part][0],sizeof(real),3,file_output);
//  }

 //                                                                        fwrite(Pos_t,sizeof(double),3,file_output);

  //for ( ID_part = 0 ; ID_part < N_part ; ID_part++ ){
  //Vel_tmp[0] = Vel_t[ID_part][0];
  //Vel_tmp[1] = Vel_t[ID_part][1];
  //Vel_tmp[2] = Vel_t[ID_part][2];
  //fwrite(Vel_tmp,sizeof(float),3,file_output);
  //}
  //
#endif
//}

fclose(file_output);
Nr++;


} while ( Nr <= 130 );




return 0;

}

/*

__global__ void Force_x( CUFFT_Real phi[][N][N], CUFFT_Real force[N][N], int i_layer)
{
int ID;
int block = blockIdx.x;
int thread = threadIdx.x;

int total =  gridDim.x * blockDim.x;

int j,k;

  for (ID = block * blockDim.x + thread ; ID < N*N ; ID += total)
  {

   j = ID/N;
   k = ID%N;

   if      ( i_layer > 0 && i_layer < N-1 ) force[j][k] = -(phi[i_layer+1][j][k]-phi[i_layer-1][j][k])*0.5;
   else if ( i_layer == 0                 ) force[j][k] = -(phi[        1][j][k]-phi[      N-1][j][k])*0.5;
   else if ( i_layer == N-1               ) force[j][k] = -(phi[        0][j][k]-phi[      N-2][j][k])*0.5;
  }
}

__global__ void Force_y( CUFFT_Real phi[][N][N], CUFFT_Real force[N][N], int i_layer)
{
int ID;
int block = blockIdx.x;
int thread = threadIdx.x;

int total =  gridDim.x * blockDim.x;

int j,k;

  for (ID = block * blockDim.x + thread ; ID < N*N ; ID += total)
  {

   j = ID/N;
   k = ID%N;

   if      ( j != 0 && j != N-1 ) force[j][k] = -(phi[i_layer][j+1][k]-phi[i_layer][j-1][k])*0.5;
   else if ( j == 0             ) force[j][k] = -(phi[i_layer][  1][k]-phi[i_layer][N-1][k])*0.5;
   else if ( j == N-1           ) force[j][k] = -(phi[i_layer][  0][k]-phi[i_layer][N-2][k])*0.5;

  }
  }

__global__ void Force_z( CUFFT_Real phi[][N][N], CUFFT_Real force[N][N], int i_layer)
{
int ID;
int block = blockIdx.x;
int thread = threadIdx.x;

int total =  gridDim.x * blockDim.x;

int j,k;

  for (ID = block * blockDim.x + thread ; ID < N*N ; ID += total)
  {

   j = ID/N;
   k = ID%N;

   if      ( k != 0 && k !=N-1 ) force[j][k] = -(phi[i_layer][j][k+1]-phi[i_layer][j][k-1])*0.5;
   else if ( k == 0            ) force[j][k] = -(phi[i_layer][j][  1]-phi[i_layer][j][N-1])*0.5;
   else if ( k == N-1          ) force[j][k] = -(phi[i_layer][j][  0]-phi[i_layer][j][N-2])*0.5;

  }

}

void Part_Pos(double Pos_t[][3], double Vel_t[][3] ,CUFFT_Real force_h_x[][N][N], CUFFT_Real force_h_y[][N][N], CUFFT_Real force_h_z[][N][N],
                  double *d_time)
{
double Fx,Fy,Fz;
int i_pre,i_aft,j_pre,j_aft,k_pre,k_aft;
double d_i,d_j,d_k;
double s000,s100,s010,s001,s110,s011,s101,s111;
double dt_vel,dt_pos;
double a_max=0.,v_max=0.;


    dt_vel=d_time[0];
    dt_pos=d_time[0];

    //omp_set_num_threads(NT);
//#  pragma omp parallel for private(i_pre, i_aft, d_i, j_pre, j_aft, d_j, k_pre, k_aft, d_k, s000, s100, s010, s001, s110, s011, s101, s111, Fx, Fy, Fz)

   k_pre = int(Pos_t[0][0])  ;
   k_aft = int(Pos_t[0][0])+1;
   d_k   = Pos_t[0][0] - k_pre;

   j_pre = int(Pos_t[0][1])  ;
   j_aft = int(Pos_t[0][1])+1;
   d_j   = Pos_t[0][1] - j_pre;

   i_pre = int(Pos_t[0][2])  ;
   i_aft = int(Pos_t[0][2])+1;
   d_i   = Pos_t[0][2] - i_pre;

   if ( k_aft == N ) k_aft = 0;
   if ( j_aft == N ) j_aft = 0;
   if ( i_aft == N ) i_aft = 0;

   s000=(1.-d_i)*(1.-d_j)*(1.-d_k);
   s100=(   d_i)*(1.-d_j)*(1.-d_k);
   s010=(1.-d_i)*(   d_j)*(1.-d_k);
   s001=(1.-d_i)*(1.-d_j)*(   d_k);
   s110=(   d_i)*(   d_j)*(1.-d_k);
   s011=(1.-d_i)*(   d_j)*(   d_k);
   s101=(   d_i)*(1.-d_j)*(   d_k);
   s111=(   d_i)*(   d_j)*(   d_k);


//   Fz = (s000+s100+s010+s001+s110+s011+s101+s111)*PartMass;
   Fz = force_h_z[i_pre][j_pre][k_pre]*s000+
        force_h_z[i_aft][j_pre][k_pre]*s100+
        force_h_z[i_pre][j_aft][k_pre]*s010+
        force_h_z[i_pre][j_pre][k_aft]*s001+
        force_h_z[i_aft][j_aft][k_pre]*s110+
        force_h_z[i_pre][j_aft][k_aft]*s011+
        force_h_z[i_aft][j_pre][k_aft]*s101+
        force_h_z[i_aft][j_aft][k_aft]*s111;

   Fy = force_h_y[i_pre][j_pre][k_pre]*s000+
        force_h_y[i_aft][j_pre][k_pre]*s100+
        force_h_y[i_pre][j_aft][k_pre]*s010+
        force_h_y[i_pre][j_pre][k_aft]*s001+
        force_h_y[i_aft][j_aft][k_pre]*s110+
        force_h_y[i_pre][j_aft][k_aft]*s011+
        force_h_y[i_aft][j_pre][k_aft]*s101+
        force_h_y[i_aft][j_aft][k_aft]*s111;

   Fx = force_h_x[i_pre][j_pre][k_pre]*s000+
        force_h_x[i_aft][j_pre][k_pre]*s100+
        force_h_x[i_pre][j_aft][k_pre]*s010+
        force_h_x[i_pre][j_pre][k_aft]*s001+
        force_h_x[i_aft][j_aft][k_pre]*s110+
        force_h_x[i_pre][j_aft][k_aft]*s011+
        force_h_x[i_aft][j_pre][k_aft]*s101+
        force_h_x[i_aft][j_aft][k_aft]*s111;

//printf("fz, fy fx %24.17e %24.17e %24.17e\n", Fz, Fy, Fx);

    Vel_t[0][0] += Fx*dt_vel;
    Vel_t[0][1] += Fy*dt_vel;
    Vel_t[0][2] += Fz*dt_vel;

    Pos_t[0][0] += Vel_t[0][0]*dt_pos;
    Pos_t[0][1] += Vel_t[0][1]*dt_pos;
    Pos_t[0][2] += Vel_t[0][2]*dt_pos;

   // if (ID_part == N*N*N/2+N*N/2+N/2) printf(" pos (%24.17e  %24.17e  %24.17e) \n",Pos_t[ID_part][0],Pos_t[ID_part][1],Pos_t[ID_part][2]);
    //if (ID_part == N*N*N/2+N*N/2+N/2) printf(" vel (%24.17e  %24.17e  %24.17e) \n",Vel_t[ID_part][0],Vel_t[ID_part][1],Vel_t[ID_part][2]);
    //if (ID_part == N*N*N/2+N*N/2+N/2) printf(" acc (%24.17e  %24.17e  %24.17e) \n",Fz,Fy,Fx);

    //if (a_max < Fx) a_max = Fx;
    //if (a_max < Fy) a_max = Fy;
    //if (a_max < Fz) a_max = Fz;

    //if (v_max < Vel_t[ID_part][0]) v_max = Vel_t[ID_part][0];
   // if (v_max < Vel_t[ID_part][1]) v_max = Vel_t[ID_part][1];
    //if (v_max < Vel_t[ID_part][2]) v_max = Vel_t[ID_part][2];


#ifdef PeriodicBondary
    if ( Pos_t[0][0] > N-1 )Pos_t[0][0]-=N;
    if ( Pos_t[0][1] > N-1 )Pos_t[0][1]-=N;
    if ( Pos_t[0][2] > N-1 )Pos_t[0][2]-=N;

    if ( Pos_t[0][0] <   0 )Pos_t[0][0]+=N;
    if ( Pos_t[0][1] <   0 )Pos_t[0][1]+=N;
    if ( Pos_t[0][2] <   0 )Pos_t[0][2]+=N;
#else

    if ( Pos_t[0][0] > N-1 )Pos_t[0][0]=N-1;
    if ( Pos_t[0][1] > N-1 )Pos_t[0][1]=N-1;
    if ( Pos_t[0][2] > N-1 )Pos_t[0][2]=N-1;

    if ( Pos_t[0][0] <   0 )Pos_t[0][0]=  0;
    if ( Pos_t[0][1] <   0 )Pos_t[0][1]=  0;
    if ( Pos_t[0][2] <   0 )Pos_t[0][2]=  0;
#endif

  }

*/


void Ext_potential(CUFFT_Real delta_p[][N][N], double *d_time_h, double t_rot)
{
 double a1 = 1. ;
 double a2 = 0. ;
 double a3 = 0. ;
 double b1 = 0. ;
 double b2 = 1. ;
 double b3 = 0. ;
 double c1 =(double)N/2 ;
 double c2 = (double)N/2 ;
 double c3 = (double)N/2 ;
 const double w = 0.0114;
 const double G =1.839e-17; 
 const double radius = 3200;
 const double M = 8e+11;
 const double m;
 double i_1;
 double j_1;
 double k_1;
 double parameter1 = -1.00e+10;
 double parameter2;
   for (i = 0 ; i < N ; i++ )
   for (j = 0 ; j < N ; j++ )
   for (k = 0 ; k < N ; k++ )
      {
       i_1 = (double)i- c1 - a1*radius*(double)cos(w*t_rot) - b1*radius*(double)sin(w*t_rot);
       j_1 = (double)j- c2 - a2*radius*(double)cos(w*t_rot) - b2*radius*(double)sin(w*t_rot);
       k_1 = (double)k- c3 - a3*radius*(double)cos(w*t_rot) - b3*radius*(double)sin(w*t_rot);

       

       delta_p[i][j][k] = parameter*G*M/sqrt(i_1*i_1 + j_1*j_1 + k_1*k_1) + parameter2*G*m/sqrt((i-(double)N/2)*(i-(double)N/2) +(j-(double)N/2)*(j-(double)N/2) + (k-(double)N/2)*(k-(double)N/2));

  //   if(i==256 && k==256)
//{ printf("%lf\n", delta_p[256][j][256]);}

  //     if( i ==0 && j ==0 && k == 0)
    //   {
      //    printf("G = %24.17e\n", G );
        //  printf("M = %24.17e\n", M );
         // printf("i_1 = %24.17e\n", i_1 );
         // printf("j_1 = %24.17e\n", j_1 );
         // printf("k_1 = %24.17e\n", k_1 );
         // printf("sqrt(i_1*i_1 + j_1*j_1 + k_1*k_1) = %24.17e\n", sqrt(i_1*i_1 + j_1*j_1 + k_1*k_1) );
         // printf("GM = %24.17e\n", G*M );
         
      // }
      // if( i ==0 && j ==0 && k == 0) printf("delta_p = %24.17e\n", (double)delta_p[0][0][0]);

      }
//printf("%lf\n", t_rot);
}

