/***
    Strainphobic swimmer
    The goal is to train a swimmer that is capable of avoiding high strain regions
    The reward function is defined with a trade-off parameter \beta in the way that
    \beta = 1 corresponds to a swimmer with the only goal of avoiding high strain regions
    and \beta = 0 corresponds to a swimmer with the only goal of swimming in positive 
    vertical direction. Other values of \beta between 0 and 1 correspond to different 
    combinations of these two goals.
    
    For the state the second order derivatvie of the flow and the absolute value 
    of the flow trSS is used with global information (components in the lab frame)

    Feb. 3,2022
    Changed the state signals to be a combination of the 3 signals u_y, TrSS, and the y_component of the TrSS gradient.
    Dividied each signal to 3 levels except TrSS regions which has only 2 levels (favorable and unfavorable)
    
    Feb.4 , 2022
    Reduced the actions to be only 3. Up, Down, and Stop.
    Also, changed the thresholds
    
    Mar. 16, 2022
    Using only signals \nabla TrSS and TrSS in local frame:
    \nabla TrSS \cdot \hat{\ve n} and TrSS is local itself
    I expect to see a good performance for strainphobic target but not for upward
***/

#define _USE_MATH_DEFINES
#include <cmath>
#include <vector>
#include <iostream>
#include <ctime>
#include "field-read.h"
#include <chrono>
#include <sstream>
#include <string>
#include <bits/stdc++.h>
#include <fstream>
#include <H5Cpp.h>
#include <sys/time.h>


using namespace std;
using namespace H5;

struct timeval tv;

typedef vector<vector<vector<double>>> triDvec;
typedef vector<vector<double>> twoDvec;
typedef vector<double> oneDvec;

double magnitude(oneDvec &v);
oneDvec normalize(oneDvec &v);
oneDvec linspace(double starting , double ending , int num);
vector<int> oneDzeros_int(int size_of);
double dot_prod(oneDvec &v1 , oneDvec &v2);
oneDvec oneDzeros(int size_of);
twoDvec twoDzeros(int D1 , int D2);
int GET_STATE(double local_grad , double trSS);
int GET_ACTION(oneDvec &q , double epsilon);
double PREF_DIR(int action);
double PREF_VS(int action);
double interpolate(double x, double y, twoDvec &f );
double get_sn(oneDvec &n, double a11 , double a12 , double a21 , double a22);
double get_sp(oneDvec &n , oneDvec &p, double a11 , double a12 , double a21 , double a22);
double AB(double X1, double F1X, double F0X);
double sign(double x);
string make_path(string field_name , string component);
string make_dataset_name(string constant_part , int variable_part);
void printQ(twoDvec &v);
void write_data_twoD(twoDvec &data , string dataset_name);
void write_data_oneD(oneDvec &data , string dataset_name);
void write_data_oneD_int(vector<int> &data , string dataset_name);
void printing(oneDvec &vec);
double get_mod(double x , double L);
twoDvec update_flow(triDvec &V , twoDvec &A , int t);
twoDvec generate_As(double sigma , double tau , int n_time_steps , double dt);
/****************** Global values definition ******************/
double  L0 = 25.0 ,  gridspace = L0/10000;
oneDvec grid = linspace(0 , L0 , 10000);
double gama = 0.999 , epsilon0 = 0.1 , alpha0 = 0.04 , sigma0 = 1000.0 , E_0 = 2000.0 , epsilon , alpha;
double dt = 0.01 , steady_T = 0.0 , T = 5.0+steady_T;
int steady_A = 500;
int n_time_steps = int(T/dt);
double c1 = 3.0/2.0 * dt , c2 = 1.0/2.0 * dt;
double omega_s = 4.5 , vs = 0.001 , v_s;
const double beta = 0.0;
const double uy_threshold = 0.0005;
const double grad_threshold = 4000.0;
const double trSS_threshold = 1.7;
int num_state = 6, num_action = 3;
const int n_fields = 3;
const int field_x = 400, field_y = 10000;
const double tau = 1.0;
const double sigma = sqrt((2.0*dt)/(n_fields*tau));
const double mean = 0.0;
const double stddev = 1.0;
const double trans_noise = sqrt(2.0*dt*0.000000001);
const double rot_noise = sqrt(2.0*dt*0.0002) ;
default_random_engine generator;
normal_distribution<double> dist(mean , stddev);

//opening HDF5 file
H5File file("Data.h5", H5F_ACC_TRUNC);
//creating groups in the file
Group group1  = file.createGroup("/Qs");
Group group2 = file.createGroup("/Episodes");

/*************** MAIN CODE **************/
int main(){
  gettimeofday(&tv, NULL);
  srand(tv.tv_usec);     //setting the random seed
  ofstream log_file;
  log_file.open("log.txt");
  log_file << "Loading the fields" << endl;
  auto start_load_field = chrono::high_resolution_clock::now();
  //defining the variables
  string field_name , path , dataset_name , group_name;
  //list of the fields names
  vector<string> fields_list = {"11" , "12" , "13"};
  //defining the variables of the field
  triDvec UXS, UYS, A11S , A12S, A21S , A22S;
  triDvec D2UX11S , D2UX12S , D2UX22S , D2UY11S , D2UY12S , D2UY22S;
  twoDvec U , V , A11 , A12 , A21 , A22 , D2UX11 , D2UX12 , D2UX22 , D2UY11 , D2UY12 , D2UY22;
  double a11 , a12 , a21 , a22;
  double d2ux11 , d2ux12 , d2ux22 , d2uy11 , d2uy12 , d2uy22;
  //loading the fields data into memory from the file
  for(int i=0 ; i < (int)fields_list.size() ; ++i){
    field_name = fields_list[i];
    path = make_path(field_name , "uxs");
    UXS.push_back(read_field(path));
    path = make_path(field_name , "uys");
    UYS.push_back(read_field(path));
    path = make_path(field_name , "A11s");
    A11S.push_back(read_field(path));
    path = make_path(field_name , "A12s");
    A12S.push_back(read_field(path));
    path = make_path(field_name , "A21s");
    A21S.push_back(read_field(path));
    path = make_path(field_name , "A22s");
    A22S.push_back(read_field(path));
    /*
    path = make_path(field_name , "d2ux11");
    D2UX11S.push_back(read_field(path));
    path = make_path(field_name , "d2ux12");
    D2UX12S.push_back(read_field(path));
    path = make_path(field_name , "d2ux22");
    D2UX22S.push_back(read_field(path));
    path = make_path(field_name , "d2uy11");
    D2UY11S.push_back(read_field(path));
    path = make_path(field_name , "d2uy12");
    D2UY12S.push_back(read_field(path));
    path = make_path(field_name , "d2uy22");
    D2UY22S.push_back(read_field(path));
    */
  }
  
  auto finish_load_field = chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_load_field = finish_load_field - start_load_field;
  cout << "Fields load completed in : " << elapsed_load_field.count() << " s\n";
  //starting the time counter of the code running
  auto start = chrono::high_resolution_clock::now();
  //initializing the state visit counter ns with zero
  //vector<int> ns = {0,0,0,0,0,0,0,0,0};
  //defining the variables
  twoDvec A = generate_As(sigma , tau , n_time_steps , dt);
  twoDvec state = twoDzeros(3,n_time_steps);
  int n_points = 5000;
  int corr_L = 500;
  oneDvec corr;
  twoDvec corr1 = twoDzeros(n_points, n_time_steps);
  /********************** episode loop **********************/
  for(int t=0 ; t < n_time_steps ; t++)
    {
      // Updating the flow
      U = update_flow(UXS , A , t); V = update_flow(UYS , A , t);
      A11 = update_flow(A11S , A , t); A21 = update_flow(A21S, A , t);
      A12 = update_flow(A12S , A , t); A22 = update_flow(A11S , A , t);
      
      corr = oneDzeros(corr_L);
      for (int point = 0; point < n_points; ++point)
	{
	  int x = rand()%10000;
	  int y = rand()%400;
	  
	  for (int l=0; l < corr_L; ++l)
	    {
	      if((x+l) < 10000)
		{
		  corr[l] += U[y][x]*U[y][x+l];
		}
	      else
		{
		  corr[l] += U[y][x]*U[y][x+l-10000];
		}
	    }
	}
      
      for (int l = 0; l < corr_L; ++l)
	{
	  corr[l] = corr[l]/(double)n_points;
	}

      double std = corr[0];
      for (int l = 0; l < corr_L; ++l)
  	{
	  corr[l] = corr[l]/std;
	}
      
      write_data_oneD(corr , make_dataset_name("corr" , t));
      if((t+1)%100 == 0){cout << (t+1)*dt << " seconds done" << endl;}
	
    }	  
  
  write_data_twoD(A , "As");
  write_data_twoD(U , "U");
  write_data_twoD(V , "V");
  //closing the file and it's components
  group1.close();
  group2.close();
  file.close();
  log_file.close();

  return 0;
}


twoDvec generate_As(double sigma , double tau , int n_time_steps , double dt)
{
  twoDvec A = twoDzeros(n_fields, (n_time_steps+steady_A));
  for (int t = 0; t < (n_time_steps-1+steady_A); ++t)
    {
      for (int i = 0; i < n_fields; ++i)
	{
	  A[i][t+1] = A[i][t] - (A[i][t]/tau)*dt + sigma*dist(generator);
	}
    }
  return A;
}


twoDvec update_flow(triDvec &V , twoDvec &A , int t) 
{
  twoDvec v = twoDzeros(field_x , field_y);
  for (int i = 0; i < n_fields; ++i)
    {
      for (int indx = 0; indx < field_x; ++indx)
	{
	  for (int indy = 0; indy < field_y; ++indy)
	    {
	      v[indx][indy] += V[i][indx][indy]*A[i][t+steady_A];
	    }
	}
    }
  return v;
}



oneDvec oneDzeros(int size_of)
{
  oneDvec v;
  for(int i=0 ; i<size_of ; i++){
    v.push_back(0.0);
  }
  return v;
}

  
void printQ(twoDvec &v)
{
  for(int i =0 ; i < (int) v.size(); i++)
    {
      for(int j=0 ; j < (int)v[i].size();j++)
	    {
	      cout << v[i][j] << "\t";
	    }
	  cout << endl;
	}

    }

double magnitude(oneDvec &v)
{
  double mag = 0;
  for(int i =0 ; i < (int)v.size(); i++){
    mag += v[i]*v[i];
  }
  return sqrt(mag);
}

oneDvec normalize(oneDvec &v)
{
  double mag = 0;
  mag = magnitude(v);
  oneDvec normal_v;
  for(int i =0 ; i < (int)v.size(); i++){
    normal_v.push_back(v[i]/mag);
  }
  return normal_v;
}

oneDvec linspace(double starting , double ending , int num)
{
  oneDvec v;
  double delta = (ending - starting)/(num);
  for(int i=0; i< num; i++){
    v.push_back(starting+delta*i);
  }
  return v;
}

vector<int> oneDzeros_int(int size_of)
{
  vector<int> v;
  for(int i=0 ; i<size_of ; i++){
    v.push_back(0);
  }
  return v;
}


double sign(double x)
{
  if(x>=0){return 1.0;}
  else{return -1.0;}
}


double dot_prod(oneDvec &v1 , oneDvec &v2)
{
  double result = 0;
  for(int i = 0 ; i < (int)v1.size() ; i++){
    result += v1[i]*v2[i];
  }
  return result;
}


// MAKES A TWO DIMENSIONAL VECTOR OF ZEROS
twoDvec twoDzeros(int D1 , int D2)
{
  twoDvec dummy;
  oneDvec dummy_col;
  for(int i = 0; i < D1 ; ++i){
    dummy_col.clear();
    for(int j = 0; j < D2 ; ++j){
      dummy_col.push_back(0.0);
    }
    dummy.push_back(dummy_col);
  }
  return dummy;
}



//STATE EVALUATION
int GET_STATE(double local_grad , double TrSS){
  int state;
  if(TrSS < trSS_threshold)
    {
      if(local_grad < -grad_threshold){state = 0;}
      else if(local_grad > grad_threshold){ state = 2;}
      else{state = 1;}
    }
  else
    {
      if(local_grad < -grad_threshold){state = 3;}
      else if(local_grad > grad_threshold){ state = 5;}
      else{state = 4;}
    }
  return state;
}


// GETTING ACTION
int GET_ACTION(oneDvec &q , double epsilon){
  int Naction = 1;
  int action = 0;
  int num_actions = q.size();
  vector<int> dummy_list;
  dummy_list.push_back(action);

  if(rand()/double (RAND_MAX) > epsilon)
    {
      for(int i = 0 ; i < (num_actions-1) ; i++)
	{
	  if(q[i+1] > q[action])
	    {
	      action = i+1;
	      Naction = 1;
	      dummy_list.clear();
	    }else if(q[i+1] == q[action])
	    {
	      dummy_list.push_back(i+1);
	      Naction++;
	    }else
	    {
	      action = action;
	    }
	}
      if(Naction > 1){
	int index = rand() % dummy_list.size();
	action = dummy_list[index];

      }
    }else
    {
      action = rand() % num_action;
    }
  return action;
}

// CHOOSING THE PREFERED DIRECTION USING CHOSEN ACTION
double PREF_DIR(int action){
  double K;
  switch(action){
  case 0:
    K = -1.0;
    break;
  case 1:
    K = 0.0;
    break;
  case 2:
    K = 1.0;
    break;
  }
  return K;
}


double PREF_VS(int action)
{
  double v;
  if(action == 1){v = 0.0;}
  else{v = vs;}
  return v;
}


//Get the modulus
double get_mod(double x , double L){
  double R;
  if(x>=0){
    R = x - floor(x/L)*L;
  }else {
    x = -x;
    R = L - (x - floor(x/L)*L) ;
  }
  return R;
}


// BIILINEAR INTERPOLATION
double interpolate(double x, double y, twoDvec &f )
{    
  int x0_ind , y0_ind;
  int x1_ind , y1_ind;
  double d1x , d2x , d1y , d2y;
  
  x0_ind =  floor(x/gridspace);
  if(x0_ind == 1023)
    {
      x1_ind = 0;
      d1x = L0 - x;
      d2x = x - grid[x0_ind];
    }
  else
    {
      x1_ind = x0_ind+1;
      d1x = grid[x1_ind] - x;
      d2x = x - grid[x0_ind];
    }

  y0_ind =  floor(y/gridspace);
  if(y0_ind == 1023)
    {
      y1_ind = 0;
      d1y = L0 - y;
      d2y = y - grid[y0_ind];
    }
  else
    {
      y1_ind = y0_ind+1;
      d1y = grid[y1_ind] - y;
      d2y = y - grid[y0_ind];
    }
  
  double fxy1 = (((d1x)/gridspace) * f[x0_ind][y0_ind]) + (((d2x)/gridspace)*f[x1_ind][y0_ind]);
  double fxy2 = (((d1x)/gridspace) * f[x0_ind][y1_ind]) + (((d2x)/gridspace)*f[x1_ind][y1_ind]);

  double fxy = (((d1y)/gridspace) * fxy1) + (((d2y)/gridspace)*fxy2);
  
  return fxy;
}


// S_N CALCULATION
double get_sn(oneDvec &n , double a11 , double a12 , double a21 , double a22){
  double s_n = a11*n[0]*n[0] + n[1]*(a12*n[0] + a21*n[0] + a22*n[1]);
  return s_n;
}




// S_P CALCULATION
double get_sp(oneDvec &n , oneDvec &p , double a11 , double a12 , double a21 , double a22){
  double s_p = a11*n[0]*p[0] + 0.5*(a12+a21)*n[1]*p[0] + 0.5*(a12+a21)*n[0]*p[1] + a22*n[1]*p[1];
  return s_p;
}


string make_path(string field_name , string component){
  stringstream ss;
  string path;
  ss << field_name << "-" << component << ".txt";
  //ss << "/cephyr/users/navidm/Vera/FLOW/" << field_name << "-" << component << ".txt";
  path = ss.str();
  return path;
}

string make_dataset_name(string constant_part , int variable_part){
  stringstream ss;
  ss << constant_part << variable_part;
  return ss.str();
}

void write_data_twoD(twoDvec &data , string dataset_name){
  int nrow = data.size();
  int ncol = data[0].size();

  oneDvec dummy;
  for(int i = 0 ; i < nrow ; i++){
    for(int j = 0 ; j < ncol ; j++){
      dummy.push_back(data[i][j]);
    }
  }
  // dataset dimensions
  hsize_t dimsf[2];
  dimsf[0] = nrow;
  dimsf[1] = ncol;
  DataSpace dataspace(2, dimsf);

  DataType datatype(H5::PredType::NATIVE_DOUBLE);
  DataSet dataset = file.createDataSet(dataset_name, datatype, dataspace);

  // dataset.write(vec2d.data(), H5::PredType::NATIVE_DOUBLE);
  dataset.write(&dummy[0], H5::PredType::NATIVE_DOUBLE);

  dataset.close();
  dataspace.close();
  //file.close();

}


void write_data_oneD(oneDvec &data , string dataset_name){
  // dataset dimensions
  hsize_t dimsf[1];
  dimsf[0] = data.size();
  DataSpace dataspace(1, dimsf);

  DataType datatype(H5::PredType::NATIVE_DOUBLE);
  DataSet dataset = file.createDataSet(dataset_name, datatype, dataspace);

  // dataset.write(vec2d.data(), H5::PredType::NATIVE_DOUBLE);
  dataset.write(&data[0], H5::PredType::NATIVE_DOUBLE);

  dataset.close();
  dataspace.close();
  //  file.close();


}

void write_data_oneD_int(vector<int> &data , string dataset_name){
  // dataset dimensions
  hsize_t dimsf[1];
  dimsf[0] = data.size();
  DataSpace dataspace(1, dimsf);

  DataType datatype(H5::PredType::NATIVE_INT32);
  DataSet dataset = file.createDataSet(dataset_name, datatype, dataspace);

  // dataset.write(vec2d.data(), H5::PredType::NATIVE_DOUBLE);
  dataset.write(&data[0], H5::PredType::NATIVE_INT32);

  dataset.close();
  dataspace.close();
  //  file.close();
}




void printing(oneDvec &vec)
{
  for(int i = 0 ; i < (int)vec.size() ; ++i)
    {
      cout << vec[i] << endl ;
    }
}














