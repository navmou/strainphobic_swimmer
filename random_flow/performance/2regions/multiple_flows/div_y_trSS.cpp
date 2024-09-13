
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
twoDvec twoDzeros(int D1 , int D2);
int GET_STATE(double gradx , double trSS);
int GET_ACTION(oneDvec &q , double epsilon);
int get_naive_action(double trSS);
oneDvec PREF_DIR(int action);
double PREF_VS(int action);
double interpolate(double x, double y, twoDvec &f );
double get_sn(oneDvec &n, double a11 , double a12 , double a21 , double a22);
double get_sp(oneDvec &n , oneDvec &p, double a11 , double a12 , double a21 , double a22);
double AB(double X1, double F1X, double F0X);
string make_path(string field_name , string component);
string make_dataset_name(string constant_part , int variable_part);
void printQ(twoDvec &v);
void write_data_twoD(twoDvec &data , string dataset_name);
void write_data_oneD(oneDvec &data , string dataset_name);
void write_data_oneD_int(vector<int> &data , string dataset_name);
void printing(oneDvec &vec);
double get_mod(double x , double L);
/****************** Global values definition ******************/

double  L0 = 0.01 ,  gridspace = L0/1024;
oneDvec grid = linspace(0 , L0 , 1024);
double dt = 0.01 , steady_T = 500.0 , T = 1000+steady_T;
double c1 = 3.0/2.0 * dt , c2 = 1.0/2.0 * dt;
double omega_s = 4.5 , vs = 0.001 , v_s;
const double beta = 0.0;
const double grad_threshold_a = 1000.0;
const double grad_threshold_b = 3000.0;
const double trSS_threshold = 1.7;

int num_state = 10, num_action = 5;


const double mean = 0.0;
const double stddev = 1.0;
const double trans_noise = sqrt(2.0*dt*0.000000001);
const double rot_noise = sqrt(2.0*dt*0.0002) ;
default_random_engine generator;
normal_distribution<double> dist(mean , stddev);

/*
Policies:
naive: swimm upward outside S_0 and stop inside S_0
beta 0.0 : {4,3,1,1,1,4,1,1,3,1};
beta 0.1 : {4,1,1,3,1,4,3,1,3,1};
beta 0.2 : {4,1,1,1,1,4,3,1,3,1};
beta 0.3 : {4,1,3,1,1,4,1,1,1,3};
beta 0.4 : {4,1,3,1,1,1,3,1,3,1};
beta 0.5 : {4,3,1,1,1,4,1,3,1,1};
beta 0.6 : {4,2,2,0,3,2,2,2,2,1};
beta 0.7 : {4,3,3,1,3,2,1,3,1,0};
beta 0.8 : {1,0,3,0,4,1,3,2,0,0};
beta 0.9 : {0,4,3,2,0,3,0,0,1,3};
beta 1.0 : {1,3,2,2,4,1,3,2,3,0};
*/


int policy[] =;

//opening HDF5 file
H5File file("beta0.h5", H5F_ACC_TRUNC);
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
  vector<string> fields_list = {"11" , "12" , "13" , "14" , "15" , "16" , "17" , "18" , "19" , "20"};
  //defining the variables of the field
  triDvec UX, UY, A11, A12, A21 , A22;
  triDvec D2UX11 , D2UX12 , D2UX22 , D2UY11 , D2UY12 , D2UY22;
  double a11 , a12 , a21 , a22;
  double d2ux11 , d2ux12 , d2ux22 , d2uy11 , d2uy12 , d2uy22;
  //loading the fields data into memory from the file
  for(int i=0 ; i < (int)fields_list.size() ; ++i){
    field_name = fields_list[i];
    path = make_path(field_name , "uxs");
    UX.push_back(read_field(path));
    path = make_path(field_name , "uys");
    UY.push_back(read_field(path));
    path = make_path(field_name , "A11s");
    A11.push_back(read_field(path));
    path = make_path(field_name , "A12s");
    A12.push_back(read_field(path));
    path = make_path(field_name , "A21s");
    A21.push_back(read_field(path));
    path = make_path(field_name , "A22s");
    A22.push_back(read_field(path));
    path = make_path(field_name , "d2ux11");
    D2UX11.push_back(read_field(path));
    path = make_path(field_name , "d2ux12");
    D2UX12.push_back(read_field(path));
    path = make_path(field_name , "d2ux22");
    D2UX22.push_back(read_field(path));
    path = make_path(field_name , "d2uy11");
    D2UY11.push_back(read_field(path));
    path = make_path(field_name , "d2uy12");
    D2UY12.push_back(read_field(path));
    path = make_path(field_name , "d2uy22");
    D2UY22.push_back(read_field(path));
  }
  
  auto finish_load_field = chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_load_field = finish_load_field - start_load_field;
  log_file << "Fields load completed in : " << elapsed_load_field.count() << " s\n";
  //starting the time counter of the code running
  auto start = chrono::high_resolution_clock::now();
  //initializing the state visit counter ns with zero
  //vector<int> ns = {0,0,0,0,0,0,0,0,0};
  //defining the variables
  int state , action , state_new , field_index , episodes_number = 10000;
  double x_0 , y_0  , phi_0 , x , y , phi , t ;
  double ux , uy ;
  double xmod , ymod;
  double omega_flow;
  oneDvec K (2);
  oneDvec reward_sum_list , reward_list , time_list;
  twoDvec traj ;
  double trSS;
  double gradx , grady;
  double init_y , trSS_sum;
  oneDvec trSS_list , delta_y_list;
  
  double trSS_counter;
  double timer1 , timer2;
  twoDvec timers;
  
  
  /********************** episode loop **********************/
  for(int episode=0 ; episode < episodes_number ; episode++)
    {
      field_index = rand() % 10; //choosing the velocity field randomly    
      auto start_episode = chrono::high_resolution_clock::now();
      //setting the initial values of the variables time = 0 , and position, direction randomly
      t = 0.0;  //time step counter
      timer1 = 0.0; timer2 = 0.0;
      x_0 = (rand()/double (RAND_MAX))*L0;
      y_0 = (rand()/double (RAND_MAX))*L0;
      phi_0 = (rand()/double (RAND_MAX))*(2.0*M_PI);
      init_y = y_0 ;
      trSS_counter = 0.0;
      trSS_sum = 0.0;
      traj.clear();
      //test_list.clear();  action_list.clear() ; n_mag_list.clear() ; p_mag_list.clear();
      //state_list.clear() ; sn_list.clear() ; sp_list.clear();

      traj.push_back({x_0 , y_0});     //make a list to save the trajectory

      //evaluate the A matrix at initial position
      xmod = get_mod(x_0, L0) , ymod = get_mod(y_0, L0);
      a11 = interpolate(xmod,ymod,A11[field_index]);
      a12 = interpolate(xmod,ymod,A12[field_index]);
      a21 = interpolate(xmod,ymod,A21[field_index]);
      a22 = interpolate(xmod,ymod,A22[field_index]);
      //evaluate the velocities at the initial position
      ux = interpolate(xmod,ymod,UX[field_index]);
      uy = interpolate(xmod,ymod,UY[field_index]);
      //evaluate the second order derivatives of the flow
      d2ux11 = interpolate(xmod,ymod,D2UX11[field_index]);
      d2ux12 = interpolate(xmod,ymod,D2UX12[field_index]);
      d2ux22 = interpolate(xmod,ymod,D2UX22[field_index]);
      d2uy11 = interpolate(xmod,ymod,D2UY11[field_index]);
      d2uy12 = interpolate(xmod,ymod,D2UY12[field_index]);
      d2uy22 = interpolate(xmod,ymod,D2UY22[field_index]);
	  
      //getting the state
      gradx = d2ux11 + 0.5*(d2uy12+d2ux22);
      grady = d2uy22 + 0.5*(d2ux12+d2uy11);
      trSS = a11*a11 + 0.5*(a12+a21)*(a12+a21) + a22*a22;
      state = GET_STATE(grady , trSS);
      state_new = state;
      omega_flow = (a21-a12)/4;
      //ns[state]++;
      //evaluation of action and preferred direction
      action = policy[state];    //choose action form the Q-matrix
      //action = rand()%num_action;
      //action = get_naive_action(trSS);
      //action = 1;
      
      //action_list.push_back(action);            //save the action history
      K = PREF_DIR(action);                     //choose a preferred direction
      v_s = PREF_VS(action);
      /********************** time step loop **********************/
      while(t < T)
	{
	  while(state_new == state)
	    {
	      x = x_0 + dt*(ux + v_s*cos(phi_0)) + dist(generator)*trans_noise;
	      y = y_0 + dt*(uy + v_s*sin(phi_0)) +  dist(generator)*trans_noise;
	      phi = phi_0 + dt*(omega_flow/2 + (omega_s * (-K[0]*y + K[1]*x))) +  dist(generator)*rot_noise;
	    
	      //saving the trajectory into traj list
	      traj.push_back({x , y});
	      //updating the position, direction, strain, and OMEGA
	      x_0 = x ; y_0  = y ; phi_0 = phi;
	      
	      //evaluating the components of A matrix at new position
	      xmod = get_mod(x, L0) , ymod = get_mod(y, L0);
	      a11 = interpolate(xmod,ymod,A11[field_index]);
	      a12 = interpolate(xmod,ymod,A12[field_index]);
	      a21 = interpolate(xmod,ymod,A21[field_index]);
	      a22 = interpolate(xmod,ymod,A22[field_index]);
	      //evaluate the velocities at the initial position
	      ux = interpolate(xmod,ymod,UX[field_index]);
	      uy = interpolate(xmod,ymod,UY[field_index]);
	      //evaluate the second order derivatives of the flow
	      d2ux11 = interpolate(xmod,ymod,D2UX11[field_index]);
	      d2ux12 = interpolate(xmod,ymod,D2UX12[field_index]);
	      d2ux22 = interpolate(xmod,ymod,D2UX22[field_index]);
	      d2uy11 = interpolate(xmod,ymod,D2UY11[field_index]);
	      d2uy12 = interpolate(xmod,ymod,D2UY12[field_index]);
	      d2uy22 = interpolate(xmod,ymod,D2UY22[field_index]);
	      
	      omega_flow = (a21-a12)/4;
	      //getting the state
	      gradx = d2ux11 + 0.5*(d2uy12+d2ux22);
	      grady = d2uy22 + 0.5*(d2ux12+d2uy11);
	      trSS = a11*a11 + 0.5*(a12+a21)*(a12+a21) + a22*a22;
	      state_new = GET_STATE(grady , trSS);
	  
	      t += dt;

	      if(t>steady_T)
		{
		  if(trSS < trSS_threshold){timer1+=dt;}
		  else{timer2+=dt;}
		  trSS_sum += trSS;
		  trSS_counter++;
		}
	      
	      if(t>T){break;}
	    }

	  //Updating the state and action
	  state = state_new;
	  action = policy[state];    //choose action form the Q-matrix
	  //action = rand()%num_action;
	  //action = get_naive_action(trSS);
	  //action = 1;
	  K = PREF_DIR(action);                     //choose a prefered direction
	  v_s = PREF_VS(action);

	}
      //saving the reward_sum of the last episode in the list
      delta_y_list.push_back((y - init_y)/(vs*T)); // to keep track of the total vertical migration per episode
      trSS_list.push_back(trSS_sum/trSS_counter);   //To keep track of total sum over trSS values received as reward per episode
      timers.push_back({timer1 , timer2});
      
      if(episode % 500 == 0){
	dataset_name = make_dataset_name("/Episodes/traj" , episode);
	write_data_twoD(traj , dataset_name);
      }
      
      
      auto finish_episode = chrono::high_resolution_clock::now();
      std::chrono::duration<double> elapsed_episode = finish_episode - start_episode;
      log_file << "Episode " << episode << " is done in : " << elapsed_episode.count() << " s\n";
    }

  //calculate the elapsed time
  auto finish = chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = finish - start;
  cout << "Elapsed time: " << elapsed.count() << " s\n";

  //saving the reward_sum_list into the file
  write_data_oneD(delta_y_list , "vertical_migration");
  write_data_oneD(trSS_list , "trSS_sum");
  write_data_twoD(timers , "times");
  
  //closing the file and it's components
  group2.close();
  file.close();
  log_file.close();
  
  return 0;
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
int GET_STATE(double grad_sx , double TrSS){
  int state;
  if(TrSS < trSS_threshold)
    {
      if(grad_sx < -grad_threshold_b){state = 0;}
      else if(grad_sx > grad_threshold_b){state = 4;}
      else
	{
	  if(grad_sx < -grad_threshold_a){state = 1;}
	  else if(grad_sx > grad_threshold_a){state = 3;}
	  else{state = 2;}
	}
    }
  else
    {
      if(grad_sx < -grad_threshold_b){state = 5;}
      else if(grad_sx > grad_threshold_b){state = 9;}
      else
	{
	  if(grad_sx < -grad_threshold_a){state = 6;}
	  else if(grad_sx > grad_threshold_a){state = 8;}
	  else{state = 7;}
	}
    }
  return state;
}

int get_naive_action(double trSS)
{
  int action;
  if(trSS > trSS_threshold){action = 1;}
  else{action = 4;}
  return action;
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
oneDvec PREF_DIR(int action){
  oneDvec K;
  switch(action){
  case 0:
    K = {1.0 , 0.0};
    break;
  case 1:
    K = {0.0 , 1.0};
    break;
  case 2:
    K = {-1.0 , 0.0};
    break;
  case 3:
    K = {0.0 , -1.0};
    break;
  case 4:
    K = {0.0 , 0.0};
    break;
  }
  return K;
}


double PREF_VS(int action)
{
  double v;
  if(action == 4){v = 0.0;}
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
  //ss << "../../../../../../../../FLOW/new-flow/" << field_name << "-" << component << ".txt";
  ss << "/cephyr/users/navidm/Vera/FLOW/" << field_name << "-" << component << ".txt";
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








