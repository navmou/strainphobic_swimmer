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

    May. 18, 2022
    Changed the flow to be time dependent by averaging the frozen flows at any time step
    by a time dependent coefficient.

    Jun. 6, 2022
    Added settling to break the symmetry and also added the state \Delta U_p (i.e. py) signal
    since it was creating a good performance in vertical migration (3D and 2D random velocity fields)
    Look at Fig. 7 in paper A we had for parameter P3.
    Since the vertical navigation strategy with \Delta U_p is not dependent on sampling high strain regions
    it might produce a good vertical navigation while the swimmer avoids high strain regions as well.
***/

#define _USE_MATH_DEFINES
#include <cmath>
#include <vector>
#include <iostream>
#include <ctime>
#include <chrono>
#include <sstream>
#include <string>
#include <bits/stdc++.h>
#include <fstream>
#include <H5Cpp.h>
#include <sys/time.h>
#include "field_load.h"

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
int GET_STATE(double local_grad , double trSS , double py);
int GET_ACTION(oneDvec &q , double epsilon);
double PREF_DIR(int action);
double PREF_VS(int action);
double interpolate(double x, double y, triDvec &f , int field_ind);
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
twoDvec generate_As(double sigma , double tau , int n_time_steps , double dt);
twoDvec update_flow(triDvec &V , twoDvec &A , int t);
int get_naive_action(double trSS);
/****************** Global values definition ******************/

//Time parameters
double dt = 0.001 , steady_T = 1.0 , T = 100+steady_T;
int n_time_steps = (int)((T+steady_T)/dt);
int steady_A = 500;
int t_step;

//Flow parameters
const int nx = 400, ny = 4000;
const double  LX = 1.0 , gridspace_x = LX/(double)nx;
const double LY = 10.0 , gridspace_y = LY/(double)ny;
const oneDvec gridx = linspace(0 , LX , nx);
const oneDvec gridy = linspace(0 , LY , ny);
const int n_fields = 50;
const double tau = 1.0;
const double sigma = sqrt((2.0*dt)/(n_fields*tau));
double u_rms = 1.0 , eta = 0.1; // This is the characteristic velocity and length of the flow data
int field_ind;

//Training parameters
double gama = 0.999 , epsilon0 = 0.1 , alpha0 = 0.04 , sigma0 = 1000.0 , E_0 = 2000.0 , epsilon , alpha;
double c1 = 3.0/2.0 * dt , c2 = 1.0/2.0 * dt;
const double PHI = 1.0 , PSI = 2.0;
const double omega_s = PSI*u_rms/eta, vs = PHI*u_rms ;
double v_s;
const double beta = 0.0;
int num_state = 18, num_action = 3;
const double vg_para = u_rms*0.023 , vg_ortho = u_rms*0.0194;
const double delta_uc = u_rms*0.0075;
const double py_threshold = delta_uc/(sqrt(vg_ortho*vg_ortho + vg_para*vg_para));
const double grad_threshold = 2300.0; // 0.25*std(gradx)
const double trSS_threshold = 2.0*log(2.0)*(u_rms*u_rms)/(eta*eta); // Analytical

//Noise parameters
const double mean = 0.0;
const double stddev = 1.0;
const double trans_noise = sqrt(2.0*dt*0.0001);
const double rot_noise = sqrt(2.0*dt*0.00001) ;
default_random_engine generator;
normal_distribution<double> dist(mean , stddev);

//opening HDF5 file
H5File file("Data.h5", H5F_ACC_TRUNC);
//creating groups in the file
Group group1  = file.createGroup("/Qs");
Group group2 = file.createGroup("/Episodes");

/*************** MAIN CODE **************/
int main(){
  
  //setting the random seed
  gettimeofday(&tv, NULL);
  srand(tv.tv_usec);     
  generator.seed(tv.tv_usec);

  //opennign the log file
  ofstream log_file;
  log_file.open("log.txt");
  log_file << "Loading the fields" << endl;

  //flow related variable definition
  string field_name , path , dataset_name , group_name;
  triDvec UXS, UYS, A11S , A12S, A21S , A22S;
  triDvec D2UX11S , D2UX12S , D2UX22S , D2UY11S , D2UY12S , D2UY22S;
  double a11 , a12 , a21 , a22;
  double d2ux11 , d2ux12 , d2ux22 , d2uy11 , d2uy12 , d2uy22;

  //defining the variables
  int state , action , state_new , episodes_number = 10 , heaviside;
  double x_0 , y_0  , phi_0 , x , y , phi , t , R;
  double ux , uy , vx , vy;
  double xmod , ymod;
  double omega_flow;
  double K;
  double trSS;
  double gradx , grady;
  double time_diff , init_time;
  double init_y , last_y , delta_y , reward_sum , trSS_sum;
  double state_transitoin_counter;
  oneDvec reward_sum_list , reward_list , time_list , ns (num_state);
  oneDvec trSS_list , delta_y_list , field_inds;
  twoDvec traj ;
  //twoDvec Q = twoDzeros(num_state , num_action);
  //twoDvec A = generate_As(sigma , tau , n_time_steps , dt);
  int policy[] = {2,0,0,2,0,0,2,0,0,2,0,0,2,0,0,2,0,0};

  
  //loading the fields data on memory from the file
  stringstream flow_number;
  string version;
  auto start_load_field = chrono::high_resolution_clock::now();   //timing the file loading process
  field_name = "st_v3_u_A_d2u_No";
  for(int i=1 ; i <= n_fields ; ++i){
    flow_number.str("");
    flow_number << i;
    version = flow_number.str();
    path = make_path(field_name , version);
    
    UXS.push_back(load_field(path, "/flowData/uys", nx , ny)); //uys is given as UXS because of transposing to preserve the structure
    UYS.push_back(load_field(path, "/flowData/uxs", nx , ny)); //uxs is given as UYS because of transposing to preserve the structure
    A11S.push_back(load_field(path, "/flowData/A11s", nx , ny));
    A12S.push_back(load_field(path, "/flowData/A12s", nx , ny));
    A21S.push_back(load_field(path, "/flowData/A21s", nx , ny));
    D2UX11S.push_back(load_field(path, "/flowData/d2ux11s", nx , ny));
    D2UX12S.push_back(load_field(path, "/flowData/d2ux11s", nx , ny));
    D2UX22S.push_back(load_field(path, "/flowData/d2ux11s", nx , ny));
    D2UY11S.push_back(load_field(path, "/flowData/d2ux11s", nx , ny));
    log_file << "field " << i << " is loaded" << endl;
  }
  
  auto finish_load_field = chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_load_field = finish_load_field - start_load_field;
  log_file << "Fields load completed in : " << elapsed_load_field.count() << " s\n";
  
  //starting the time counter of the code running
  auto start = chrono::high_resolution_clock::now();
  /********************** episode loop **********************/
  for(int episode=0 ; episode < episodes_number ; episode++)
    {
      field_ind = rand()%n_fields; //choosing one of the flows randomly
      field_inds.push_back(field_ind);
      t_step = 0;
      //Producing the new flow
      auto start_episode = chrono::high_resolution_clock::now();       //timing the episode
      //updating the learning and exploration rates
      alpha = alpha0/(1.0+episode/sigma0);
      if(episode < E_0)	{epsilon = epsilon0*(E_0 - episode)/E_0;}
      else{epsilon = 0;}      
      
      //setting the initial values of the variables time = 0 , and position, direction randomly
      t = 0.0;  
      init_time = t;
      state_transitoin_counter = 0.0;
      x_0 = (rand()/double (RAND_MAX))*LX;
      y_0 = (rand()/double (RAND_MAX))*LY;
      phi_0 = (rand()/double (RAND_MAX))*(2.0*M_PI);
      init_y = y_0 ; last_y = y_0;
      reward_sum = 0.0; trSS_sum = 0.0;
      //test_list.clear();  action_list.clear() ; n_mag_list.clear() ; p_mag_list.clear();
      //state_list.clear() ; sn_list.clear() ; sp_list.clear();


      //interpolation of the fields in the current position and time of the flow
      xmod = get_mod(x_0, LX) , ymod = get_mod(y_0, LY);
      ux = interpolate(xmod,ymod,UXS,field_ind);
      uy = interpolate(xmod,ymod,UYS,field_ind);
      a11 = interpolate(xmod,ymod,A11S,field_ind);
      a12 = interpolate(xmod,ymod,A12S,field_ind);
      a21 = interpolate(xmod,ymod,A21S,field_ind);
      a22 = -a11;
      d2ux11 = interpolate(xmod,ymod,D2UX11S,field_ind);
      d2ux12 = interpolate(xmod,ymod,D2UX12S,field_ind);
      d2ux22 = interpolate(xmod,ymod,D2UX22S,field_ind);
      d2uy11 = interpolate(xmod,ymod,D2UY11S,field_ind);
      d2uy12 = -d2ux11;
      d2uy22 = -d2ux12;


      //evaluating the state
      gradx = 4.0*a11*d2ux11 + (a12 + a21)*(d2uy11 + d2ux12);
      grady = 4.0*a11*d2ux12 + (a12 + a21)*(d2ux22 + d2uy12);
      trSS = a11*a11 + 0.5*(a12+a21)*(a12+a21) + a22*a22;
      state = GET_STATE((gradx*cos(phi_0)+grady*sin(phi_0)) , trSS , cos(phi_0));
      state_new = state;
      omega_flow = (a21-a12)/4;
      ns[state]++;

      traj.clear();
      traj.push_back({x_0 , y_0 , phi_0 , gradx , grady , trSS});     //make a list to save the trajectory

      //evaluation of action and preferred direction
      action = policy[state];//GET_ACTION(Q[state],epsilon);//choose action form the Q-matrix
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
	      vx = ux + v_s*cos(phi_0) + (vg_ortho - vg_para)*(sin(phi_0)*cos(phi_0));
	      vy = uy + v_s*sin(phi_0) - vg_ortho + (vg_ortho - vg_para)*(sin(phi_0)*sin(phi_0));
	      
	      x = x_0 + dt*vx + dist(generator)*trans_noise;
	      y = y_0 + dt*vy +  dist(generator)*trans_noise;
	      phi = phi_0 + dt*(omega_flow/2.0 + omega_s*K) +  dist(generator)*rot_noise;
	    
	      //updating the position, direction, strain, and OMEGA
	      x_0 = x ; y_0  = y ; phi_0 = phi;
	      
	      //evaluating the flow information at new position and time
	      xmod = get_mod(x, LX) , ymod = get_mod(y, LY);
	      ux = interpolate(xmod,ymod,UXS,field_ind);
	      uy = interpolate(xmod,ymod,UYS,field_ind);
	      a11 = interpolate(xmod,ymod,A11S,field_ind);
	      a12 = interpolate(xmod,ymod,A12S,field_ind);
	      a21 = interpolate(xmod,ymod,A21S,field_ind);
	      a22 = -a11;
	      d2ux11 = interpolate(xmod,ymod,D2UX11S,field_ind);
	      d2ux12 = interpolate(xmod,ymod,D2UX12S,field_ind);
	      d2ux22 = interpolate(xmod,ymod,D2UX22S,field_ind);
	      d2uy11 = interpolate(xmod,ymod,D2UY11S,field_ind);
	      d2uy12 = -d2ux11;
	      d2uy22 = -d2ux12;
	      
	      omega_flow = (a21-a12)/4;
	      //getting the state
	      gradx = 4.0*a11*d2ux11 + (a12 + a21)*(d2uy11 + d2ux12);
	      grady = 4.0*a11*d2ux12 + (a12 + a21)*(d2ux22 + d2uy12);
	      trSS = a11*a11 + 0.5*(a12+a21)*(a12+a21) + a22*a22;
	      state_new = GET_STATE((gradx*cos(phi)+grady*sin(phi)), trSS , cos(phi));
      	      //saving the trajectory into traj list
	      traj.push_back({x_0 , y_0 , phi_0 , gradx , grady , trSS});

	      trSS_sum += trSS;
	      t += dt;
	      t_step++;
	      if(t>T){break;}
	    }
	  //Updating the state and action
	  state = state_new;
	  ns[state]++;
	  action = policy[state];//GET_ACTION(Q[state], epsilon);   //choose action form the Q-matrix
	  //action = rand()%num_action;
	  //action = get_naive_action(trSS);
	  //action = 1;
	  K = PREF_DIR(action);                     //choose a prefered direction
	  v_s = PREF_VS(action);
	}
      //saving the reward_sum of the last episode in the list
      reward_sum_list.push_back(reward_sum); //to keep track of the total reward per episode
      delta_y_list.push_back((y - init_y)/(vs*T)); // to keep track of the total vertical migration per episode
      //To keep track of total sum over trSS values received as reward per episode
      trSS_list.push_back(trSS_sum/(double)n_time_steps);
      //saving Q matrix of the current episode to file
      dataset_name = make_dataset_name("/Qs/" , episode);

      if(episode % 1 == 0){
	dataset_name = make_dataset_name("/Episodes/traj" , episode);
	write_data_twoD(traj , dataset_name);
      }
      auto finish_episode = chrono::high_resolution_clock::now();
      std::chrono::duration<double> elapsed_episode = finish_episode - start_episode;
      log_file << "episode " << episode << " is done in : " << elapsed_episode.count() << " s\n";
      //cout << "Episode " << episode << " is done in : " << elapsed_episode.count() << " s\n";
    }

  //calculate the elapsed time
  auto finish = chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = finish - start;
  cout << "Elapsed time: " << elapsed.count() << " s\n";

  //saving the reward_sum_list into the file
  write_data_oneD(reward_sum_list , "reward_sum");
  write_data_oneD(delta_y_list , "vertical_migration");
  write_data_oneD(trSS_list , "trSS_sum");
  write_data_oneD(field_inds , "field_inds");
  write_data_oneD(ns , "ns");
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
	  A[i][t+1] = A[i][t] + ((1.0/n_fields) - A[i][t]/tau)*dt + sigma*dist(generator);
	}
    }
  return A;
}


twoDvec update_flow(triDvec &V , twoDvec &A , int t) 
{
  twoDvec v = twoDzeros(nx , ny);
  for (int i = 0; i < n_fields; ++i)
    {
      for (int indx = 0; indx < nx; ++indx)
	{
	  for (int indy = 0; indy < ny; ++indy)
	    {
	      v[indx][indy] += V[i][indx][indy]*A[i][t+steady_A];
	    }
	}
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



int get_naive_action(double trSS)
{
  int a;
  if (trSS <trSS_threshold){a = 1;}
  else
    {
      if(rand()/(double) RAND_MAX < 0.5){a = 0;}
      else{a = 2;}
    }

  return a;
}


//STATE EVALUATION
int GET_STATE(double local_grad , double TrSS , double py){
  int state;
  if(TrSS < trSS_threshold)
    {
      if(local_grad < -grad_threshold)
	{
	  if(py < -py_threshold){state = 0;}
	  else if(py > py_threshold){ state = 2;}
	  else{state = 1;}
	}
      else if(local_grad > grad_threshold)
	{
	  if(py < -py_threshold){state = 6;}
	  else if(py > py_threshold){ state = 8;}
	  else{state = 7;}
	}
      else
	{
	  if(py < -py_threshold){state = 3;}
	  else if(py > py_threshold){ state = 5;}
	  else{state = 4;}
	}
    }
  else
        {
      if(local_grad < -grad_threshold)
	{
	  if(py < -py_threshold){state = 9;}
	  else if(py > py_threshold){ state = 11;}
	  else{state = 10;}
	}
      else if(local_grad > grad_threshold)
	{
	  if(py < -py_threshold){state = 15;}
	  else if(py > py_threshold){ state = 17;}
	  else{state = 16;}
	}
      else
	{
	  if(py < -py_threshold){state = 12;}
	  else if(py > py_threshold){ state = 14;}
	  else{state = 13;}
	}
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
double interpolate(double x, double y, triDvec &f , int field_ind)
{    
  int x0_ind , y0_ind;
  int x1_ind , y1_ind;
  double d1x , d2x , d1y , d2y;
  
  x0_ind =  floor(x/gridspace_x);
  if(x0_ind == nx-1)
    {
      x1_ind = 0;
      d1x = LX - x;
      d2x = x - gridx[x0_ind];
    }
  else
    {
      x1_ind = x0_ind+1;
      d1x = gridx[x1_ind] - x;
      d2x = x - gridx[x0_ind];
    }

  y0_ind =  floor(y/gridspace_y);
  if(y0_ind == ny-1)
    {
      y1_ind = 0;
      d1y = LY - y;
      d2y = y - gridy[y0_ind];
    }
  else
    {
      y1_ind = y0_ind+1;
      d1y = gridy[y1_ind] - y;
      d2y = y - gridy[y0_ind];
    }

  //calculate the time dependent flow around the swimmer position for interpolation
  //g00 = f[x0,y0] , g01 = f[x0,y1] , g10 = f[x1,y0] , g11 = f[x1,y1]
  double g00 , g01 , g10 , g11 ;
  g00 = f[field_ind][y0_ind][x0_ind];
  g10 = f[field_ind][y1_ind][x0_ind];
  g01 = f[field_ind][y0_ind][x1_ind];
  g11 = f[field_ind][y1_ind][x1_ind];
  
  double fxy1 = (((d1x)/gridspace_x) * g00) + (((d2x)/gridspace_x)*g10);
  double fxy2 = (((d1x)/gridspace_x) * g01) + (((d2x)/gridspace_x)*g11);

  double fxy = (((d1y)/gridspace_y) * fxy1) + (((d2y)/gridspace_y)*fxy2);
  
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
  //ss << "../" << field_name << "-" << component << ".txt";
  ss << "/cephyr/users/navidm/Vera/FLOW/" << field_name << component << ".hdf5";
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
