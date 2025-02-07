#include <iostream>
#include <string>
#include <vector>
#include <H5Cpp.h>

using namespace std;
using namespace H5;

typedef vector<vector<vector<double>>> triDvec;
typedef vector<vector<double>> twoDvec;
typedef vector<double> oneDvec;

twoDvec load_field(string file_path , string group_path , int nx , int ny)
{
    // Open HDF5 file handle, read only
    H5File fp(file_path,H5F_ACC_RDONLY);

    // access the required dataset by path name
    DataSet dset = fp.openDataSet(group_path);

    // get the dataspace
    DataSpace dspace = dset.getSpace();

    // get the dataset type class
    H5T_class_t type_class = dset.getTypeClass();

    // get the size of the dataset
    hsize_t rank;
    hsize_t dims[2];
    rank = dspace.getSimpleExtentDims(dims, NULL); // rank = 1
    //cout<<"Datasize: "<<dims[0]<<endl; // this is the correct number of values

    // Define the memory dataspace
    hsize_t dimsm[1];
    dimsm[0] = dims[0];
    DataSpace memspace (1,dimsm);


    // create a vector the same size as the dataset
    oneDvec data;
    data.resize(dims[0]);
    //cout<<"Vectsize: "<<data.size()<<endl;

    //Initializing the data_out in memory
    double *data_out = (double *) malloc(data.size()*sizeof(double));
    for (int i=0;i<data.size();i++)
      {
        data_out[i]=0;
      }
    
    
    // pass pointer to the array (or vector) to read function, along with the data type and space.
    dset.read(data_out, PredType::NATIVE_DOUBLE, memspace, dspace);
    
    /*
    //to return the data as it is saved
    // I used this to calculate the correlations of the flow
    int c = nx;
    vector<vector<double>> u;
    vector<double> dumb;
    for(int i = 0  ; i < ny ; i++){
        dumb.clear();
        for(int j = 0; j < nx ; j++){
                int index = i*c + j;
                double value = data_out[index];
                dumb.push_back(value);
            }
        u.push_back(dumb);
    }
    */

    //to return the transpose of the data since the dimension is (lx,0.1lx)
    // I need large data in y coordinate so I transpose
    int c = ny;
    vector<vector<double>> u;
    vector<double> dumb;
    for(int i = 0  ; i < ny ; i++){
        dumb.clear();
        for(int j = 0; j < nx ; j++){
                int index = i + j*c;
                double value = data_out[index];
                dumb.push_back(value);
            }
        u.push_back(dumb);
    }

    
    // close the HDF5 file
    fp.close();
    free(data_out); data_out = NULL;



    return u;
}

