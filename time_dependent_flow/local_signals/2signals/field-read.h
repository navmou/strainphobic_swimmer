#include <vector>
#include <iostream>
#include <fstream>

using namespace std;


vector<double> read_file(string path){
    vector<double> U;
    double x;
    ifstream f;
    f.open(path);
    if (!f) {
        cerr << "Unable to open file datafile.txt";
        exit(1);   // call system to stop
    }
    while (f >> x) {
      U.push_back(x);
    }
    f.close();
    return U;
}

vector<vector<double>> read_field(string path){
    vector<double> U = read_file(path);

    int c = 10000;
    vector<vector<double>> u;
    vector<double> dumb;
    for(int i = 0  ; i < 400 ; i++){
        dumb.clear();
        for(int j = 0; j < 10000 ; j++){
                int index = i*c + j;
                double value = U[index];
                dumb.push_back(value);
            }
        u.push_back(dumb);
    }
    return u;
}



