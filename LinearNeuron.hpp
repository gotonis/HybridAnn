#include <cmath>
#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include "Neuron.hpp"

using namespace std;

//Neuron with linear activation function

class LinearNeuron: public Neuron {

public:
  float activation(vector<float> inputs){
    if(inputs.size() < N){
      printf("ERROR: INSUFFICIENT PARAMETERS. \n");
      return 0;
    }
    else{
      float z = bias;
      for(int i=0;i<N;i++){
	z += inputs[i]*weights[i];
      }
      return z;
    }
  }

  float fprime(vector<float> inputs){
    return 1;
  }


};
