#include <cmath>
#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include "Neuron.hpp"

using namespace std;

//Neuron with logistic activation function
class SigNeuron: public Neuron {

public:

  SigNeuron(int nInputs, float lowerBd, float upperBd, float * LR):Neuron(nInputs, lowerBd, upperBd, LR){}
  float activation(vector<float> inputs){
    if(inputs.size() < N){
      printf("ERROR: INSUFFICIENT PARAMETERS.\n");
      return 0;
    }
    else{
      float z = bias;
      for(int i=0;i<N;i++){
	z += inputs[i]*weights[i];
      }
      return 1/(1+exp(-z));
    }
  }

  float fprime(vector<float> inputs){
    float a = activation(inputs);
    return a*(1-a);
  }
  
};
