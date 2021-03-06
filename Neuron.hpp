#pragma once
#include <cmath>
#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <vector>

using namespace std;

//abstract class for a neuron in a neural network
class Neuron {

  //Standard neuron with weights and a bias all between lowerBd and upperBd inclusive
public:
  Neuron(int nInputs, float lowerBd, float upperBd ,float * LR){
    N = nInputs;
    learningRate = LR;
    weights = vector<float>();
    float t;
    for(int i=0;i<nInputs;i++){
      t = rand();
      //printf("%f\n",t);
      weights.push_back(t*(upperBd-lowerBd)/((float)RAND_MAX) + lowerBd);
    }
    t = rand();
    bias = t*(upperBd-lowerBd)/RAND_MAX + lowerBd;

  }
  
  virtual float activation(vector<float> inputs) = 0;
  float * learningRate;
  virtual void update(vector<float> inputs, float dcda){
    if(inputs.size() < N){
      printf("ERROR: INSUFFICIENT PARAMETERS.\n");
      return;
    }
    //Standard Backpropegation
    float fpz = fprime(inputs); //Calculates derivative for the activation function
    for(int i=0;i<N;i++){
      weights[i] -= fpz*dcda*inputs[i]*(*learningRate);
    }
    bias -= fpz*dcda*(*learningRate);
  }

  
  virtual vector<float> getPrevDCDAs(vector<float> inputs, float dcda){
    //printf("getting DCDAs\n");
    float fpz = fprime(inputs);
    //printf("%i\n",N);
    vector<float> dcdas = vector<float>(N,fpz*dcda); 
    for(int i=0;i<N;i++){
      dcdas[i]*=weights[i];
      //printf("DCDA %i = %f\n",i,dcdas[i]);
    }

    return dcdas;
  }
  
  /*
    Output layer: 
    dCost/dActivation = 2(a-y)
    dActivation/dWeight = activation'(Z)*[previous activation]
    dActivation/dBias = activation'(Z)
    Z = sum(w_i*activation_i)+b
    
    Nonfinal layer: (neuron k of L_n)
    dCost/dActivation = sum for next layer of w_{jk}*(dCost/dActivation)_j*activation_j'(Z_j)
  */

  virtual float fprime(vector<float> inputs) = 0;

protected:
  vector<float> weights;
  float bias;
  int N;

};
