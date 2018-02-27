#include <cmath>
#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include "Neuron.hpp"

using namespace std;

class NeuralNet {
public:

  NeuralNet(vector<vector<Neuron*> > neurons, float * LR){
    layers = neurons;
    sizes = vector<int>();
    for(int i=0;i<layers.size();i++){
      sizes.push_back(layers[i].size());
    }
    learningRate = LR;
  }
  
  //Runs an input vector through the network
  vector<float> evaluate(vector<float> input);

  //Does a single backprop step on an input and its associated expected output
  void train(vector<float> input, vector<float> output);
  
protected:
  float * learningRate;
  vector<vector<Neuron*> > layers;
  vector<int> sizes;

};
