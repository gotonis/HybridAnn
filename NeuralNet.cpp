#include <cmath>
#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include "Neuron.hpp"
#include "NeuralNet.hpp"

using namespace std;

vector<float> NeuralNet::evaluate(vector<float> input){
  vector<float> t1 = input;
  vector<float> t2;
  vector<Neuron*> current;
  for(int i=0;i<layers.size();i++){
    t2 = vector<float>();
    current = layers[i];
    for(int j=0;j<sizes[i];j++){
      t2.push_back(current[j]->activation(t1));
    }
    t1 = t2;
  }
  return t2;
}

void NeuralNet::train(vector<float> input, vector<float> output){
  vector<vector<float> > temp = vector<vector<float> >();
  temp.push_back(input);
  vector<Neuron*> current;
  vector<float> t1;
  vector<float> t2;
  for(int i=0;i<layers.size();i++){
    t1 = temp[i];
    t2 = vector<float>();
    current = layers[i];
    for(int j=0;j<sizes[i];j++){
      t2.push_back(current[j]->activation(t1));
    }
    temp.push_back(t2);
    //printf("evaluated layer %i\n",i);
  }

  vector<float> costs = vector<float>();
  for(int i=0;i<output.size();i++){
    costs.push_back(t2[i]-output[i]);
  }

  //printf("calculated costs\n");
  vector<float> temp3;
  vector<float> temp4;
  for(int j=layers.size()-1;j>=0;j--){
    //printf("costs for layer %i are ", j);
    /*
    for(int s=0;s<costs.size();s++){
      printf("%f ",costs[s]);
    }
    printf("\n");
    */
    current = layers[j];
    temp3 = vector<float>(temp[j].size(),0);
    for(int i=0;i<current.size();i++){
      //printf("adjusting neuron %i of layer %i\n", i, j);
      temp4 = current[i]->getPrevDCDAs(temp[j],costs[i]);
      //printf("got DCDAs\n");
      for(int k=0;k<temp3.size();k++){
	//printf("%i, %u, %u\n",k, temp3.size(),temp4.size());
	temp3[k]+=temp4[k];
	//printf("%i\n",k);
      }
      current[i]->update(temp[j],costs[i]);
    }
    costs = temp3;
    
  }
}
