#include <cmath>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <vector>
#include <algorithm>
#include "Neuron.hpp"
#include "SigNeuron.hpp"
#include "LinearNeuron.hpp"
#include "NeuralNet.hpp"

using namespace std;

float raw_in[4][150];
int labels[150];
int nEpochs = 500;

int main(){
  srand(time(NULL)); //initialize RNG

  //Read in the data
  fstream fs;
  fs.open("Iris.csv", fstream::in);
  string temp;
  for(int i=0;i<150;i++){
    fs >> raw_in[0][i];
    fs >> raw_in[1][i];
    fs >> raw_in[2][i];
    fs >> raw_in[3][i];
    fs >> temp;
    if(temp == "Iris-setosa"){
      labels[i] = 1;
    } else if(temp == "Iris-versicolor"){
      labels[i] = 2;
    } else if(temp == "Iris-virginica"){
      labels[i] = 3;
    }
    else {
      cout << "Invalid label: " << temp << endl;
      labels[i] = -1;
      return 0;
    }
  }

  fs.close();
  cout << "Successfully read in data!" << endl;
  
  float max[4] = {0,0,0,0};
  float min[4] = {100,100,100,100};
  for(int j=0;j<4;j++){
    //float max[j] = 0;
    //float min[j] = 100;
    for(int i=0;i<150;i++){
      if(raw_in[j][i] > max[j]){
	max[j] = raw_in[j][i];
      }
      if(raw_in[j][i] < min[j]){
	min[j] = raw_in[j][i];
      }
    }
  }


    //normalize and separate data into training and testing
    
  vector<vector<float> > trainInput = vector<vector<float> >(75, vector<float>(4,0));
  vector<vector<float> > trainLabels = vector<vector<float> >(75, vector<float>(3,0));
  vector<vector<float> > testInput = vector<vector<float> >(75, vector<float>(4,0));
  vector<vector<float> > testLabels = vector<vector<float> >(75, vector<float>(3,0));
  int t1 = 0;
  int t2 = 0;
  for(int i=0;i<150;i++){
    
    // printf("%i\n",i);
    int t = (rand()%2);
    if((t && t1<=74) || t2>74){
      //printf("attempting training %i\n",t1 + 1);
      trainInput[t1][0] = (raw_in[0][i]-min[0])/(max[0]-min[0]);
      trainInput[t1][1] = (raw_in[1][i]-min[1])/(max[1]-min[1]);
      trainInput[t1][2] = (raw_in[2][i]-min[2])/(max[2]-min[2]);
      trainInput[t1][3] = (raw_in[3][i]-min[3])/(max[3]-min[3]);
      trainLabels[t1][0] = (labels[i]==1);
      trainLabels[t1][1] = (labels[i]==2);
      trainLabels[t1][2] = (labels[i]==3);
      t1++;
      //printf("there are %i training samples\n",t1);
    }else{
      //printf("attempting testing %i\n",t2 + 1);
      testInput[t2][0] = (raw_in[0][i]-min[0])/(max[0]-min[0]);
      testInput[t2][1] = (raw_in[1][i]-min[1])/(max[1]-min[1]);
      testInput[t2][2] = (raw_in[2][i]-min[2])/(max[2]-min[2]);
      testInput[t2][3] = (raw_in[3][i]-min[3])/(max[3]-min[3]);
      testLabels[t2][0] = (labels[i]==1);
      testLabels[t2][1] = (labels[i]==2);
      testLabels[t2][2] = (labels[i]==3);
      t2++;
      //printf("there are %i testing samples\n", t2);
    }
  }

  printf("done sorting and normalizing\n");

  //construct neural net
  float LR = 0.7; //learning rate
  printf("set the learning rate\n");
  vector<Neuron*> layer1 = vector<Neuron*>();
  for(int i=0;i<3;i++){
    layer1.push_back(new SigNeuron(4,-1,1,&LR));
    printf("created a neuron and added it to layer 1\n");
  }
  vector<Neuron*> layer2 = vector<Neuron*>();
  for(int i=0;i<3;i++){
    layer2.push_back(new SigNeuron(3,-1,1,&LR));
    printf("created a neuron and added it to layer 2\n");
  }

  vector<vector<Neuron*> > neurons = vector<vector<Neuron*> >();
  neurons.push_back(layer1);
  neurons.push_back(layer2);

  NeuralNet network = NeuralNet(neurons, &LR);
  printf("created a neural net\n");

  vector<int> sIndices = vector<int>(75,0); //shuffleable indices for random orderings of training samples
  for(int i=0;i<75;i++){
    //printf("setting shuffle index %i\n",i);
    sIndices[i] = i;
  }

  vector<int> errors = vector<int>(nEpochs,0);
  fs.open("SigResults2.csv", fstream::out);
 
  for(int n=0;n<nEpochs;n++){
    //printf("starting epoch %i\n",n);
    //random_shuffle(sIndices.begin(), sIndices.end());
    for(int i=0;i<75;i++){
      int j = sIndices[i];
      network.train(trainInput[j],trainLabels[j]);
    }

    for(int i=0;i<75;i++){
      int j = sIndices[i];
      vector<float> y = network.evaluate(testInput[j]);
      int labelMax;
      float outMax = -100000;
      int outMaxI;

      for(int k=0;k<3;k++){
	if(i == 5)
	  printf("%f`,%f\n",testLabels[j][k],y[k]);
	if(testLabels[j][k] > 0)
	  labelMax = k;
	if(y[k] > outMax){
	  outMax = y[k];
	  outMaxI = k;
	}
      }
      if(outMaxI != labelMax){
	errors[n] += 1;
	
      }
    }
    fs << errors[n] << "," << endl;
    printf("epoch %i had %i errors\n",n,errors[n]);
      
  }

  fs.close();

  return 0;
}
