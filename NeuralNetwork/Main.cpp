#include <stdio.h>
#include <iostream>
#include <string>
#include <vector>
#include "Eigen/Dense"
#include <stdlib.h>
#include <time.h>
#include <Windows.h>
#include <fstream>

using Eigen::MatrixXd;
using namespace std;

MatrixXd sigmoid(MatrixXd x)
{
	MatrixXd expReturn(x.rows(), x.cols());
	for (int i = 0; i < x.rows(); i++)
	{
		for (int j = 0; j < x.cols(); j++)
		{
			expReturn(i, j) = 1 / (1 + exp(-x(i, j)));
		}
	}
	return expReturn;
}

MatrixXd binaryStep(MatrixXd x)
{
	MatrixXd expReturn(x.rows(), x.cols());
	for (int i = 0; i < x.rows(); i++)
	{
		for (int j = 0; j < x.cols(); j++)
		{
			if (x(i, j) < 0.1)
			{
				expReturn(i, j) = 0;
			}
			else if(x(i,j) > 0.9)
			{
				expReturn(i, j) = 1;
			}
			else
			{
				expReturn(i, j) = x(i, j);
			}
		}
	}
	return expReturn;
}

MatrixXd sigmoidDerivative(MatrixXd x)
{
	MatrixXd oneMatrix = MatrixXd::Constant(x.rows(), x.cols(), 1);
	return (x.cwiseProduct(oneMatrix - x));
}

class NeuronLayer
{
public:
	MatrixXd synapticWeights;
	NeuronLayer(int numberOfNeurons, int numberOfInputsPerNeuron) : synapticWeights(numberOfInputsPerNeuron, numberOfNeurons)
	{
		for (int i = 0; i < numberOfInputsPerNeuron; i++)
		{
			for (int j = 0; j < numberOfNeurons; j++)
			{
				synapticWeights(i, j) = (rand() % 10000 - 5000) / 5000.0;
			}
		}
	}
	NeuronLayer()
	{
	}
};

class NeuralNetwork
{
public:
	NeuronLayer layer1;
	NeuronLayer layer2;
	vector<NeuronLayer*> neuronLayersVec;
	NeuralNetwork(NeuronLayer lay1, NeuronLayer lay2)
	{
		layer1 = lay1;
		layer2 = lay2;
		neuronLayersVec.push_back(&layer1);
		neuronLayersVec.push_back(&layer2);
	}

	void train(MatrixXd trainingSetInputs, MatrixXd trainingSetOutputs, int numberOfTrainingIterations)
	{
		for (int i = 0; i < numberOfTrainingIterations; i++)
		{
			for (int j = 0; j < trainingSetInputs.rows(); j++)
			{
				vector<MatrixXd> output = think(trainingSetInputs.row(j));

				MatrixXd layer2error = trainingSetOutputs.row(j) - output[1];
				MatrixXd layer2delta = layer2error.cwiseProduct(sigmoidDerivative(output[1]));

				MatrixXd layer1error = layer2delta * (layer2.synapticWeights.transpose());
				MatrixXd layer1delta = layer1error.cwiseProduct (sigmoidDerivative(output[0]));

				MatrixXd layer1adjustment = trainingSetInputs.row(j).transpose() * (layer1delta);
				MatrixXd layer2adjustment = output[0].transpose() * (layer2delta);


				layer1.synapticWeights += layer1adjustment;
				layer2.synapticWeights += layer2adjustment;
			}
			cout << "\r Iteration: " << i;

			/*vector<MatrixXd> output = think(trainingSetInputs);

			MatrixXd layer2error = trainingSetOutputs - output[1];
			MatrixXd layer2delta = layer2error * (sigmoidDerivative(output[1]));

			MatrixXd layer1error = layer2delta * (layer2.synapticWeights.transpose());
			MatrixXd layer1delta = layer1error * (sigmoidDerivative(output[0]));

			MatrixXd layer1adjustment = trainingSetInputs.transpose() * (layer1delta);
			MatrixXd layer2adjustment = output[0].transpose() * (layer2delta);


			layer1.synapticWeights += layer1adjustment;
			layer2.synapticWeights += layer2adjustment;*/
		}
	}

	vector<MatrixXd> think(MatrixXd inputs)
	{
		MatrixXd outputFromLayer1 = sigmoid(inputs * (layer1.synapticWeights));
		MatrixXd outputFromLayer2 = sigmoid(outputFromLayer1 * (layer2.synapticWeights));
		vector<MatrixXd> out;
		out.push_back(outputFromLayer1);
		out.push_back(outputFromLayer2);
		return out;
	}
};

int main()
{

	srand(GetTickCount());
	vector<string> input;


	std::ifstream infile("balance-scale.data.txt");
	std::string line;
	while (std::getline(infile, line))
	{
		input.push_back(line);
		//cout << line << endl;
	}


	random_shuffle(input.begin(), input.end());

	int trainingSize = input.size() * 0.7;

	MatrixXd trainingSetInputs(trainingSize, 20);
	MatrixXd trainingSetOutputs(trainingSize, 3);

	for (int i = 0; i < trainingSize; i++)
	{
		if (input[i][0] == 'R')
		{
			trainingSetOutputs(i, 0) = 0;
			trainingSetOutputs(i, 1) = 0;
			trainingSetOutputs(i, 2) = 1;
		}
		else if (input[i][0] == 'L')
		{
			trainingSetOutputs(i, 0) = 1;
			trainingSetOutputs(i, 1) = 0;
			trainingSetOutputs(i, 2) = 0;
		}
		else
		{
			trainingSetOutputs(i, 0) = 0;
			trainingSetOutputs(i, 1) = 1;
			trainingSetOutputs(i, 2) = 0;
		}

		int count = 0;
		cout << input[i] << endl;
		for (int j = 0; j < 4; j++)
		{
			int temp = (int)(input[i][j * 2 + 2] - '0');
			for (int k = 0; k < 5; k++)
			{
				if (temp == (k + 1))
				{
					trainingSetInputs(i, count) = 1;
				}
				else
				{
					trainingSetInputs(i, count) = 0;
				}
				count++;
			}
		}
	}
	cout << "Training Set Inputs: \n" << trainingSetInputs << endl << endl;


	NeuronLayer layer1 = NeuronLayer(50, 20);
	NeuronLayer layer2 = NeuronLayer(3, 50);

	NeuralNetwork neural_network = NeuralNetwork(layer1, layer2);

	cout << "Random starting synaptic weights: " << endl;
	cout << layer1.synapticWeights << endl
		<< endl;

	cout << "Beginning Training: " << endl<<endl;
	neural_network.train(trainingSetInputs, trainingSetOutputs, 1000);
	cout << endl;
	MatrixXd testSetInputs(input.size() - trainingSize, 20);
	MatrixXd testExpected(input.size() - trainingSize, 3);

	for (int i = trainingSize; i < input.size(); i++)
	{
		if (input[i][0] == 'R')
		{
			testExpected(i - trainingSize, 0) = 0;
			testExpected(i - trainingSize, 1) = 0;
			testExpected(i - trainingSize, 2) = 1;
		}
		else if (input[i][0] == 'L')
		{
			testExpected(i - trainingSize, 0) = 1;
			testExpected(i - trainingSize, 1) = 0;
			testExpected(i - trainingSize, 2) = 0;
		}
		else
		{
			testExpected(i - trainingSize, 0) = 0;
			testExpected(i - trainingSize, 1) = 1;
			testExpected(i - trainingSize, 2) = 0;
		}

		int count = 0;
		for (int j = 0; j < 4; j++)
		{
			int temp = (int)(input[i][j * 2 + 2] - '0');
			for (int k = 0; k < 5; k++)
			{
				if (temp == (k + 1))
				{
					testSetInputs(i - trainingSize, count) = 1;
				}
				else
				{
					testSetInputs(i - trainingSize, count) = 0;
				}
				count++;
			}
		}
	}

	vector<MatrixXd> testOut = neural_network.think(testSetInputs);

	cout << "Considering new test data -> ?: " << endl;

	testOut[1] = binaryStep(testOut[1]);

	for (int i = 0; i < testOut[1].rows(); i++)
	{
		cout << testExpected.row(i) << " : " << testOut[1].row(i) << endl;
	}

	cout << "Final error: " << endl;

	MatrixXd networkError = testExpected - testOut[1];
	cout << binaryStep(networkError) << endl;

	

	cin.get();
	return 0;
}