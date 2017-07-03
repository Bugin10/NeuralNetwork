#include <stdio.h>
#include <iostream>
#include <string>
#include <vector>
#include "Eigen/Dense"
#include <stdlib.h>
#include <time.h>
#include <Windows.h>
#include <fstream>



#include <algorithm>

#include <GL/glew.h>

#include <glfw3.h>
GLFWwindow* window;

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/norm.hpp>

#include <common/shader.cpp>

using namespace glm;
using Eigen::MatrixXd;
using namespace std;



double zoom = 1.0f;
int prevoffset = 0;

struct camera
{
	float x = 0;
	float y = 0;
};

camera cam;

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{

}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{

	(yoffset > 0) ? zoom *= 0.5f : zoom *= 2.0f;
	//glScalef(zoom, zoom, 1);
	//glOrtho(-1.0*zoom, 1.0*zoom, -1.0*zoom, 1.0*zoom, -1.0, 1.0);
	//glRotatef(zoom, 0, 0, 0);
}


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
			else if (x(i, j) > 0.9)
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
	int numNeurons;
	int numInputs;
	NeuronLayer(int numberOfNeurons, int numberOfInputsPerNeuron) : synapticWeights(numberOfInputsPerNeuron, numberOfNeurons)
	{
		numNeurons = numberOfNeurons;
		numInputs = numberOfInputsPerNeuron;
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






class NetworkDisplay
{
public:
	vector<NeuronLayer*> layers;

	GLuint VertexArrayID;
	GLuint programID;
	GLuint vertexbuffer;
	static const GLfloat g_vertex_buffer_data[];
	GLuint billboard_vertex_buffer;
	GLuint particles_position_buffer;

	// constructor
	NetworkDisplay(vector<NeuronLayer*> inlayers)
	{
		layers = inlayers;

		if (!glfwInit())
		{
			fprintf(stderr, "Failed to initialize GLFW\n");
			getchar();
		}
		glfwWindowHint(GLFW_SAMPLES, 4);
		glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
		glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
		glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);


		// Open a window and create its OpenGL context
		window = glfwCreateWindow(1920, 1080, "Network Display", NULL, NULL);
		if (window == NULL) {
			fprintf(stderr, "Failed to open GLFW window. If you have an Intel GPU, they are not 3.3 compatible. Try the 2.1 version of the tutorials.\n");
			getchar();
			glfwTerminate();
		}
		glfwMakeContextCurrent(window);
		
		// Initialize GLEW
		glewExperimental = true; // Needed for core profile
		if (glewInit() != GLEW_OK) {
			fprintf(stderr, "Failed to initialize GLEW\n");
			getchar();
			glfwTerminate();
		}
		// Dark blue background
		glClearColor(0.0f, 0.0f, 0.4f, 0.0f);
		glMatrixMode(GL_PROJECTION);
		glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);

	
		// Clear the screen
		glClear(GL_COLOR_BUFFER_BIT);

		glEnable(GL_LINE_SMOOTH);
		glLineWidth(3.0f);

		glPointSize(15.0f);


		for (int i = 0; i < inlayers[0]->numInputs; i++)
		{
			glBegin(GL_POINTS);
			glVertex2f(-1.0f + 0.5f, (2*1.8f / inlayers[0]->numInputs) *i - 2 * 0.9f);
			glEnd();
		}


		glPointSize(15.0f);
		for (int i = 0; i < inlayers[0]->numNeurons; i++)
		{
			glBegin(GL_POINTS);
			glVertex2f(-1.0f + 1.0f, (2 * 1.8f / inlayers[0]->numNeurons) *i - 2 * 0.9f);
			glEnd();
		}

		for (int i = 0; i < inlayers[1]->numNeurons; i++)
		{
			glBegin(GL_POINTS);
			glVertex2f(-1.0f + 1.5f, (2 * 1.8f / (inlayers[1]->numNeurons -1)) *i - 2 * 0.9f);
			glEnd();
		}




		glfwSetKeyCallback(window, key_callback);
		glfwSetScrollCallback(window, scroll_callback);
		// Swap buffers
		glfwSwapBuffers(window);

	}

	void displayNetwork()
	{

		

		// Clear the screen
		glClear(GL_COLOR_BUFFER_BIT);


		for (int i = 0; i < layers[0]->numInputs; i++)
		{
			glColor3f(1.0f, 1.0f, 1.0f);
			glBegin(GL_POINTS);
			glVertex2f(-1.0f + 0.5f, (2 * 1.8f / (layers[0]->numInputs - 1)) *i - 2 * 0.9f);
			glEnd();
		}

		for (int i = 0; i < layers[0]->numNeurons; i++)
		{
			glColor3f(1.0f, 1.0f, 1.0f);
			glBegin(GL_POINTS);
			glVertex2f(-1.0f + 1.0f, (2 * 1.8f / (layers[0]->numNeurons -1)) *i - 2 * 0.9f);
			glEnd();
			for (int j = 0; j < layers[0]->numInputs; j++)
			{
				if (layers[0]->synapticWeights(j, i) < -0.1f)
				{
					glColor3f(abs(layers[0]->synapticWeights(j, i)), 0.0f,0.0f);
					glBegin(GL_LINES);
					glVertex2f(-1.0f + 1.0f, (2 * 1.8f / (layers[0]->numNeurons - 1)) *i - 2 * 0.9f);
					glVertex2f(-0.5f, (2 * 1.8f / (layers[0]->numInputs - 1)) *j - 2 * 0.9f);
					glEnd();
				}
				else if (layers[0]->synapticWeights(j, i) > 0.1f)
				{
					glColor3f(0.0f, abs(layers[0]->synapticWeights(j, i)), 0.0f);
					glBegin(GL_LINES);
					glVertex2f(-1.0f + 1.0f, (2 * 1.8f / (layers[0]->numNeurons - 1)) *i - 2 * 0.9f);
					glVertex2f(-0.5f, (2 * 1.8f / (layers[0]->numInputs - 1)) *j - 2 * 0.9f);
					glEnd();
				}
				
				
			}
		}

		for (int i = 0; i < layers[1]->numNeurons; i++)
		{
			glColor3f(1.0f, 1.0f, 1.0f);
			glBegin(GL_POINTS);
			glVertex2f(-1.0f + 1.5f, (2 * 1.8f / (layers[1]->numNeurons - 1)) *i - 2 * 0.9f);
			glEnd();
			for (int j = 0; j < layers[1]->numInputs; j++)
			{
				if (layers[1]->synapticWeights(j, i) < -0.1f)
				{
					glColor3f(abs(layers[1]->synapticWeights(j, i)), 0.0f, 0.0f);
					glBegin(GL_LINES);
					glVertex2f(-1.0f + 1.5f, (2 * 1.8f / (layers[1]->numNeurons - 1)) *i - 2 * 0.9f);
					glVertex2f(0.0f, (2 * 1.8f / (layers[1]->numInputs - 1)) *j - 2 * 0.9f);
					glEnd();
				}
				else if (layers[1]->synapticWeights(j, i) > 0.1f)
				{
					glColor3f(0.0f, abs(layers[1]->synapticWeights(j, i)), 0.0f);
					glBegin(GL_LINES);
					glVertex2f(-1.0f + 1.5f, (2 * 1.8f / (layers[1]->numNeurons - 1)) *i - 2 * 0.9f);
					glVertex2f(0.0f, (2 * 1.8f / (layers[1]->numInputs - 1)) *j - 2 * 0.9f);
					glEnd();
				}
				else
				{
					glColor3f(0.0f, layers[1]->synapticWeights(j, i) , 0.0f);
				}
				
			}
		}




		if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
			//glTranslatef(0.002f, 0, 0);
			cam.x -= 0.002*zoom;
		if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
			//glTranslatef(-0.002f, 0, 0);
			cam.x += 0.002*zoom;
		if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
			//glTranslatef(0, -0.002f, 0);
			cam.y += 0.002*zoom;
		if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
			//glTranslatef(0, 0.002f, 0);
			cam.y -= 0.002*zoom;


		if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
			exit(0);
		
		glLoadIdentity();
		glOrtho(0.5f * (cam.x - 1.0f*zoom), 0.5f * (cam.x + 1.0f*zoom), cam.y - 1.0f*zoom, cam.y + 1.0f*zoom, -1.0, 1.0);
		

		
		// Swap buffers
		glfwSwapBuffers(window);
		glfwPollEvents();
	}

};


















class NeuralNetwork
{
public:
	NeuronLayer layer1;
	NeuronLayer layer2;
	vector<NeuronLayer*> layers;
	NetworkDisplay *networkDisplay;
	NeuralNetwork(NeuronLayer lay1, NeuronLayer lay2)
	{
		layer1 = lay1;
		layer2 = lay2;
		layers.push_back(&layer1);
		layers.push_back(&layer2);
		networkDisplay = new NetworkDisplay(layers);
	}

	void train(MatrixXd trainingSetInputs, MatrixXd trainingSetOutputs, int numberOfTrainingIterations)
	{
		float iterationspersecond = 0.0f;
		float prevtime = GetTickCount();
		MatrixXd errorMatrix = MatrixXd::Constant(1, trainingSetOutputs.cols(), 0);
		for (int i = 0; i < numberOfTrainingIterations; i++)
		{
			float errors = 0;
			MatrixXd errorMatrix = MatrixXd::Constant(1, trainingSetOutputs.cols(), 0);
			for (int j = 0; j < trainingSetInputs.rows(); j++)
			{
				vector<MatrixXd> output = think(trainingSetInputs.row(j));

				MatrixXd layer2error = trainingSetOutputs.row(j) - output[1];
				MatrixXd layer2delta = layer2error.cwiseProduct(sigmoidDerivative(output[1]));

				MatrixXd layer1error = layer2delta * (layer2.synapticWeights.transpose());
				MatrixXd layer1delta = layer1error.cwiseProduct(sigmoidDerivative(output[0]));

				MatrixXd layer1adjustment = 0.01f * trainingSetInputs.row(j).transpose() * (layer1delta);
				MatrixXd layer2adjustment = 0.01f * output[0].transpose() * (layer2delta);


				layer1.synapticWeights += layer1adjustment;
				layer2.synapticWeights += layer2adjustment;


				if (binaryStep(layer2error) != errorMatrix)
				{
					errors++;
				}
				

			}
			if(networkDisplay)
				networkDisplay->displayNetwork();

			if (i%1000 == 0)
			{
				float deltatime = (GetTickCount() - prevtime);
				iterationspersecond = (1000.0f / deltatime) * 1000 ;
				prevtime = GetTickCount();
				cout << "\r Iteration: " << i << " IPS: " << iterationspersecond << " Error Rate: " << errors / (float)trainingSetInputs.rows();
			}
			

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

	int trainingSize = (int)(input.size() * 0.7);

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



	NeuronLayer layer1 = NeuronLayer(5, 20);
	NeuronLayer layer2 = NeuronLayer(3, 5);

	NeuralNetwork neural_network = NeuralNetwork(layer1, layer2);

	cout << "Random starting synaptic weights: " << endl;
	cout << layer1.synapticWeights << endl
		<< endl;

	cout << "Beginning Training: " << endl << endl;
	//for (int i = 0; i < 100000; i++)
	//{
	neural_network.train(trainingSetInputs, trainingSetOutputs, 25000);
	//}
	
	cout << endl;

	vector<MatrixXd> testOut = neural_network.think(testSetInputs);

	cout << "Considering new test data -> ?: " << endl;

	testOut[1] = binaryStep(testOut[1]);

	for (int i = 0; i < testOut[1].rows(); i++)
	{
		cout << testExpected.row(i) << " : " << testOut[1].row(i) << endl;
	}

	cout << "Final error: " << endl;

	MatrixXd networkError = testExpected - testOut[1];
	networkError = binaryStep(networkError);
	cout << networkError << endl;


	while (true)
	{
		neural_network.networkDisplay->displayNetwork();
	}
	


	cin.get();
	return 0;
}