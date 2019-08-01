import numpy as np 
from tqdm import tqdm


class NeuralNetwork():

  def __init__(self):

    #seeding random number generation
    np.random.seed(1)

    #defining matrix of weights
    self.weights = 2 * np.random.random((6,1)) - 1

  def sigmoid(self, x):
    #apply the sigmoid function
    return 1/(1 + np.exp(-x))
  
  def sigmoid_derivative(self, x):
    #derivative of the sigmoid function
    return x * (1-x)
  
  def train(self, training_inputs, training_outputs, training_iterations):

    #train the model to make accurate predictions while iterating weights
    for i in tqdm(range(training_iterations)):
      output = self.think(training_inputs)

      error = training_outputs - output

      adjustments = np.dot(training_inputs.T, error * self.sigmoid_derivative(output))

      self.weights += adjustments
  
  def think(self, inputs):
    #passes the inputs via the neuron to generate the outputs
    inputs = inputs.astype(float)
    output = self.sigmoid(np.dot(inputs, self.weights))
    return output

if __name__ == '__main__':

  #initialise the neuron class
  neural_network = NeuralNetwork()

  #generate and show initial weights
  print('Generating random weights')
  print(neural_network.weights)

  #importing inputs & outputs
  training_inputs = np.array([
    [0,0,1,0,0,0],
    [1,1,1,0,0,0],
    [1,0,1,0,0,0],
    [0,1,1,0,0,0],[1,1,0,0,0,0],[0,1,0,0,0,0],[0,0,0,0,1,0],[0,0,0,1,0,1],[0,1,1,1,1,1]])
  
  training_outputs = np.array([[0,1,1,0,1,0,0,0,0]]).T

  #training
  neural_network.train(training_inputs,training_outputs,200000)

  print('Final weights after training: ')
  print(neural_network.weights)

  np.random.seed()
  randomarray = np.random.randint(0,1+1,(6,1)).T

  print('Building random array:')
  print(randomarray)

  print('Considering...')
  print('New output: ')
  print(neural_network.think(randomarray))
