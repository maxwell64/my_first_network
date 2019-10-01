import numpy as np

testInput = np.array([1,0,0])
trainingInput = np.array([
[1,1,1],
[0,1,0],
[1,1,0],
[0,0,1]])

trainingOutput = np.array([[1,0,1,0]]).T

class Perceptron():

    def __init__(self):

        #sets the seed for the initial weights
        np.random.seed(1)

        #sets the initial weights
        self.weights = 2 * np.random.random((3,1)) - 1

    def sigmoid(self, x):

        return 1 / (1 + np.exp(-x))

    def sigmoidDerivative(self, x):

        return x * (1 - x)

    def train(self, trainingInput, trainingOutput, trainingIterations):

        for i in range(trainingIterations):

            output = self.think(trainingInput)

            error = trainingOutput - output

            adjustments = np.dot(trainingInput.T, error * self.sigmoidDerivative(output))

            self.weights += adjustments

    def think(self, input):

        input = input.astype(float)

        output = self.sigmoid(np.dot(input, self.weights))

        return output

perceptron = Perceptron()

print("Initial weights are:")
print(perceptron.weights)

perceptron.train(trainingInput, trainingOutput, 20000)

print("Final weights are:")
print(perceptron.weights)

np.random.seed()
randomArray = np.random.randint(0, 1+1,(1,3))

print("Building random array:")
print(randomArray)

print("Output is:")
print(perceptron.think(randomArray))
