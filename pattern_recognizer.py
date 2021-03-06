from numpy import exp, random, array, dot

class neural_network:
    def __init__(self):
        random.seed(1)
        self.weights = 2 * random.random((3, 1)) - 1

    def train(self, inputs, outputs, num):
        for iteration in range(num):
            output = self.think(inputs)
            error = outputs - output
            adjustment = dot(inputs.T, error * output*(1-output))
            self.weights += adjustment

    def think(self, inputs):
        return self.__sigmoid(dot(inputs, self.weights))

    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

network = neural_network()
inputs = array([[1, 1, 1], [1, 0, 1], [0, 1, 1]])
outputs = array([[1, 1, 0]]).T
network.train(inputs, outputs, 100000)

print(network.think(array([1, 0, 0])))
