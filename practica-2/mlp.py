'''
    Implementacion de una red neuronal multicapa
'''

import numpy as np

class MLPerceptron():
    def __init__(self, layers):
        self.W = [np.random.randn(i, j) for i, j in zip(layers[1:], layers[:-1])]
        self.layers = layers[1:]
        self.L = len(layers) - 1

    def fit(self, x, y, lr, epochs):
        # Agregar el bias al vector de pesos y de entrada
        x = np.concatenate((np.ones((x.shape[0],1)), x), axis=1)
        self.W = [np.concatenate((np.ones((w.shape[0], 1)), w), axis=1) for w in self.W]
        z = [np.zeros(n) for n in self.layers]       # Preactivacion de la neurina j en la capa r
        y_h = [np.zeros(n) for n in self.layers]     # Activacion de la neurona j en la capa r
        ynj = x[0]                                   # Condicion inicial
        for epoch in range(epochs):
            for n in range(len(x)):                  # Para cada dato
                # ------------- Feed Forward
                for r in range(self.L):              # Para cada capa
                    for j in range(len(self.W[r])):  # Para cada neurona
                            z_njr = np.sum([theta*y for theta, y in zip(list(self.W[r]), list(ynj))]) # Calcular preactivacion
                            y_njr = self.sigma(z_njr)
                            z[r][j] = z_njr
                            y_h[r][j] = y_njr
                # ------------- Backpropagation
                # Calcular la sensibilidad de la ultima capa
                for j in range(self.layers[-1]):
                   d_njr1 = (y_h[self.L-1][j]) * self.dxSigma(z[self.L-1][j])
                   print(d_njr1)



    def sigma(self, x):
        return 1/(1+np.exp(-x))

    def dxSigma(self, x):
        return self.sigma(x) * (1 - self.sigma(x))

x = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,1,1,0])

net = MLPerceptron([2, 2, 1])
net.fit(np.array([[1,2],[3,4]]), 1, 1, 1)
