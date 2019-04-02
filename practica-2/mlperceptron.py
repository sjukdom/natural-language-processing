import numpy as np
from tqdm import tqdm

class MLPerceptron():
    def __init__(self, layers, activations):
        self.W = [np.random.randn(i, j)*np.sqrt(1/(i+j)) for i, j in zip(layers[1:], layers[:-1])]       # Lista de pesos por capa
        #self.b = [np.random.randn(n) for n in layers[1:]]                                                # Lista de bias por capa
        self.b = [np.random.zeros(n) for n in layers[1:]]  
        self.f = self.getActivations(activations)                                                        # Funciones de activacion
        self.df = self.getDerivatives(activations)                                                       # Derivada de activaciones
        self.layers = layers[1:]                                                                         # Neuronas por capa
        self.L = len(layers) - 1                                                                         # Numero de capas de la red
        self.n = [0]*(len(layers)-1)                                                                     # Lista de preactivaciones
        self.a = [0]*len(layers)                                                                         # Lista de activaciones

    def getActivations(self, activations):
        functions = {
            "sigmoid": self.sigma,
            "purelin": self.purelin,
            "tanh": self.tanh
        }
        return [functions[a] for a in activations]

    def getDerivatives(self, activations):
        functions = {
            "sigmoid": self.dxSigma,
            "purelin": self.dxPurelin,
            "tanh": self.dxTanh
        }
        return [functions[a] for a in activations]

    def FeedForward(self, x):
        '''
            Calcular la salida de la red neuronal para el vector x
        '''
        ar = x
        self.a[0] = x
        #print('-----FEEDFORWARD-----')
        for r in range(self.L):
            #print('< {}, {} > = {}'.format(self.W[r], ar, np.dot(self.W[r], ar)))
            nr = np.dot(self.W[r], ar) + self.b[r]
            #print('{} + {} = {}'.format(np.dot(self.W[r], ar), self.b[r], nr))
            #print('n[{}] = {}'.format(r, nr))
            ar = self.f[r](nr)
            #print('a[{}] = {}'.format(r, ar))
            self.n[r] = nr
            self.a[r+1] = ar
        return ar

    def Error(self, t, y):
        return t - y

    def Backpropagation(self, error):
        '''
            Retropropagar el error hacia las capas anteriores.
            Se modifica la lista de sensibilidades
        '''
        #print("---BACKPROPAGATION---")
        # Lista de sensibilidades por capa
        s = [0]*self.L
        for r in range(self.L-1, -1, -1):
            F = np.diag(self.df[r](self.n[r].reshape(-1)))
            if r == self.L-1:
                #print('F =>\n', F)
                #print('error => ', error)
                sr = -2*np.dot(F, error)
            else:
                #print('F =>\n', F)
                #print('W =>\n', self.W[r+1].T)
                #print('S =>\n', sr)
                sr = np.dot(F, np.dot(self.W[r+1].T, sr))
            #print('s[r={}] = {}'.format(r, sr))
            s[r] = sr
        return s

    def UpdateWeights(self, s, lr):
        '''
            Actualizar los pesos y los bias
        '''
        #print("---UPDATING THE WEIGHTS---")
        for r in range(self.L-1, -1, -1):
            dW = np.dot(s[r].reshape(-1,1), self.a[r].reshape(-1,1).T)
            dB = s[r]
            #print("Gradiente:")
            #print("Delta W[{}] =>\n{}".format(r, dW))
            #print("Delta b[{}] =>\n{}".format(r, dB))
            #print("Pesos a actualziar")
            #print("W[{}] =>\n{}".format(r, self.W[r]))
            #print("b[{}] =>\n{}".format(r, self.b[r]))
            self.W[r] -= lr*dW
            self.b[r] -= lr*dB
            #print("Nuevos pesos")
            #print("W[{}] =>\n{}".format(r, self.W[r]))
            #print("b[{}] =>\n{}".format(r, self.b[r]))

    def fit(self, x, t, lr, epochs):
        errors = []
        #data = [(xn, tn) for xn, tn in zip(x, t)]
        for i in tqdm(range(epochs)):
            #np.random.shuffle(data)
            for xn, tn in zip(x, t):
            #for xn, tn in data:
                #print('### DATO[{}] ### EPOCH[{}]'.format(xn, i))
                prediction = self.FeedForward(xn)
                error = self.Error(tn, prediction)
                delta = self.Backpropagation(error)
                self.UpdateWeights(delta, lr)
            errors.append(error)
        return errors

    def predict(self, x):
        y = [0]*len(x)
        for i, xn in enumerate(x):
            y[i] = self.FeedForward(xn)[0]
        return y

    def sigma(self, x):
        return 1.0/(1.0+np.exp(-x))

    def dxSigma(self, x):
        return self.sigma(x) * (1.0 - self.sigma(x))

    def purelin(self, x):
        return x

    def dxPurelin(self, x):
        return np.array([1.0])

    def tanh(self, x):
        return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

    def dxTanh(self, x):
        return 1 - np.square(self.tanh(x))

x = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,1,1,0])

#net = MLPerceptron([2, 2, 1], ['sigmoid', 'purelin'])
#net = MLPerceptron([2, 2, 1], ['sigmoid', 'sigmoid'])
net = MLPerceptron([2, 2, 1], ['tanh', 'purelin'])
print("--- Pesos Iniciales ---")
print(net.W)
print("")
print(net.b)
print("-------------")

### Realice un ejercicio a mano con estos datos para comprobar, es correcto el
### procedimiento pero no converge a la solucion
# w1 = np.array([[0.5, 0.5], [0.5, 0.5]])
# w2 = np.array([[1.,1.]])
# b1 = np.array([1.,1.])
# b2 = np.array([1.])
# W = [w1, w2]
# b = [b1, b2]
# net.W[0] = W[0]
# net.W[1] = W[1]
# net.b[0] = b[0]
# net.b[1] = b[1]
#
# print("--- Pesos 2---")
# print(net.W)
# print("--- Bias ---")
# print(net.b)
# print("-------------")
###

errs = net.fit(x, y, 0.05, 1000)
preds = net.predict(x)

import matplotlib.pyplot as plt
plt.figure()
plt.title('Error')
plt.plot(np.arange(len(errs)), errs)
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.grid(True)
plt.show()

print("Prediccion => ", preds)
print("Suma error cuadratico medio => ", (np.sum([(target-real)**2 for target, real in zip(y, preds)])/len(y)))

print("--- Pesos Finales ---")
print(net.W)
print("")
print(net.b)
print("-------------")

'''
INICIALIZACION DE PESOS:
    https://medium.com/usf-msds/deep-learning-best-practices-1-weight-initialization-14e5c0295b94
    https://hackernoon.com/how-to-initialize-weights-in-a-neural-net-so-it-performs-well-3e9302d4490f
'''
