import numpy as np


class MLPerceptron():
    def __init__(self, layers, activations):
        self.W = self.initializeWeights(layers)
        self.f = self.getActivations(activations)
        self.df = self.getDerivatives(activations)
        self.layers = layers[1:]
        self.L = len(layers) - 1

    def initializeWeights(self, layers):
        W = [np.random.randn(i, j) for i, j in zip(layers[1:], layers[:-1])]
        W = [np.concatenate((np.ones((w.shape[0], 1)), w), axis=1) for w in W]
        return W

    def getActivations(self, activations):
        functions = {
            "s": self.sigma,
            "purelin": self.purelin
        }
        return [functions[a] for a in activations]

    def getDerivatives(self, activations):
        functions = {
            "s": self.dxSigma,
            "linear": self.dxPurelin
        }
        return [functions[a] for a in activations]

    def FeedForward():
        pass

    def Error():
        pass

    def fit(self, x, t, lr, epochs):
        x = np.concatenate((np.ones((x.shape[0],1)), x), axis=1)
        n = []   # Salidas de la preactivacion en cada capa
        a = []   # Salidas de la activacion en cada capa
        s = []   # Sensibilidades en cada capa
        for epoch in range(epochs):
            for xn, tn in zip(x, t):
                # --- FeedForward
                # Para cada capa
                for r in range(self.L):
                    if r == 0:
                        ar = xn
                        a.append(ar)
                    else:
                        ar = np.concatenate((np.ones(1), ar))
                        a.append(ar)
                    print("Ar[{}] :\n{}".format(r, ar))
                    print("Wr[{}] :\n{}".format(r, self.W[r]))
                    nr = np.dot(self.W[r], ar)
                    print('n[{}] = {}'.format(r, nr))
                    ar = self.f[r](nr)
                    print('a[{}] = {}'.format(r, ar))
                    n.append(nr)
                    #a.append(ar)
                    #print("epoch: {}, a={}".format(epoch, a))
                print("Preactivations:\n", n)
                print("Activations:\n", a)
                # Calcular el error
                error = ar - tn
                # --- Backpropagation
                print("====== B A C K P R O P A G A T I O N ======")
                for r in range(self.L-1, -1, -1):
                    F = np.diag(self.df[r](n[r]))
                    print("F[{}] = \n{}".format(r, F))
                    #print('F')
                    #print(F.shape)
                    #print('Error')
                    #print(error.shape)
                    if r == self.L-1:
                        sr = -2*np.matmul(F, error)
                        print('r({}), sr= {}'.format(r, sr))
                    else:
                        #print(F.shape)
                        #print(self.W[r+1][:,1:].T.shape)
                        #print(sr.shape)
                        sr = np.matmul(np.matmul(F, self.W[r+1][:,1:].T), sr)
                        print('r({}), sr= {}'.format(r, sr))
                    s.append(sr)
                s.reverse()
                print('s: ', s)
                # --- Actualizacion de pesos
                for r in range(self.L-1, -1, -1):
                    print('r = ', r)
                    print('w dim = ', self.W[r].shape)
                    print(self.W[r])
                    print('s dim = ', s[r].shape)
                    print(s[r])
                    print('a dim = ', a[r].T.shape)
                    print(a[r].T)
                    self.W[r] = self.W[r] - lr*np.matmul(s[r].reshape((len(s[r]),1)), a[r].reshape((len(a[r]),1)).T)


    def predict(self, x):
        pass


    def sigma(self, x):
        return 1/(1+np.exp(-x))

    def dxSigma(self, x):
        return self.sigma(x) * (1 - self.sigma(x))

    def purelin(self, x):
        return x

    def dxPurelin(self, x):
        return 1


x = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,1,1,0])

net = MLPerceptron([2, 2, 1], ['s', 's'])
print("--- Pesos ---")
print(net.W)
print("-------------")
#print(net.activations[0](0))
net.fit(x, y, 0.01, 1)
