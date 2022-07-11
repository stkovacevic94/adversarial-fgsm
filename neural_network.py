import numpy as np
import activations
import pickle as pk
#Module neural_network.py

class NeuralNetwork:

    def __init__(self, c):
        self.__c = c
        self.__d = len(c)

    @staticmethod
    def load(path):
        with open(path, 'rb') as handle:
            return pk.load(handle)

    def run_epoch(self, X, Y, learning_rate=0.001):
        Theta_Grad = self.__initializeGradTensor(X.shape[1], Y.shape[1])
        Y_prim = self.forward(X)
        
        print("Epoch ", self.__epoch," completed.",
            " Loss: ", activations.crossentropy(Y_prim, Y))
        self.__epoch = self.__epoch + 1

        Delta = (Y_prim - Y)
        Theta_Grad[self.__d+1]=np.dot(
            activations.sigmoid(self.__H[self.__d]).T,
            Delta)
        self.__ThetaTensor[self.__d+1] -= learning_rate*Theta_Grad[self.__d+1]

        for l in range(self.__d, 0, -1):
            Delta = np.multiply(
                np.dot(Delta, self.__ThetaTensor[l+1].T),
                activations.sigmoid(self.__H[l], True))
            Theta_Grad[l] = np.dot(activations.sigmoid(self.__H[l-1]).T, Delta)
            self.__ThetaTensor[l] -= learning_rate*Theta_Grad[l]

    def train(self, X, Y, learning_rate=0.001, epoch=100):
        self.initialize(X.shape[1], Y.shape[1])

        for e in range(epoch):
            self.run_epoch(X, Y, learning_rate)
             
    def __initializeGradTensor(self, input_dim, output_dim):
        GradTensor = [np.zeros((1, 1))] # Dummy matrix for easier interpretation
        GradTensor.append(np.zeros((input_dim, self.__c[0])))
        for l in range(self.__d-1):
            GradTensor.append(
                np.zeros((self.__c[l], self.__c[l+1])))
        GradTensor.append(
            np.zeros((self.__c[self.__d-1], output_dim)))
        return GradTensor

    def run_adversarial(self, x, y_target, num_iter=1, step_size=0.01):
        adversarial_x = x

        for _ in range(num_iter):
            y_prim = self.forward(adversarial_x)
            
            delta = (y_prim - y_target)
            for l in range(self.__d, 0, -1):
                delta = np.multiply(
                    np.dot(delta, self.__ThetaTensor[l+1].T),
                    activations.sigmoid(self.__H[l], True))
            adv_perturbation = np.dot(delta, self.__ThetaTensor[1].T)

            adversarial_x = adversarial_x - step_size*adv_perturbation
        
        return adversarial_x, adv_perturbation

    def run_ifgsm(self, x, y_target, eps=1, step_size=1, momentum=0, num_iter=1):
        r'''Iterative FGSM.
        x: Input image
        y_target: Optional target value
        eps: L_{\infty} norm of the perturbation
        num_iter: Number of iterations
        '''

        clip_min =  x - eps
        clip_max = x + eps

        adversarial_x = x
        g = 0
        for _ in range(num_iter):
            _, gradient = self.run_adversarial(x, y_target)
            grad_1norm = np.linalg.norm(gradient, 1)

            g = momentum*g + (gradient/grad_1norm)

            signed_grad = np.sign(g) * step_size
            adversarial_x = np.clip(adversarial_x - signed_grad, clip_min, clip_max)
        adv_perturbation = adversarial_x - x

        return adversarial_x, adv_perturbation
        
    def initialize(self, input_dim, output_dim):
        self.__ThetaTensor = [np.zeros((1, 1))] # Dummy matrix for easier interpretation
        self.__ThetaTensor.append(
            np.random.normal(scale=0.1, size=(input_dim, self.__c[0])))
        for l in range(self.__d-1):
            self.__ThetaTensor.append(
                np.random.normal(scale=0.1, size=(self.__c[l], self.__c[l+1])))
        self.__ThetaTensor.append(
            np.random.normal(scale=0.1, size=(self.__c[self.__d-1], output_dim)))

        self.__epoch = 1

    def save(self, path):
        with open(path, 'wb+') as handle:
            pk.dump(self, handle, protocol=pk.HIGHEST_PROTOCOL)

    def forward(self, X):
        self.__H = [X]

        H_o = X
        # Compute hidden layers
        for l in range(1, self.__d+1):
            H_i = np.dot(H_o, self.__ThetaTensor[l])
            self.__H.append(H_i)
            H_o = activations.sigmoid(H_i)

        # Compute output Softmax layer
        H_i = np.dot(H_o, self.__ThetaTensor[self.__d + 1])
        self.__H.append(H_i)
        return activations.softmax(H_i)