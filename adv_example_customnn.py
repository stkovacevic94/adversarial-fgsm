from tensorflow.examples.tutorials.mnist import input_data
from neural_network import NeuralNetwork
import numpy as np
import matplotlib.pyplot as plt

def run_report_gd_adversarial(model, x, target, num_iter, step_size):
    adversarial_image, _ = model.run_adversarial(x, target, num_iter=num_iter, step_size=step_size)
    adv_perturbation = adversarial_image - x
    gradient_logits = k.forward(adversarial_image)
    print("GRADIENT DESCENT ADVERSARIAL ATTACK")
    print("\tLogits:", gradient_logits, " Class:", np.argmax(gradient_logits))

    print("\tPerturbation norm:", np.linalg.norm(adv_perturbation, 2))
    print("\tMax pixel in abs(perturbation):", adv_perturbation[0, np.argmax(np.absolute(adv_perturbation))])
    print("\tAvg pixel in abs(perturbation):", np.average(np.absolute(adv_perturbation)))
    print("")

    return adversarial_image, adv_perturbation

def run_report_ifgsm(model, x, target, num_iter, step_size, eps, momentum):
    adversarial_image, adv_perturbation = model.run_ifgsm(
        x, 
        target, 
        num_iter=num_iter, 
        step_size=step_size, 
        eps=eps, 
        momentum=momentum)

    gradient_logits = k.forward(adversarial_image)
    print("ITERATIVE FAST GRADIENT SIGN METHOD")
    print("\tGradient descent logits:", gradient_logits, " Class:", np.argmax(gradient_logits))

    print("\tPerturbation norm:", np.linalg.norm(adv_perturbation, 2))
    print("\tPerturbation \infty norm:", np.max(adv_perturbation))
    print("\tMax pixel in abs(perturbation):", adv_perturbation[0, np.argmax(np.absolute(adv_perturbation))])
    print("\tAvg pixel in abs(perturbation):", np.average(np.absolute(adv_perturbation)))
    print()
    
    return adversarial_image, adv_perturbation

# Load data and model
mnist_data = input_data.read_data_sets("MNIST_data", one_hot=True)

path = "MODEL_DATA/mnist_custom/model.pkl"
k = NeuralNetwork.load(path)
target = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0] # Digit 1

img = np.reshape(mnist_data.test.images[1], (1, 28*28))

# Calclate original logits
orig_logits = k.forward(img)
print("Original logits:", orig_logits, " Class:", np.argmax(orig_logits))

gd_adv_img, gd_adv_pert = run_report_gd_adversarial(k, img, target, num_iter=7, step_size=0.1)
ifgsm_adv_img, ifgsm_adv_pert = run_report_ifgsm(k, img, target, num_iter=7, eps=0.24, step_size=0.1, momentum=0.2)

plt.subplot(321)
plt.imshow(np.reshape(img, (28, 28)), cmap="gray")
plt.subplot(323)
plt.imshow(np.reshape(gd_adv_pert, (28, 28)), cmap="gray")
plt.subplot(325)
plt.imshow(np.reshape(gd_adv_img, (28, 28)), cmap="gray")
plt.subplot(322)
plt.imshow(np.reshape(img, (28, 28)), cmap="gray")
plt.subplot(324)
plt.imshow(np.reshape(ifgsm_adv_pert, (28, 28)), cmap="gray")
plt.subplot(326)
plt.imshow(np.reshape(ifgsm_adv_img, (28, 28)), cmap="gray")
plt.show()