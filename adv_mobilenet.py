import tensorflow as tf
import tensorflow_hub as hub
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import ast

IMAGE_PATH = "adv_example_snail.jpg"
GRAD_VAL_NUM_ITER = 5

def get_labels():
    with open('imagenet_labels.txt', 'r') as f:
        s = f.read()
        return ast.literal_eval(s)

def get_input_image_data(path):
    img = cv2.imread(IMAGE_PATH)
    img = cv2.resize(img, (299 , 299))
    np_image_data = cv2.normalize(np.float32(np.asarray(img)), None, 0, 1, cv2.NORM_MINMAX)
    tensor_data = np.expand_dims(np_image_data,axis=0)
    return tensor_data

def one_hot_target(target_index):
    onehot = np.zeros((1, 1001))
    onehot[np.arange(1), np.array([target_index+1])] = 1
    return onehot

def gradient(input_tensor, logits_tensor, target):
    xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits_tensor, labels=target)

    return tf.gradients(xs=[input_tensor], ys=xentropy)

def get_gradient_value_perturbation(sess, input_tensor, logits_tensor, image_data, target, num_iter=1, step_size=0.01):
    adv_step = tf.constant(step_size)
    
    # Adversarial setup    
    grad_image = gradient(input_tensor, logits_tensor, target)
    adv_image = input_tensor - adv_step*grad_image

    adversary_x = image_data
    for _ in range(num_iter):
        result = sess.run(adv_image, feed_dict={input_tensor: adversary_x})
        adversary_x = np.reshape(result, (1, 299, 299, 3))
    return adversary_x

def get_fgsm_with_momentum_perturbation(sess, input_tensor, logits_tensor, image_data, target, num_iter=1, step_size=0.01, eps=0.01, momentum=0):
    adv_step = tf.constant(step_size)
    mu = tf.constant(momentum)
    clip_min = tf.constant(image_data - eps)
    clip_max = tf.constant(image_data + eps) 
    

    # Adversarial setup    
    g_ = tf.placeholder(tf.float32)

    grad_image = gradient(input_tensor, logits_tensor, target)

    g = mu * g_ + grad_image / tf.norm(grad_image, ord=1)

    signed_g = tf.sign(g)

    adv_image = tf.clip_by_value(tf.reshape(input_tensor - adv_step*signed_g, shape=(1, 299, 299, 3)), clip_min, clip_max)
  
    adversary_x = image_data
    previous_g = 0
    for _ in range(num_iter):
        [result, previous_g] = sess.run([adv_image, g], feed_dict={input_tensor: adversary_x, g_:previous_g})
        adversary_x = np.reshape(result, (1, 299, 299, 3))

    return adversary_x

def printReport(adversary_x, x, title, class_str, probability):
    adversary_perturbation = np.reshape(x - adversary_x, (299*299*3))
    print(title)
    print("\tClassified:", class_str, " Probability:", probability)

    print("\tPerturbation L2 norm:", np.linalg.norm(adversary_perturbation, 2))
    print("\tPerturbation \infty norm:", np.max(adversary_perturbation))
    print("\tAvg pixel in abs(perturbation):", np.average(np.absolute(adversary_perturbation)))
    print()

with tf.Graph().as_default():
    # Load computational graph
    module_url = "https://tfhub.dev/google/imagenet/inception_v3/classification/1"
    model = hub.Module(module_url)

    # Connect input and output tensors
    image_placeholder = tf.placeholder(
        tf.float32, shape=(None, 299, 299, 3), name='input_image')
    logits = model(image_placeholder)
    y_prim = tf.nn.softmax(logits, name="y_prim") # Softmax layer used for reports

    labels = get_labels()
    image_data = get_input_image_data(IMAGE_PATH)
    target = one_hot_target(21) # kite
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())

        # Original logits
        results = sess.run(y_prim, feed_dict={image_placeholder: image_data})
        idx = np.argmax(results)
        print("Original Class: ", labels[idx-1], " Probability:", results[0, idx])

        # Gradient value attack
        adversary_x = get_gradient_value_perturbation(sess, image_placeholder, logits, image_data, target, num_iter=GRAD_VAL_NUM_ITER, step_size=0.1)
        results = sess.run(y_prim, feed_dict={image_placeholder: adversary_x})
        idx = np.argmax(results)
        printReport(adversary_x, image_data, "GRADIENT VALUE ADVERSARIAL ATTACK", labels[idx-1], results[0, idx])

        # FGSM with momentum attack
        adversary_x_fgsm = get_fgsm_with_momentum_perturbation(sess,
            image_placeholder,
            logits,
            image_data,
            target,
            num_iter=GRAD_VAL_NUM_ITER,
            step_size=0.05,
            eps=0.016,
            momentum=0.8)
        results = sess.run(y_prim, feed_dict={image_placeholder: adversary_x_fgsm})
        idx = np.argmax(results)
        printReport(adversary_x_fgsm, image_data, "FGSM MOMENTUM ADVERSARIAL ATTACK", labels[idx-1], results[0, idx])

        #Plot everything
        plt.subplot(321)
        plt.imshow(np.reshape(image_data, (299, 299, 3)))
        plt.subplot(323)
        plt.imshow(np.reshape(adversary_x - image_data, (299, 299, 3)))
        plt.subplot(325)
        plt.imshow(np.reshape(adversary_x, (299, 299, 3)))
        plt.subplot(322)
        plt.imshow(np.reshape(image_data, (299, 299, 3)))
        plt.subplot(324)
        plt.imshow(np.reshape(adversary_x_fgsm - image_data, (299, 299, 3)))
        plt.subplot(326)
        plt.imshow(np.reshape(adversary_x_fgsm, (299, 299, 3)))
        plt.show()

