import numpy

"""
    Fichier contenant tout les activations functions
    utilisables par la classe NeuralNetwork
"""

def sigmoid(x,derivative=False):
    if not derivative:
        return 1 / (1 + numpy.exp(-x)) #Retourne x simgoided
    return x * (1 - x) #Retourne le derivative


def softmax(x,derivative=False):
    if not derivative:
        e_x = numpy.exp(x - numpy.max(x))
        return e_x / e_x.sum(axis=0) #Retourne x softmaxed
    return x * x.sum(axis=0) #Retourne le derivative
