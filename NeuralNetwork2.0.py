import numpy
import random
from Activation_fct import *
"""
    Classe servant à crée des Neural Networks
    Elle sera au début hardcoder en grande partie
    mais deviendra de plus en plus modulables
"""

class NeuralNetwork:

    #Constructeur
    def __init__(self,nb_input,nb_hidden_nodes,nb_output,activation_fct,learning_rate):
        self.activation_fct = activation_fct
        self.learning_rate = learning_rate
        nb_nodes = [nb_input] + nb_hidden_nodes
        nb_nodes.append(nb_output)

        #Création des matrix pour weights and biases
        self.list_matrix_w,self.list_matrix_b = self.cree_matrix(nb_nodes)

    #Fait un guess
    def guess(self,features):
        features = numpy.asmatrix(features)

        self.calculated_sum = [features]

        #Calcul de la weighted sum
        for i in range(0,len(self.list_matrix_w)):
            weighted_sum = numpy.dot(self.list_matrix_w[i],self.calculated_sum[i])
            weighted_sum = numpy.add(weighted_sum,self.list_matrix_b[i])
            weighted_sum = self.activation_fct[i](weighted_sum)
            self.calculated_sum.append(weighted_sum)

        return self.calculated_sum[len(self.calculated_sum) - 1]

    def train(self,features,label):
        features = numpy.asmatrix(features)
        labels = numpy.asmatrix(label)

        guess = self.guess(features)
        error_array = []

        for i in range(len(self.list_matrix_w) - 1,-1,-1):

            compteur_reversed = len(self.list_matrix_w) -1 - i

            #Calcul de l'erreur
            if compteur_reversed == 0:
                erreur = numpy.subtract(label,guess)
            else:
                erreur = numpy.transpose(self.list_matrix_w[i + 1])
                erreur = numpy.dot(erreur,error_array[-1])
            error_array.append(erreur)

            #Calcul du gradient
            gradient = numpy.asmatrix(self.activation_fct[i](numpy.asarray(self.calculated_sum[i + 1]),True))
            gradient = numpy.multiply(gradient,erreur)
            gradient = numpy.multiply(gradient,self.learning_rate)

            #Calcul des deltas
            if compteur_reversed == len(self.list_matrix_w) -1:
                delta_weights = numpy.transpose(numpy.asarray(features))
            else:
                delta_weights = numpy.transpose(numpy.asarray(self.calculated_sum[i]))

            delta_weights = numpy.multiply(gradient,delta_weights)

            #Application des deltas
            self.list_matrix_w[i] = numpy.add(self.list_matrix_w[i],delta_weights)
            self.list_matrix_b[i] = numpy.add(self.list_matrix_b[i],gradient)

    #Prend les nodes en parmamètres et crée des matrix appropriés
    def cree_matrix(self,node_list):

        tableau_matrix = []
        tableau_bias = []

        for i in range(0,len(node_list) - 1):
            tableau_matrix.append(numpy.asmatrix(numpy.random.uniform(-1,1,(node_list[i + 1],node_list[i]))))
            tableau_bias.append(numpy.asmatrix(numpy.random.uniform(-1,1,(node_list[i + 1],1))))
        return (tableau_matrix,tableau_bias)

nn = NeuralNetwork(2,[16,2],1,[sigmoid,sigmoid,softmax],0.1)

total_features = [
    [
        [[1],[0]],
        [[1]]
    ],

    [
        [[1],[1]],
        [[0]]
    ],

    [
        [[0],[0]],
        [[0]]
    ],

    [
        [[0],[1]],
        [[1]]
    ]



]



for i in range(0,17000):
    element = random.choice(total_features)
    nn.train(element[0],element[1])


print(nn.guess([[0],[1]])) ##1
print(nn.guess([[1],[0]])) ## 1
print(nn.guess([[0],[0]])) ## 0
print(nn.guess([[1],[1]])) ## 0


