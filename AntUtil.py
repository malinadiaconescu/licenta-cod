import os

import generator
import random
import  numpy as np
import copy
import matplotlib.pyplot as plt
import time
import csv

def save_to_random_file(fileName, systemDimension):
    if os.path.exists("results/" + fileName + str(systemDimension) + '.png'):
        plt.savefig("results/" + fileName + str(systemDimension) + '_{}.png'.format(int(time.time())))
    else:
        plt.savefig("results/" + fileName + str(systemDimension) + '.png')

def save_to_file_real_predicted_data(realData, predictedData):

    with open('TestBank.csv', 'a+', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Real Solutions: " + str([np.asarray(realData).flatten()])])
        writer.writerow(["Predicted Solutions: " + str([np.asarray(predictedData).flatten()])])
        writer.writerow("")
        f.close()


def pseudorandom_values(domain):
    """

    :param domain:
    :return:
    """
    return random.uniform(-domain, domain)


def fitnes_function(temporary_solution, indepedent_values, system_matrix):
    """

    :param temporary_solution:
    :param indepedent_values:
    :param system_matrix:
    :return:
    """
    sum = 0
    for j in range(0, len(system_matrix)):
        auxiliar_sum = 0
        for i in range(0, len(indepedent_values)):
            auxiliar_sum += temporary_solution[i] * system_matrix[j][i]
        sum += abs(indepedent_values[j] - auxiliar_sum)
    return sum


def ant_colony_optimization_solution():
    """

    :return:
    """
    (system_matrix , independent_values , actual_solutions) = generator.generate_test()
    alpha = [3.0] * len(actual_solutions)
    beta = 0.8
    (best_current_solution , best_current_pheromone) = generate_starting_solution(system_matrix,independent_values,alpha[0])

    list_fitness_values=[]
    list_iterations=[]
    for iteration in range(0,1000):
        auxiliar_solution = copy.copy(best_current_solution)

        for contor in range(0,len(auxiliar_solution)):
            auxiliar_value = pseudorandom_values(alpha[contor])
            auxiliar_solution[contor] = auxiliar_solution[contor] + auxiliar_value
            auxiliar_pheromone = fitnes_function(auxiliar_solution,independent_values,system_matrix)

            if auxiliar_pheromone < best_current_pheromone:
                best_current_pheromone = auxiliar_pheromone
                best_current_solution = copy.copy(auxiliar_solution)
                print("fitness function: "+str(best_current_pheromone))
                alpha[contor] = alpha[contor] * beta

            auxiliar_solution[contor] = auxiliar_solution[contor] - auxiliar_value
        list_fitness_values.append(best_current_pheromone)
        list_iterations.append(iteration)

    plt.plot(list_iterations,list_fitness_values)
    plt.xlabel('iteration')
    plt.ylabel('fitness function value')

    save_to_random_file("FitnessFunctionIterationsGraphs",len(best_current_solution) )
    f=open("systemtested.txt","a")
    f.write(str(system_matrix))
    f.write(str(independent_values))
    return (best_current_solution, actual_solutions)


def generate_starting_solution(system_matrix, independent_values, alpha_value):
    """

    :param system_matrix:
    :param independent_values:
    :param alpha_value:
    :return:
    """
    best_solution = [0] * len(independent_values)
    best_pheromone = fitnes_function( best_solution, independent_values, system_matrix)
    for starting_solution in range(0, 1000):
        auxiliar_sol = np.round(np.random.uniform(-alpha_value, alpha_value, len(system_matrix[0])), 18)
        auxiliar_pheromone = fitnes_function(auxiliar_sol,independent_values,system_matrix)
        if best_pheromone > auxiliar_pheromone:
            best_solution = copy.copy(auxiliar_sol)
            best_pheromone = auxiliar_pheromone

    return (best_solution, best_pheromone)


if __name__ == "__main__":
    (best_solution, actual_solutions) =ant_colony_optimization_solution()

    x = []

    for i in range (len(best_solution)):
        x.append(abs(best_solution-actual_solutions))
    #x = np.array(x)
    #x = x.flatten()
    #plt.hist(x)
    print(best_solution)
    print("")
    print(actual_solutions)
    #save_to_random_file("HistogramAbsoluteRealPredictedValues",len(best_solution))
    #save_to_file_real_predicted_data(actual_solutions,best_solution)
