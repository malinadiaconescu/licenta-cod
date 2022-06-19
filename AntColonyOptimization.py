import generator
import random
import  numpy as np
import copy


def pseudorandom_values(domain):
    """

    :param domain: the domain in which we generate the pseudorandom values
    :return: a pseudorandom value
    """
    return random.uniform(-domain, domain)


def fitness_function(temporary_solution, indepedent_values, system_matrix):
    """

    :param temporary_solution: the temporary solution for which
     it is calculated the fitness value
    :param indepedent_values: the matrix B presented, that contains
    the independet values, the ones after the = sign
    :param system_matrix: system's matrix, the one that contains
     the dependent values
    :return: the fitness value of the respective temporary
    solution
    """
    sum = 0
    for j in range(0, len(system_matrix)):
        auxiliar_sum = 0
        for i in range(0, len(indepedent_values)):
            # each equation of the systems is calculated, in order to
            # see how far it is from the respective independent value
            auxiliar_sum += temporary_solution[i] * system_matrix[j][i]
        sum += abs(indepedent_values[j] - auxiliar_sum)
    return sum


def ant_colony_optimization_solution():
    """
    :return: the best solution found for the system
    """
    # we take a generated test from the generator pack
    (system_matrix , independent_values , actual_solutions) = generator.generate_test()
    # acknowledged values for fitnessing the solutions
    alpha = [3.0] * len(actual_solutions)
    beta = 0.8
    # taking the starting solution
    (best_current_solution , best_current_pheromone) = \
        generate_starting_solution(system_matrix,independent_values,alpha[0])

    # stopping condition
    for iteration in range(0,1000):
        # it is formed an auxiliar solution starting with the values from
        # the best solution at the current iteration
        # it is used an auxiliar one in order to not loose the best one
        auxiliar_solution = copy.copy(best_current_solution)

        for contor in range(0,len(auxiliar_solution)):
            # transform the auxiliar solution into a new one,
            # in order to form a completely new solution
            # pseudorandom values are random values in the given domain
            auxiliar_value = pseudorandom_values(alpha[contor])
            auxiliar_solution[contor] = auxiliar_solution[contor] + auxiliar_value
            auxiliar_pheromone = fitness_function(auxiliar_solution,independent_values,system_matrix)

            # it is checked whether the newly formed solution is better than the best solution
            # if yes, the best solution is changed
            if auxiliar_pheromone < best_current_pheromone:
                best_current_pheromone = auxiliar_pheromone
                best_current_solution = copy.copy(auxiliar_solution)
                print("fitness function: "+str(best_current_pheromone))
                # there exists a certain domain for each solution, which is changed
                # wether we found a better solution, so the searching is narrowed
                alpha[contor] = alpha[contor] * beta

            auxiliar_solution[contor] = auxiliar_solution[contor] - auxiliar_value

    return best_current_solution


def generate_starting_solution(system_matrix, independent_values, alpha_value):
    """

    :param system_matrix: system's matrix,
    the one that contains the dependent values
    :param independent_values: the matrix B presented,
    that contains the independet values, the ones after the = sign
    :param alpha_value: the domain of search
    :return: a tuple containing the best solution
    found and the respetive pheromone
    """
    best_solution = [0] * len(independent_values)
    best_pheromone = fitness_function( best_solution, independent_values, system_matrix)
    for starting_solution in range(0, 1000):
        auxiliar_sol = np.round(
            np.random.uniform(-alpha_value, alpha_value, len(system_matrix[0])), 18)
        auxiliar_pheromone = fitness_function(
            auxiliar_sol,independent_values,system_matrix)
        if best_pheromone > auxiliar_pheromone:
            best_solution = copy.copy(auxiliar_sol)
            best_pheromone = auxiliar_pheromone

    return (best_solution, best_pheromone)


if __name__ == "__main__":
    ant_colony_optimization_solution()