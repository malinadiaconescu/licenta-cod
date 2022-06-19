import random
import numpy

SystemMatrixSize = 10
GivenDomain = 1

def generate_system_matrix():
    """
    :return: the system's matrix
    """
    matrix = numpy.random.uniform( - GivenDomain , GivenDomain , [SystemMatrixSize , SystemMatrixSize])
    return matrix

def generate_system_free_coefficients():
    '''

    :return: the matrix that contains the free
    or independent values
    '''
    free_values = numpy.random.uniform( -GivenDomain , GivenDomain , [SystemMatrixSize , 1])
    return free_values

def generate_actual_solutions(system_matrix , independent_coefficients):
    '''

    :param system_matrix:  the system's matrix
    :param independent_coefficients: the matrix that contains the free
    or independent values
    :return: the real solutions of the system
    '''
    return numpy.linalg.solve(system_matrix , independent_coefficients)

def generate_test():
    '''

    :return: the whole system, including the system's matrix
     and independent coefficients, and the real solution
    '''
    system_matrix = generate_system_matrix()
    independent_coefficients = generate_system_free_coefficients()

    while numpy.linalg.det(system_matrix) == 0:
        system_matrix = generate_system_matrix()
        independent_coefficients = generate_system_free_coefficients()

    actual_solution = generate_actual_solutions(system_matrix,independent_coefficients)
    return (system_matrix , independent_coefficients , actual_solution)


