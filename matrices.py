import numpy as np
import parameters as params
from scipy.stats import truncnorm

"""Global variables, specified in parameter file."""
N = params.N  # nr of simulated people
C = params.C  # nr of characteristics of these people
M = params.M  # nr of normal microskills
Q = params.Q  # nr of microskills that are part of the IQ test
T = params.T  # time expressed as the nr of occasions
F = params.F  # nr of factors that determine the structure of the microskills


def create_personality_matrix(mean=params.pers['mean'], sd=params.pers['sd'], twin_type=params.pers['twin_type']):
    """
    Matrix that codes the value of person i on characteristic c (cognitive capacity and concentration)
    twin_type (string): default is None
    Return (float64): numpy matrix size N X C
    """
    # TODO: implement twin options

    arrays = [truncnorm.rvs(a=0, b=np.inf, loc=mean, scale=sd, size=N) for i in range(C)]
    personality = np.vstack(arrays)

    return personality


def create_knowledge_matrix():
    # TODO: set factors
    #  - sample cognitive capacity from normal distribution, mu increases as function of m, convolve with sinusoid
    #  - sample other factors from normal distribution with negative values set to 0

    knowledge = np.zeros((M, F))
    return knowledge


def create_test_matrix():
    # TODO: set factors

    test = np.zeros((Q, F))
    return test
