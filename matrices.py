import numpy as np
import parameters as params
import random
from scipy.stats import truncnorm
from scipy.signal import find_peaks_cwt
import helper
import matplotlib.pyplot as plt

"""Global variables, specified in parameter file."""
N = params.nrOfSimulatedPeople
C = params.nrOfCharacteristics
M = params.nrOfMicroskillsNormal
Q = params.nrOfMicroskillsIQ
T = params.nrOfTestOccasions
F = params.nrOfFactors

np.random.seed(42)

def create_personality_matrix(mean, sd, twin_type):
    """
    Matrix that codes the value of person i on characteristic c (cognitive capacity and concentration) (M x C)
    """
    # TODO: implement twin options

    arrays = [truncnorm.rvs(a=0, b=np.inf, loc=mean, scale=sd, size=C) for i in range(N)]
    personality = np.vstack(arrays)

    helper.plot_distribution(personality, None, 'Cognitive Capacity of People', 'People (N)', 'Cognitive Capacity per Charachteristic')
    return personality


def create_knowledge_matrix(sd):
    """
    Matrix that codes how each factor f loads onto each microskill m (M x F)
    """
    # TODO: determine mean range (based on max capcity in personality?) + convolve cog_cap with sinusoid + truncated normal or set negative values to 0?

    mean = np.arange(0, 2.5, 0.0025)
    # time = np.arange(0, 100, 0.1)
    # sinusoid = np.sin(time)
    # conv_mean = signal.convolve(mean, sinusoid, 'same')

    cog_cap = [np.random.normal(loc=u, scale=sd, size=1) for u in mean]
    other = [truncnorm.rvs(a=0, b=np.inf, loc=0, scale=sd, size=F-1) for i in range(M)]
    knowledge = np.vstack(other)
    knowledge = np.hstack((cog_cap, knowledge))

    # helper.plot_distribution(knowledge[:, 0], None, 'Cognitive Capacity needed for Microskill(M)', 'Microskill(M)', 'Mean for Cognitive Capacity Sample')

    return knowledge


def create_test_matrix(knowledge_matrix):
    """
    Matrix that codes how each factor f loads onto each microskill m (Q x F)
    """
    # TODO: determine how to calculate peak width for find_peaks_cwt() function + copy microskills and add noise

    cog_cap = knowledge_matrix[:, 0]

    # Select microskills on last 5 peaks and last 5 valleys of cognitive capacity
    peak_skills = find_peaks_cwt(cog_cap,  np.arange(1, 5))[-5:]
    inv_data = cog_cap * (-1)
    valley_skills = find_peaks_cwt(inv_data,  np.arange(1, 5))[-5:]
    peak_valley_skills = np.concatenate((peak_skills, valley_skills))

    # Permutate factor values
    test_matrix_ten_items = knowledge_matrix[peak_valley_skills, 1:]
    permuted  = np.random.permutation(test_matrix_ten_items)
    test_matrix_ten_items_perm = np.hstack((cog_cap[peak_valley_skills, np.newaxis], permuted))

    # Copy items with randomly distributed noise

    # helper.plot_distribution(knowledge_matrix[:, 0])
    # plt.plot(cog_cap)
    # plt.plot(valley_skills, cog_cap[valley_skills], 'ro')
    # plt.show()


    return None