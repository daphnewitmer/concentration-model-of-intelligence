import numpy as np
import parameters as params
from scipy.stats import truncnorm
import scipy.signal as signal
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
    Matrix that codes the value of person i on characteristic c (cognitive capacity and concentration) (N x C)
    """
    # TODO: implement twin options

    personality = truncnorm.rvs(a=0, b=np.inf, loc=mean, scale=sd, size=(N, C))

    helper.plot_distribution(personality, None, 'Charachteristics of People', 'People (N)', 'Cognitive Capacity and Concentration (C)')
    return personality


def create_knowledge_matrix(sd):
    """
    Matrix that codes how each factor f loads onto each microskill m (M x F)
    """
    # TODO: determine mean range (based on max capcity in personality?) + convolve cog_cap with sinusoid + truncated normal or set negative values to 0?

    mean = np.linspace(0, 2.5, M)
    # time = np.linspace(0, 100, 1500)
    # sinusoid = np.sin(time)
    # helper.plot_distribution(time, sinusoid, '', 'time', 'sin')
    # conv_mean = np.convolve(mean, sinusoid, 'same')
    # helper.plot_distribution(conv_mean)

    cog_cap = [np.random.normal(loc=u, scale=sd, size=1) for u in mean]
    other = truncnorm.rvs(a=0, b=np.inf, loc=0, scale=sd, size=(M, F-1))
    knowledge = np.hstack((cog_cap, other))

    # helper.plot_distribution(knowledge[:, 1], None, 'Factor 1 needed for Microskill(M)', 'Microskill(M)', 'Factor 1')

    return knowledge


def create_test_matrix(knowledge_matrix):
    """
    Matrix that codes how each factor f loads onto each microskill m (Q x F)
    """
    # TODO: determine how to calculate peak width for find_peaks_cwt() function + check permutation

    cog_cap = knowledge_matrix[:, 0]

    # Select microskills on last 5 peaks and last 5 valleys of cognitive capacity
    peak_skills = signal.find_peaks_cwt(cog_cap,  np.arange(1, 5))[-5:]
    inv_data = cog_cap * (-1)
    valley_skills = signal.find_peaks_cwt(inv_data,  np.arange(1, 5))[-5:]
    peak_valley_skills = np.concatenate((peak_skills, valley_skills))

    # Permutate factor values
    factors_without_cog_cap = knowledge_matrix[peak_valley_skills, 1:]
    factors_permuted = np.random.permutation(factors_without_cog_cap)
    ten_skills_perm = np.hstack((cog_cap[peak_valley_skills, np.newaxis], factors_permuted))

    # Copy items with randomly distributed noise
    rest_skills_without_noise = np.tile(ten_skills_perm, [9, 1])
    noise = np.random.normal(0, .1, rest_skills_without_noise.shape)
    rest_skills_with_noise = rest_skills_without_noise + noise

    test_matrix = np.vstack((ten_skills_perm, rest_skills_with_noise))

    # helper.plot_distribution(knowledge_matrix[:, 0])
    # plt.plot(cog_cap)
    # plt.plot(peak_skills, cog_cap[peak_skills], 'ro')
    # plt.show()

    return test_matrix

def create_schooling_matrix(first_period, second_period, third_period):
    """
    Matrix that codes which microskill m is offered at which time step t. This matrix is created anew for every person i (T x M)
    """

    total_years = 25
    time_steps_per_year = int(T / total_years)
    schooling_matrix = np.zeros((T, M))
    microskills = []

    for i in range(total_years):

        if i < first_period:
            perc_rand = float(0.75)
            replace = True
        elif i >= first_period and i < second_period or i >= third_period:
            perc_rand = float(0.5)
            replace = True
        elif i >= second_period and i < third_period:
            perc_rand = float(0.5)
            replace = False

        random = np.random.choice(M,
                                  size=int(time_steps_per_year * perc_rand),
                                  replace=True)  # Sample from all microskills

        fitting_for_period = np.random.choice(np.arange(i * 25, (i * 25 + 25)),
                                              size=int(time_steps_per_year * (1 - perc_rand)),
                                              replace=replace)  # Sample from skills associated with age

        microskills.extend(np.append(random, fitting_for_period))

    for t, m in enumerate(microskills):
        schooling_matrix[t, m] = 1

    return schooling_matrix