import numpy as np
import parameters as params
from scipy.stats import truncnorm
import scipy.signal as signal
# import tests
import matplotlib.pyplot as plt

"""Global variables, specified in parameter file."""
N = params.nrOfSimulatedPeople
C = params.nrOfCharacteristics
M = params.nrOfMicroskillsNormal
Q = params.nrOfMicroskillsIQ
T = params.nrOfTestOccasions
F = params.nrOfFactors

TOTAL_YEARS = params.TOTAL_YEARS_OF_SIMULATION

# np.random.seed(42)
# TODO: move numbers to parameters file + structure?

def create_personality_matrix(mean, sd, twin_type):
    """
    Matrix that codes the value of person i on characteristic c (cognitive capacity and concentration) (N x C)
    """
    # TODO: implement twin options

    personality = truncnorm.rvs(a=0, b=np.inf, loc=mean, scale=sd, size=(N, C))

    # tests.plot_distribution(personality[:,0], None, 'Charachteristics of People', 'People (N)', 'Cognitive Capacity and Concentration (C)')
    return personality


def create_knowledge_matrix(personality_matrix, sd):
    """
    Matrix that codes how each factor f loads onto each microskill m (M x F)
    """

    max_cog_cap_personality = np.max(personality_matrix[:, 0])

    mean = np.linspace(0, max_cog_cap_personality, M)
    time = np.linspace(0, M, M)
    sinusoid = np.sin(time / (M / TOTAL_YEARS) * (2 * np.pi))
    conv_mean = mean + 0.5 * np.multiply(mean, sinusoid)

    cog_cap = [truncnorm.rvs(a=0, b=np.inf, loc=u, scale=sd, size=1) for u in conv_mean]
    other = truncnorm.rvs(a=0, b=np.inf, loc=0, scale=sd, size=(M, F-1))
    knowledge = np.hstack((cog_cap, other))

    # tests.plot_distribution(knowledge[:, 0], None, 'Factor 1 needed for Microskill(M)', 'Microskill(M)', 'Factor 1')

    return knowledge


def create_test_matrix(knowledge_matrix):
    """
    Matrix that codes how each factor f loads onto each microskill m (Q x F)
    """
    # TODO: don't use find_peaks_cwt() function, but calculate with pi

    import pandas as pd
    cog_cap = knowledge_matrix[:, 0]

    # Select microskills on last 5 peaks and last 5 valleys of cognitive capacity
    peak_skills = signal.find_peaks(cog_cap, height=2.5, threshold=0.3, distance=4)[0][-5:]
    inv_data = cog_cap * (-1)
    valley_skills = signal.find_peaks_cwt(inv_data,  np.arange(1, 5))[-5:]
    peak_valley_skills = np.concatenate((peak_skills, valley_skills))

    # Permutate factor values except cog_cap
    factors_without_cog_cap = knowledge_matrix[peak_valley_skills, 1:]
    factors_permuted = cog_cap[peak_valley_skills, np.newaxis]

    for column in factors_without_cog_cap.T:
        column_permuted = np.random.permutation(column)
        factors_permuted = np.hstack((factors_permuted, np.expand_dims(column_permuted, axis=1)))

    # Copy items with randomly distributed noise
    rest_skills_without_noise = np.tile(factors_permuted, [9, 1])  # repeat ten skills another 9 times
    noise = np.random.normal(0, .1, rest_skills_without_noise.shape)
    rest_skills_with_noise = rest_skills_without_noise + noise

    test_matrix = np.vstack((factors_permuted, rest_skills_with_noise))

    # plt.plot(cog_cap)
    # plt.plot(peak_skills, cog_cap[peak_skills], 'ro')
    # plt.show()
    # exit()

    return test_matrix


def create_schooling_array(first_period, second_period, third_period, skill_sample_age):
    """
    List that codes which microskill m is offered at which time step t. This matrix is created anew for every person i (M)
    """

    time_steps_per_year = int(T / TOTAL_YEARS)
    schooling_array = []

    for i in range(TOTAL_YEARS):

        if i < first_period:
            perc_rand = float(0.75)
            replace = True
        elif i >= first_period and i < second_period or i >= third_period:
            perc_rand = float(0.5)
            replace = True
        elif i >= second_period and i < third_period:
            perc_rand = float(0.5)
            replace = False

        random = np.random.choice(np.arange(M),
                                  size=int(time_steps_per_year * perc_rand),
                                  replace=True)  # Sample from all microskills

        fitting_for_period = np.random.choice(np.arange(i * skill_sample_age, (i * skill_sample_age + skill_sample_age), dtype=int),
                                              size=int(time_steps_per_year * (1 - perc_rand)),
                                              replace=replace)  # Sample from skills associated with age

        schooling_array.extend(np.append(random, fitting_for_period))

    # tests.plot_distribution(schooling_array, None, 'Presented Microskill at Timestep t', 'Timestep(T)', 'Microskill(M)')

    return schooling_array


class AchievementMatrix:
    # TODO: check dot.product interpretation (why not use correlation?) + finish parabola equation + add test achievements

    def __init__(self, personality_matrix, knowledge_matrix):
        self.achievement_matrix = np.zeros((N, M), dtype=bool)
        self.learning_matrix = np.zeros((Q, N), dtype=bool)
        self.personality_matrix = personality_matrix
        self.knowledge_matrix = knowledge_matrix
        self.microskill_similarity_matrix = self.knowledge_matrix.dot(self.knowledge_matrix.T)

    def update(self, person, schooling_array):
        """ Update schooling array for person n for every timestep t """

        cog_cap = self.get_cog_cap(person)
        nr_Learned_skills_per_timestep = []

        for timestep in range(T):
            microskill = schooling_array[timestep]

            if self.is_learned(person, microskill, timestep, cog_cap):
                self.achievement_matrix[person, timestep] = True

            nr_Learned_skills_per_timestep.append(self.achievement_matrix[person].sum())

        return nr_Learned_skills_per_timestep

    def is_learned(self, n, m, t, cog_cap):
        """ Check whether person n was able to learn the microskill m """

        req_cog_cap = self.knowledge_matrix[m, 0]
        concentration = self.get_concentration(n)
        acquired_know = self.get_acquired_knowledge(n, m)
        cog_cap = cog_cap[t]

        total_req_cog_cap = req_cog_cap - acquired_know
        avail_cog_cap = cog_cap * concentration

        if self.achievement_matrix[n, m] is False and (total_req_cog_cap < avail_cog_cap):
            return True
        elif self.achievement_matrix[n, m] is True:
            #TODO: can a microskill be unlearned?
            return True

        return False

    def get_cog_cap(self, n):
        """ Get the cognitive capacity of person n at time t (cog_cap changes as a function of age) """

        max_cog_cap = self.personality_matrix[n, 0]
        x = np.arange(T)
        a = max_cog_cap / (np.power(450, 2) - 450000)
        b = -1000 * a
        y = (a * np.power(x, 2)) + (b * x)

        plt.plot(x, y)
        plt.title('Max cog cap: ' + str(max_cog_cap) + ' Max y :' + str(np.max(y)))
        plt.show()
        exit()

        return y

    def get_concentration(self, n):
        """ Get concentration of person n, random noise is added """

        max_concentration = self.personality_matrix[n, 1]
        rand_noise = truncnorm.rvs(a=0, b=np.inf, loc=0, scale=0.5)
        concentration = float(max_concentration - rand_noise)

        concentration = 0 if concentration < 0 else concentration

        return concentration

    def get_acquired_knowledge(self, n, m):
        """ Get sum of already acquired microskill similar to microskil m """

        # test_matrix = np.random.rand(5, 3)
        # print(test_matrix)
        # # print(test_matrix.T)
        # # print(test_matrix.dot(test_matrix.T))
        # print(np.corrcoef(test_matrix))  # negative correlation?
        # exit()
        acquired_microskills = np.argwhere(self.achievement_matrix[n, :] > 0)

        return sum(self.microskill_similarity_matrix[m, acquired_microskills])
