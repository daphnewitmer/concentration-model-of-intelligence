import numpy as np
from src import parameters as params, matrices
from scipy.stats import truncnorm
import matplotlib.pyplot as plt

"""Global variables, specified in parameter file."""
N = params.nrOfSimulatedPeople
C = params.nrOfCharacteristics
M = params.nrOfMicroskillsNormal
Q = params.nrOfMicroskillsIQ
T = params.nrOfTestOccasions
F = params.nrOfFactors

TOTAL_YEARS = params.TOTAL_YEARS_OF_SIMULATION

np.random.seed(42)

class Simulation:
    # TODO: add test achievements + learning matrix

    def __init__(self):
        self.personality_matrix = matrices.create_personality_matrix(params.PERS_MEAN, params.PERS_SD, params.PERS_TWIN)
        self.knowledge_matrix = matrices.create_knowledge_matrix(self.personality_matrix, params.KNOW_SD)
        self.test_matrix = matrices.create_test_matrix(self.knowledge_matrix)
        self.microskill_similarity_matrix = self.knowledge_matrix.dot(self.knowledge_matrix.T)
        self.achievement_matrix = np.zeros((N, M), dtype=bool)
        self.learning_matrix = np.zeros((T + Q, N), dtype=bool)

    def run(self):
        """ Create schooling matrix for every person and update achievement and learning matrix for every person """

        for person in range(1):  # N
            schooling_array = matrices.create_schooling_array(params.FIRST_PERIOD, params.SECOND_PERIOD,
                                                              params.THIRD_PERIOD, params.SKILLS_TO_SAMPLE_FROM_PER_AGE)

            self.update(person, schooling_array, params.TEST_AGE)

        return self.achievement_matrix, self.learning_matrix

    def update(self, person, schooling_array, test_age):
        """ Update schooling array for person n for every timestep t """

        cog_cap = self.get_cog_cap(person)
        test_timestep = int((T / TOTAL_YEARS) * test_age)

        for timestep in range(T):
            microskill = schooling_array[timestep]

            if self.is_learned(person, microskill, timestep, cog_cap, test=False):
                self.achievement_matrix[person, microskill] = True  # not timestep but microskill
                self.learning_matrix[timestep, person] = True

            if timestep == test_timestep:
                self.take_test(person, timestep, cog_cap)

        return

    def take_test(self, person, overall_timestep, cog_cap):
        """ Take IQ test """

        for microskill in range(Q):
            test_timestep = T + microskill
            if self.is_learned(person, microskill, overall_timestep, cog_cap, test=True):
                self.learning_matrix[test_timestep, person] = True


    def is_learned(self, n, m, t, cog_cap, test):
        """ Check whether person n was able to learn the microskill m """

        if test:
            req_cog_cap = self.test_matrix[m, 0]
            concentration = self.personality_matrix[n, 1]
        else:
            req_cog_cap = self.knowledge_matrix[m, 0]
            concentration = self.get_concentration(n)

        acquired_know = self.get_acquired_knowledge(n, m)
        cog_cap = cog_cap[t]

        total_req_cog_cap = req_cog_cap - acquired_know
        avail_cog_cap = cog_cap * concentration

        if not self.achievement_matrix[n, m] and total_req_cog_cap < avail_cog_cap:
            return True
        elif self.achievement_matrix[n, m] is True:
            #TODO: can a microskill be unlearned?
            return True

        return False

    def get_cog_cap(self, n):
        """ Get the cognitive capacity of person n at time t (cog_cap changes as a function of age) """

        x_max_cog_cap = (T / TOTAL_YEARS) * params.PEAK_YEAR_COG_CAP
        y_max_cog_cap = self.personality_matrix[n, 0]
        x = np.arange(T)
        start_perc_cog_cap = 0.2

        a = (-y_max_cog_cap) / np.power(x_max_cog_cap, 2)
        y = ((a * np.power((x - x_max_cog_cap), 2)) + y_max_cog_cap) * (1 - start_perc_cog_cap) + start_perc_cog_cap * y_max_cog_cap

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

        acquired_microskills = np.argwhere(self.achievement_matrix[n, :] > 0)

        if len(acquired_microskills) == 0:
            return 0

        return sum(self.microskill_similarity_matrix[m, acquired_microskills])
