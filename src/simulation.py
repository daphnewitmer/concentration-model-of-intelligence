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
COG_CAP_INDEX = params.COG_CAP_INDEX
CONC_INDEX = params.CONC_INDEX

np.random.seed(42)


class Simulation:
    # TODO: speed (remove loops) + parameter tests (child smaller variation than adult)

    def __init__(self):
        self.personality_matrix = matrices.create_personality_matrix(params.PERS_TWIN)
        self.knowledge_matrix = matrices.create_knowledge_matrix(self.personality_matrix)
        self.test_matrix = matrices.create_test_matrix(self.knowledge_matrix)
        self.microskill_similarity_matrix = self.knowledge_matrix.dot(self.knowledge_matrix.T)
        self.achievement_matrix = np.zeros((N, M), dtype=bool)
        self.learning_matrix = np.zeros((T + Q, N), dtype=bool)
        self.concentration_matrix = np.zeros((M, N))
        self.cog_cap_matrix = np.zeros((M, N))

    def run(self):
        """ Create schooling matrix for every person and update achievement and learning matrix for every person """

        for person in range(params.nrOfPersInTest):  # N
            schooling_array = matrices.create_schooling_array()

            self.update(person, schooling_array)

        return self.achievement_matrix, self.learning_matrix

    def update(self, person: int, schooling_array: list):
        """ Update schooling array for person n for every timestep t """

        cog_cap = self.get_cog_cap(person)
        self.cog_cap_matrix[:, person] = cog_cap
        test_timestep = int((T / TOTAL_YEARS) * params.TEST_AGE)

        for timestep in range(T):
            microskill = schooling_array[timestep]

            if self.is_learned(person, microskill, timestep, cog_cap, test=False):
                self.achievement_matrix[person, microskill] = True  # not timestep but microskill
                self.learning_matrix[timestep, person] = True

            if timestep == test_timestep:
                self.take_test(person, timestep, cog_cap)

        return

    def take_test(self, person: int, overall_timestep: int, cog_cap: np.ndarray):
        """ Take IQ test """

        for microskill in range(Q):
            test_timestep = T + microskill
            if self.is_learned(person, microskill, overall_timestep, cog_cap, test=True):
                self.learning_matrix[test_timestep, person] = True

    def is_learned(self, person: int, microskill: int, timepoint: int, cog_cap: np.ndarray, test: bool):
        """ Check whether person n was able to learn the microskill m """

        if test:
            req_cog_cap = self.test_matrix[person, COG_CAP_INDEX]
            concentration = self.personality_matrix[person, CONC_INDEX]
        else:
            req_cog_cap = self.knowledge_matrix[microskill, COG_CAP_INDEX]
            concentration = self.get_concentration(person)
            self.concentration_matrix[microskill, person] = concentration

        acquired_know = self.get_acquired_knowledge(person, microskill)
        cog_cap = cog_cap[timepoint]

        total_req_cog_cap = req_cog_cap - (acquired_know * params.ACQ_KNOWL_WEIGHT)
        avail_cog_cap = cog_cap * concentration

        if not self.achievement_matrix[person, microskill] and total_req_cog_cap < avail_cog_cap:
            return True
        elif self.achievement_matrix[person, microskill] is True:
            #TODO: can a microskill be unlearned?
            return True

        return False

    def get_cog_cap(self, person: int):
        """ Get the cognitive capacity of person n at time t (cog_cap changes as a function of age) """

        x_max_cog_cap = (T / TOTAL_YEARS) * params.PEAK_YEAR_COG_CAP
        y_max_cog_cap = self.personality_matrix[person, COG_CAP_INDEX]
        x = np.arange(T)
        start_perc_cog_cap = params.START_PERC_COG_CAP

        a = (-y_max_cog_cap) / np.power(x_max_cog_cap, 2)
        y = ((a * np.power((x - x_max_cog_cap), 2)) + y_max_cog_cap) * (1 - start_perc_cog_cap) + start_perc_cog_cap * y_max_cog_cap

        return y

    def get_concentration(self, person: int):
        """ Get concentration of person n, random noise is added """
        # TODO: test with these params (concentration is often 0) + add params in parameter file

        max_concentration = self.personality_matrix[person, CONC_INDEX]
        rand_noise = truncnorm.rvs(a=0, b=np.inf, loc=params.MEAN_CONC, scale=params.SD_CONC)
        concentration = float(max_concentration - rand_noise)

        concentration = 0 if concentration < 0 else concentration

        return concentration

    def get_acquired_knowledge(self, person: int, microskill: int):
        """ Get sum of already acquired microskills similar to microskil m """

        acquired_microskills = np.argwhere(self.achievement_matrix[person, :] > 0)

        if len(acquired_microskills) == 0:
            return int(0)

        return sum(self.microskill_similarity_matrix[microskill, acquired_microskills])
