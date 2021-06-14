import numpy as np
from src import parameters as params
from scipy.stats import truncnorm
import matplotlib.pyplot as plt

""" Global variables, specified in parameter file """
N = params.nrOfSimulatedPeople
C = params.nrOfCharacteristics
M = params.nrOfMicroskillsNormal
Q = params.nrOfMicroskillsIQ
T = params.nrOfTestOccasions
F = params.nrOfFactors

np.random.seed(42)
# TODO: move numbers to parameters file + structure? + concentration/cog_cap index as param?


def create_personality_matrix(twin_type):
    """
    Matrix that codes the value of person i on characteristic c (cognitive capacity and concentration) (N x C)
    """
    # TODO: implement twin options

    cog_cap = truncnorm.rvs(a=0, b=np.inf, loc=params.PERS_MEAN_COP_CAP, scale=params.PERS_SD_COG_CAP, size=(N, 1))
    conc = truncnorm.rvs(a=0, b=np.inf, loc=params.PERS_MEAN_CONC, scale=params.PERS_SD_CONC, size=(N, 1))
    personality = np.hstack((cog_cap, conc))

    return personality


def create_knowledge_matrix(personality_matrix):
    """
    Matrix that codes how each factor f loads onto each microskill m (M x F) F0=cog_cap
    """

    max_cog_cap_personality = np.max(personality_matrix[:, 0])
    first_years = int(params.YEARS_ZERO_COG_CAP * (M / params.TOTAL_YEARS_OF_SIMULATION))

    mean = np.linspace(0, max_cog_cap_personality * params.PERC_OF_MAX_COG_CAP, M - first_years)
    time = np.linspace(0, M-first_years, M-first_years)
    sinusoid = np.sin(time / (M / params.TOTAL_YEARS_OF_SIMULATION) * (2 * np.pi))
    conv_mean = mean + 0.5 * np.multiply(mean, sinusoid)

    cog_cap_first_years = np.zeros((first_years, 1))
    cog_cap_last_years = [truncnorm.rvs(a=0, b=np.inf, loc=u, scale=params.KNOW_SD_COG_CAP, size=1) for u in conv_mean]
    cog_cap = np.vstack((cog_cap_first_years, cog_cap_last_years))
    other = truncnorm.rvs(a=0, b=np.inf, loc=params.KNOW_MEAN, scale=params.KNOW_SD, size=(M, F - 1))

    knowledge = np.hstack((cog_cap, other))

    return knowledge


def create_test_matrix(knowledge_matrix):
    """
    Matrix that codes how each factor f loads onto each microskill m (Q x F)
    """
    # TODO: check peaks and valleys


    cog_cap = knowledge_matrix[:, 0]
    TOTAL_YEARS = params.TOTAL_YEARS_OF_SIMULATION
    test_types = {'child':  np.array([10, 11, 12, 13, 14], dtype=int), 'adult': np.array([18, 19, 20, 21, 22], dtype=int)}
    part_matrix = {}

    for type, age in test_types.items():
        # Select microskills on last 5 peaks and last 5 valleys of cognitive capacity
        max_sine = (T / TOTAL_YEARS) * age + ((T / TOTAL_YEARS) / 4)
        min_sine = (T / TOTAL_YEARS) * age + ((T / TOTAL_YEARS) / 4) * 3
        peak_valley_skills = np.concatenate((max_sine.astype(int), min_sine.astype(int)))

        # plt.plot(cog_cap)
        # plt.plot(peak_valley_skills, cog_cap[peak_valley_skills], 'ro')
        # plt.show()
        # exit()

        # Permutate factor values except cog_cap
        factors_without_cog_cap = knowledge_matrix[peak_valley_skills, 1:]
        factors_permuted = cog_cap[peak_valley_skills, np.newaxis]


        for column in factors_without_cog_cap.T:
            column_permuted = np.random.permutation(column)
            factors_permuted = np.hstack((factors_permuted, np.expand_dims(column_permuted, axis=1)))

        # Copy items with randomly distributed noise
        rest_skills_without_noise = np.tile(factors_permuted, [9, 1])  # Tile: repeat ten skills another x (=9) times
        noise = np.random.normal(0, .1, rest_skills_without_noise.shape)
        rest_skills_with_noise = rest_skills_without_noise + noise

        part_matrix[type] = np.vstack((factors_permuted, rest_skills_with_noise))

    test_matrix = np.vstack((part_matrix['child'], part_matrix['adult']))

    return test_matrix


def create_schooling_array():
    """
    List that codes which microskill m is offered at which time step t. This matrix is created for every person i (M)
    """

    skill_sample_age = params.SKILLS_TO_SAMPLE_FROM_PER_AGE
    periods = params.PERIODS
    all_perc_rand = params.PERC_RAND
    TOTAL_YEARS = params.TOTAL_YEARS_OF_SIMULATION

    time_steps_per_year = int(T / TOTAL_YEARS)
    schooling_array = []

    for i in range(TOTAL_YEARS):

        if i < periods['first_period']:
            perc_rand = all_perc_rand['first_period']
            replace = True
        elif i >= periods['first_period'] and i < periods['second_period']:
            perc_rand = all_perc_rand['second_period']
            replace = True

        elif i >= periods['second_period'] and i < periods['third_period']:
            perc_rand =all_perc_rand['third_period']
            replace = False

        elif i >= periods['third_period']:
            perc_rand = all_perc_rand['fourth_period']
            replace = True

        try:
            perc_rand, replace
        except NameError:
            # TODO: Error handling
            return

        sample_size_all_skills = np.round((time_steps_per_year * perc_rand), 0).astype(int)
        sample_size_skills_age = np.round((time_steps_per_year * (1 - perc_rand)), 0).astype(int)

        random = np.random.choice(np.arange(M),
                                  size=sample_size_all_skills,
                                  replace=True)  # Sample from all microskills

        fitting_for_period = np.random.choice(
            np.arange(i * skill_sample_age, (i * skill_sample_age + skill_sample_age), dtype=int),
            size=sample_size_skills_age,
            replace=replace)  # Sample from skills associated with age

        selected_skills = np.append(random, fitting_for_period)
        np.random.shuffle(selected_skills)
        schooling_array.extend(selected_skills)

    return schooling_array
