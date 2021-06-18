import numpy as np

"""
Matrix sizes
"""
nrOfSimulatedPeople = int(100)      # N
nrOfCharacteristics = int(2)        # C
nrOfMicroskillsNormal = int(1000)   # M
nrOfMicroskillsIQ = int(100)        # Q
nrOfTestOccasions = int(1000)       # T
nrOfFactors = int(10)               # F

TOTAL_YEARS_OF_SIMULATION = 25

nrOfPersInTest = int(10)  # run simulation for a part of the possible simulated persons (faster simulation for testing)

"""
Parameters to create personality matrix
"""
PERS_MEAN_COP_CAP = float(0.7)  # mean of truncated normal distribution to sample cognitive capacity from
PERS_SD_COG_CAP = float(0.5)  # sd of truncated normal distribution to sample cognitive capacity from
PERS_MEAN_CONC = float(0.5)  # mean of truncated normal distribution to sample concentration from
PERS_SD_CONC = float(0.2)  # sd of truncated normal distribution to sample concentration from
PERS_TWIN = 'none'  # none, mono, diz

"""
Parameters to create knowledge matrix
"""
KNOW_MEAN = int(0)  # mean of truncated normal distribution to sample factors from (all but cog_cap)
KNOW_SD = float(0.5)  # sd of truncated normal distribution to sample factors from (all but cog_cap)
KNOW_SD_COG_CAP = float(0.5)  # sd of truncated normal distribution to sample cog cap from
PERC_OF_MAX_COG_CAP = float(0.7)  # required cog cap is set to this percentage of max cog cap of persons
YEARS_ZERO_COG_CAP = int(2)  # nr of years where microskills require zero cognitive capacity

"""
Parameters to create schooling matrix
"""
SKILLS_TO_SAMPLE_FROM_PER_AGE = int(25)  # first year the first x skills can be sampled, second year the second x
PERIODS = {
    'first_period': int(4),  # 4 * 40 = 160 timestep
    'second_period': int(6),  # 6 * 40 = 240 timestep
    'third_period': int(18)  # 18 * 40 = 720 timestep
}

PERC_RAND = {
    'first_period': float(0.75),  # 0.75
    'second_period': float(0.5),  # 0.5
    'third_period': float(0.5),  # sample cannot be lower than SKILLS_TO_SAMPLE_FROM_PER_AGE because sampled without replacement
    'fourth_period': float(0.5)  # 0.5
}

"""
Parameters to create test matrix
"""
NORMALIZE_TEST = False

"""
Parameters to create achievement matrix
"""
PEAK_YEAR_COG_CAP = int(18)
TEST_AGES = np.array([10, 14, 18, 22])
START_PERC_COG_CAP = float(0)
ACQ_KNOWL_WEIGHT = float(0.001)  # 0.001
SD_CONC_NOISE = float(0.2)  # sd to sample concentration noise from
MEAN_CONC_NOISE = float(0)  # mean to sample concantration noise from