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
COG_CAP_INDEX = int(0)  # index for cog_cap in personality_matrix, knowledge_matrix and test_matrix
CONC_INDEX = int(1)  # index for concentration in personality_matrix

nrOfPersInTest = int(20)  # run simulation for a part of the possible simulated persons

"""
Parameters to create personality matrix
"""
PERS_MEAN = float(0.7)  # mean of truncated normal distribution to sample cognitive capacity from
PERS_SD = float(0.5)  # sd of truncated normal distribution to sample cognitive capacity from
PERS_MEAN_CONC = float(0.6)  # mean of truncated normal distribution to sample concentration from
PERS_SD_CONC = float(0.25)  # sd of truncated normal distribution to sample concentration from
PERS_TWIN = None

"""
Parameters to create knowledge matrix
"""
KNOW_MEAN = int(0)  # mean of truncated normal distribution to sample factors from (all but cog_cap)
KNOW_SD = float(0.5)  # sd of truncated normal distribution to sample factors from (all but cog_cap)
KNOW_SD_COG_CAP = float(0.5)  # sd of truncated normal distribution to sample cog cap from
PERC_OF_MAX_COG_CAP = float(0.7)  # required cog cap is set to this percentage of max cog cap of persons

"""
Parameters to create schooling matrix
"""
SKILLS_TO_SAMPLE_FROM_PER_AGE = int(25)  # first year the first x skills can be sampled, second year the second x
PERIODS = {
    'first_period': int(4),
    'second_period': int(6),
    'third_period': int(18)
}

PERC_RAND = {
    'first_period': float(0.75),
     'second_period': float(0.5),
     'third_period': float(0.5),
     'fourth_period': float(0.5)
}

"""
Parameters to create achievement matrix
"""
PEAK_YEAR_COG_CAP = int(18)
TEST_AGE = int(18)
START_PERC_COG_CAP = float(0.4)
ACQ_KNOWL_WEIGHT = float(0.002)
SD_CONC = float(0.2)  # sd to sample concentration noise from
MEAN_CONC = float(0)  # mean to sample concantration noise from