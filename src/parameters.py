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

"""
Parameters to create personality matrix
"""
PERS_MEAN = 0  # mean of truncated normal distribution to sample characteristics from
PERS_SD = 1  # sd of truncated normal distribution to sample characteristics from
PERS_TWIN = None

"""
Parameters to create knowledge matrix
"""
KNOW_SD = 0.5

"""
Parameters to create schooling matrix
"""
FIRST_PERIOD = int(4)
SECOND_PERIOD = int(6)
THIRD_PERIOD = int(18)
SKILLS_TO_SAMPLE_FROM_PER_AGE = int(25)  # first year the first x skills from the knowledge matrix can be sampled, second year the second x

"""
Parameters to create achievement matrix
"""
PEAK_YEAR_COG_CAP = int(18)
TEST_AGE = int(18)