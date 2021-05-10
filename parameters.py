"""
Matrix sizes
"""
nrOfSimulatedPeople = int(100)      # N
nrOfCharacteristics = int(2)        # C
nrOfMicroskillsNormal = int(1000)   # M
nrOfMicroskillsIQ = int(100)        # Q
nrOfTestOccasions = int(1000)       # T
nrOfFactors = int(10)               # F

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