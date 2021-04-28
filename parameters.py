"""
Matrix sizes
"""
N = int(100)  # nr of simulated people
C = int(2)  # nr of characteristics of these people
M = int(1000)  # nr of normal microskills
Q = int(100)  # nr of microskills that are part of the IQ test
T = int(1000)  # time expressed as the nr of occasions
F = int(10)  # nr of factors that determine the structure of the microskills

"""
Parameters to create personality matrix
"""
pers = {
    "mean": 0,  # mean of truncated normal distribution to sample characteristics from
    "sd": 1,  # sd of truncated normal distribution to sample characteristics from
    "twin_type": None
}
