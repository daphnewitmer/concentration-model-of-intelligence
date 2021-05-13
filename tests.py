import matplotlib.pyplot as plt
from matrices import AchievementMatrix
import numpy as np


def plot_distribution(x, y=None, title='', x_title='', y_title=''):
    if y is None:
        plt.plot(x)
    else:
        plt.plot(x, y)
    plt.title(title)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.show()

def check_knowledge_structure(personality_matrix, knowledge_matrix):
    # TODO: make usable for test_matrix as well
    achievement_matrix = AchievementMatrix(personality_matrix, knowledge_matrix)

    for person in range(1):
        cog_cap = AchievementMatrix.get_cog_cap(achievement_matrix, person)

    # TODO: plot over 25 years instead of 1000 timepoints
    plt.plot(np.arange(1000), knowledge_matrix[:, 0])
    plt.plot(np.arange(1000), cog_cap)
    plt.show()

    microskill_similarity_matrix = knowledge_matrix.dot(knowledge_matrix.T)
    sum_knowl_array = microskill_similarity_matrix.sum(axis=0)

    plt.plot(np.arange(1000), sum_knowl_array)
    plt.show()

def check_learning_structure(nr_Learned_skills_per_timestep):
    plt.plot(np.arange(1000), nr_Learned_skills_per_timestep)
    plt.show()

    # TODO: second and third part of section
