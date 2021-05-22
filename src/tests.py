import matplotlib.pyplot as plt
import numpy as np
from src import parameters as params


def check_knowledge_structure(simulation, achievement_matrix, learning_matrix):
    # TODO: make usable for test_matrix as well

    for person in range(1):
        cog_cap = simulation.get_cog_cap(person)

        plt.plot(np.arange(params.nrOfMicroskillsNormal), simulation.knowledge_matrix[:, 0], label='Required Cognitive Capacity')
        plt.plot(np.arange(params.nrOfMicroskillsNormal), cog_cap, label='Person (p=' + str(person) + ') Cognitive Capacity')
        plt.xlabel('Microskills')
        plt.ylabel('Cognitive Capacity')
        plt.title('Knowledge Structure')
        plt.legend()
        plt.show()
#
#     # TODO: remove cog_cap from knowledge here
#     microskill_similarity_matrix = knowledge_matrix.dot(knowledge_matrix.T)
#     sum_knowl_array = microskill_similarity_matrix.sum(axis=0)
#
#     plt.plot(np.arange(params.nrOfMicroskillsNormal), sum_knowl_array)
#     plt.xlabel('Microskills')
#     plt.ylabel('Sum Knowledge')
#     plt.title('')
#     plt.show()
#
# def check_learning_structure(nr_Learned_skills_per_timestep):
#     plt.plot(np.arange(1000), nr_Learned_skills_per_timestep)
#     plt.show()
#
#     # TODO: second and third part of section
