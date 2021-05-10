import matrices
import parameters as params

personality_matrix = matrices.create_personality_matrix(params.PERS_MEAN, params.PERS_SD, params.PERS_TWIN)
# print(personality_matrix.shape)

knowledge_matrix = matrices.create_knowledge_matrix(params.KNOW_SD)
# print(knowledge_matrix.shape)

# test_matrix = matrices.create_test_matrix(knowledge_matrix)
# print(test_matrix.shape)

achievement_matrix = matrices.AchievementMatrix(personality_matrix, knowledge_matrix)

# create schooling matrix for every person and update achievement matrix for every person
for person in range(1):
    schooling_array = matrices.create_schooling_matrix(params.FIRST_PERIOD, params.SECOND_PERIOD, params.THIRD_PERIOD)

    achievement_matrix.update(person, schooling_array)


