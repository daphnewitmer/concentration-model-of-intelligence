import matrices
import parameters as params

# personality_matrix = matrices.create_personality_matrix(params.PERS_MEAN, params.PERS_SD, params.PERS_TWIN)
# print(personality_matrix.shape)

knowledge_matrix = matrices.create_knowledge_matrix(params.KNOW_SD)
# print(knowledge_matrix.shape)

test_matrix = matrices.create_test_matrix(knowledge_matrix)
print(test_matrix.shape)



