from src import simulation, tests

""" Simulation """
simulation = simulation.Simulation()
achievement_matrix, learning_matrix = simulation.run()

""" Tests """
# tests.check_knowledge_structure(simulation)
# tests.check_test_structure(simulation)
# tests.check_learning_structure(simulation, learning_matrix)
# tests.check_learning_correlations(simulation, learning_matrix)
tests.check_iq_scores(learning_matrix)


