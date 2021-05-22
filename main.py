from src import simulation, tests

""" Simulation """
simulation = simulation.Simulation()
achievement_matrix, learning_matrix = simulation.run()

""" Tests """
# tests.check_learning_structure()
tests.check_knowledge_structure(simulation, achievement_matrix, learning_matrix)


