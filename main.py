from src import simulation, tests

""" Simulation """
simulation = simulation.Simulation()
achievement_matrix, learning_matrix = simulation.run()

""" Tests """
tests = tests.Test(simulation, learning_matrix)
tests.run()
