from src import simulation, tests
import time

""" Simulation """
start_time = time.time()

simulation = simulation.Simulation()
achievement_matrix, learning_matrix = simulation.run()

print("--- %s seconds ---" % (time.time() - start_time))

""" Tests """
tests = tests.Test(simulation, learning_matrix)
tests.run()
