from src import simulation, tests
import time

start_time = time.time()

""" Run Simulation """
simulation = simulation.Simulation()
simulation.run()

print("--- Simulation took %s seconds ---" % round(time.time() - start_time, 2))

""" Run Tests """
tests = tests.Test(simulation)
tests.run()
