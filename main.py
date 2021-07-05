from src import simulation, tests
import time
import src.parameters as params

start_time = time.time()

""" Run Simulation """
simulation = simulation.Simulation()
simulation.run()

print("--- Simulation took %s seconds ---" % round(time.time() - start_time, 2))

""" Run Tests """
tests = tests.Test(simulation)
# tests.run()

""" Compare multiple simulations """
# tests.save_iq_test()
tests.compute_heritability()