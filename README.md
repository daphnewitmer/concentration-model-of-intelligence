# Concentration Model of Intelligence

In this computational model of intelligence a population of individuals is simulated, each with different characteristic 
and their own opportunities to learn micro skills (skill that is either learned or not, instead of evaluated on a scale). 
At several times in the simulated life course, a simulated IQ test is performed. Important concepts:
```
N: nr of simulated people
C: nr of characteristics of these people
M: nr of normal micro skills 
Q: nr of micro skills that are part of the IQ test
T: time expressed as the nr of occasions
F: nr of factors that determine the structure of the micro skills
```

## Usage
Run the simulation with:
```bash
main.py
```
All parameters can be adjusted in the `parameters.py` file. For development the number of people in the simulation can be adjusted 
here to speed up the simulation (*nrOfPersInTest*). Test results will be less reliable so only use this for testing code. 

## Code Structure
### matrices.py
This file contains four functions, each creating a different matrix that is needed for the simulation. 
- `create_personality_matrix(twin_type)` N x C matrix that codes the value 
of person i on characteristic c.
- `create_knowledge_matrix(personality_matrix)` M x F matrix that codes how each factor f loads onto each micro skill m.
- `create_test_matrix(knowledge_matrix)` Q x F matrix that codes how each factor f loads onto each micro skill m.
- `create_schooling_array()` array of size M that codes which micro skill m is offered at which time step t. This array is created anew for every person i.

### simulation.py
In this file the life course of every person is simulated. Micro skills from the *schooling_array* are offered to every 
person and evaluated whether they learn the micro skill or not. The following matrices are created:
- Learning matrix: (T+Q) x N matrix that codes whether the micro skill offered at time step t was learned by person i.
- Achievement matrix: N x M matrix that codes whether micro skill m has been acquired by person i.

### tests.py
### parameters.py
