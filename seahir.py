import sys
import math
import numpy as np
from typing import Dict, Tuple
from mpi4py import MPI
from dataclasses import dataclass
import random
from repast4py import core, schedule, logging
from repast4py import context as ctx
from repast4py.network import UndirectedSharedNetwork
from repast4py.parameters import create_args_parser, init_params
import networkx as nx

class Person(core.Agent):
    """
    Person Agent with Susceptible, Exposed, Asymptomatic, Hospitalized, Infected, and Recovered States
    """
    PERSON_TYPE = 1
    #Agent States
    SUSCEPTIBLE = 0
    EXPOSED = 1
    ASYMPTOMATIC = 2 #Asymp
    HOSPITALIZED = 3 #Moderate/Severe Cases
    ISOLATED = 4 #Mild Cases
    REMOVED = 5
    RECOVERED = 6
        
    def __init__(self, id, state=SUSCEPTIBLE):
        super().__init__(id, Person.PERSON_TYPE)
        self.state = state
        self.days = 0 #How many days the person has been in their current state

    def step(self):
        self.days += 1

        if self.state == self.SUSCEPTIBLE:
            self.expose()
        elif self.state == self.EXPOSED:
            #Decides what case they are
            self.infected()
        elif self.state == self.ASYMPTOMATIC:
            #Figure out what happens if their asymptomatic
            self.asymp()
        elif self.state == self.HOSPITALIZED:
            #Maybe add a check to see how many hospital beds are being taken up and if they die
            self.hospital()
        elif self.state == self.ISOLATED:
            self.isolated()
        elif self.state == self.RECOVERED:
            self.recover()
    
    def expose(self):
        """Person goes from susceptible to exposed"""
        infectionRate = 0.5
        if random.random() > infectionRate:
            self.state = self.EXPOSED
            self.days = 0
    
    def infected(self):
        """Person goes from exposed to either asymptomatic, isolate, or hospitalized"""
        asympRate = 0.3
        isolateRate = 0.55
        #hospitalizedRate = 0.15
        chance = random.random()

        if chance < asympRate:
            self.state = self.ASYMPTOMATIC
        elif chance < asympRate + isolateRate:
            self.state = self.ISOLATED
        else:
            self.state = self.HOSPITALIZED
    

    def asymp(self):
        """Person is asymptomatic and can infect others"""
        if self.days == 5:
            self.state = self.RECOVERED

    #Maybe split people who are hospitalized into ICU and Hospitalization
    def hospital(self):
        """Person is currently hospitalized and can either die or recover"""
        deathRateHospitalization = 0.18
        #deathRateICU = 0.44
        daysInHospital = 18
        chance = random.random()

        if chance < deathRateHospitalization:
            self.state = self.REMOVED
        elif self.days >= daysInHospital:
            self.state = self.RECOVERED


    def isolated(self):
        """Person is currently isolating and has the chance to recover or become hospitalized"""
        hospitalizedRate = 0.15
        chance = random.random()

        if chance < hospitalizedRate:
            self.state = self.HOSPITALIZED
        elif self.days == 14:
            self.state = self.RECOVERED

    def recover(self):
        """Person has recovered and is removed"""
        #TODO Finish Function

class Model:
    """Model Class"""
    SUSCEPTIBLE: int = 0
    EXPOSED: int = 0
    ASYMPTOMATIC: int = 0 
    HOSPITALIZED: int = 0 
    ISOLATED: int = 0
    REMOVED: int = 0
    RECOVERED: int = 0

    def __init__(self, comm, populationSize):
        self.context = ctx.SharedContext(comm)
        self.schedule = schedule.Schedule()
        self.populationSize = populationSize
        self.time = 0
        self.init_population()

    def init_population(self):
        for i in range(self.populationSize):
            person = Person(i)
            self.context.add(person)
        self.schedule.schedule_repeating_event(1, 1, self.step)

    def step(self):
        for agent in self.context.agents():
            agent.step()

    def run(self, time):
        for t in range(time):
            self.schedule.execute()
            counts = self.counts()
            print(f"Day {t + 1}: {counts}")

    def counts(self):
        counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
        for person in self.context.agents():
            counts[person.state] += 1
        return counts

def main():
    comm = MPI.COMM_WORLD
    model = Model(comm, 1000)
    model.run(time=30)

if __name__ == "__main__":
    main()