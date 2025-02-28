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
import random
import csv

random.seed(42)  # Set a random seed for reproducibility

class Agent:
    def __init__(self, node_id):
        self.id = node_id
        self.local_rank = None  # To be set based on the rank

    def set_local_rank(self, rank):
        self.local_rank = rank

class Person(core.Agent):
    PERSON_TYPE = 1
    SUSCEPTIBLE = 0
    EXPOSED = 1
    ASYMPTOMATIC = 2
    HOSPITALIZED = 3
    ISOLATED = 4
    REMOVED = 5
    RECOVERED = 6

    def __init__(self, id, local_rank, state=SUSCEPTIBLE):
        super().__init__(id, Person.PERSON_TYPE)
        self.state = state
        print(state)
        self.days = 0
        self.location = "home"
        self.local_rank = local_rank
        occupations = ["worker", "student", "unemployed"]
        proportions = [0.5, 0.3, 0.2]
        self.occupation = random.choices(occupations, proportions)[0]

    def step(self):
        self.days += 1
        if self.state == self.SUSCEPTIBLE:
            print(f"Agent {self.id} is SUSCEPTIBLE")
            self.expose()
        elif self.state == self.EXPOSED:
            print(f"Agent {self.id} is EXPOSED")
            self.infected()
        elif self.state == self.ASYMPTOMATIC:
            print(f"Agent {self.id} is ASYMPTOMATIC")
            self.asymp()
        elif self.state == self.HOSPITALIZED:
            print(f"Agent {self.id} is HOSPITALIZED")
            self.hospital()
        elif self.state == self.ISOLATED:
            print(f"Agent {self.id} is ISOLATED")
            self.isolated()
        elif self.state == self.RECOVERED:
            print(f"Agent {self.id} is RECOVERED")
            self.recover()
        elif self.state == self.REMOVED:
            print(f"Agent {self.id} is REMOVED")

    def expose(self):
        infectionRate = 0.1
        if random.random() > infectionRate:
            self.state = self.EXPOSED
            self.days = 0

    def infected(self):
        asympRate = 0.5  # Increased chance of becoming ASYMPTOMATIC
        isolateRate = 0.3  # Reduced chance of becoming ISOLATED
        hospitalRate = 0.2  # Added to ensure probabilities sum to 1.0
        chance = random.random()
        if chance < asympRate:
            self.state = self.ASYMPTOMATIC
        elif chance < asympRate + isolateRate:
            self.state = self.ISOLATED
        else:
            self.state = self.HOSPITALIZED

    def asymp(self):
        if self.days == 5:
            self.state = self.RECOVERED

    def hospital(self):
        deathRateHospitalization = 0.18
        daysInHospital = 18
        if self.days >= daysInHospital:  # Check minimum days first
            if random.random() < deathRateHospitalization:
                self.state = self.REMOVED
            else:
                self.state = self.RECOVERED

    def isolated(self):
        hospitalizedRate = 0.15
        if self.days >= 14:  # Check minimum days first
            if random.random() < hospitalizedRate:
                self.state = self.HOSPITALIZED
            else:
                self.state = self.RECOVERED
    def recover(self):
        pass

class Model:
    def __init__(self, comm, populationSize, network_file):
        self.context = ctx.SharedContext(comm)
        self.schedule = schedule.Schedule()
        self.populationSize = populationSize
        self.time = 0
        self.network = UndirectedSharedNetwork("contact_network", comm)
        self.init_population()
        self.load_network_from_file(network_file)

    def init_population(self):
        rank = self.context.comm.Get_rank()
        for i in range(self.populationSize):
            person = Person(i, local_rank=rank, state=Person.SUSCEPTIBLE)
            self.context.add(person)
            self.network.add_nodes([person])  # Add node for each person
        self.schedule.schedule_repeating_event(0, 1, self.step)

    def load_network_from_file(self, filename):
        rank = self.context.comm.Get_rank()  # Get the MPI rank of the current process
        node_to_agent = {}  # Dictionary to map node IDs to Person objects

        with open(filename, "r") as file:
            reader = csv.reader(file)
            next(reader)  # Skip the header row
            
            for row in reader:
                node_id = int(row[0])  # Extract node ID
                connections = list(map(int, row[2].strip('"').split(", ")))  # Parse connections
                
                # Ensure the node exists in the network
                if node_id not in node_to_agent:
                    # Create a new Person object for this node
                    person = Person(node_id, local_rank=rank)  # Pass the MPI rank as local_rank
                    self.context.add(person)
                    self.network.add_nodes([person])  # Add the Person object to the network
                    node_to_agent[node_id] = person  # Map node ID to Person object
                
                # Add edges for each connection
                for neighbor in connections:
                    if neighbor not in node_to_agent:
                        # Create a new Person object for the neighbor
                        neighbor_person = Person(neighbor, local_rank=rank)  # Pass the MPI rank as local_rank
                        self.context.add(neighbor_person)
                        self.network.add_nodes([neighbor_person])  # Add the Person object to the network
                        node_to_agent[neighbor] = neighbor_person  # Map neighbor ID to Person object
                    
                    # Add the edge (if it doesn't already exist)
                    if not self.network.graph.has_edge(node_to_agent[node_id], node_to_agent[neighbor]):
                        self.network.add_edge(node_to_agent[node_id], node_to_agent[neighbor])

    def step(self):
        for agent in self.context.agents():
            agent.step()

    def run(self, days):
        for day in range(days):
            for hour in range(24):
                self.schedule.execute()
            counts = self.counts()
            print(f"Day {day + 1}: {counts}")

    def counts(self):
        counts = {i: 0 for i in range(7)}
        for person in self.context.agents():
            counts[person.state] += 1
        return counts

def main():
    comm = MPI.COMM_WORLD
    network_file = "contact_network.txt"
    model = Model(comm, 5, network_file)
    model.run(days=3)

if __name__ == "__main__":
    main()