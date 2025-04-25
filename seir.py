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
    INFECTED = 2
    REMOVED = 3

    def __init__(self, id, local_rank, context, network, state=SUSCEPTIBLE):
        super().__init__(id, Person.PERSON_TYPE)
        self.state = state
        #print(state)
        self.days = 0 #Represents days in current state
        self.days_since_infected = 0 #Represents days since infected
        self.location = "home"
        self.local_rank = local_rank
        occupations = ["worker", "student", "unemployed"]
        proportions = [0.5, 0.3, 0.2]
        self.occupation = random.choices(occupations, proportions)[0]
        self.mask = random.random() < 0.2 #80% chance of wearing a mask
        self.context = context
        self.network = network

    def step(self):
        if self.state != self.SUSCEPTIBLE and self.state != self.REMOVED:
            self.days_since_infected += 1  # Only increment if infected
        self.days += 1

        #print(f"Agent {self.id} is {self.state}")
        if self.state == self.SUSCEPTIBLE:
            #print(f"Agent {self.id} is SUSCEPTIBLE")
            self.expose()
        elif self.state == self.EXPOSED:
            #print(f"Agent {self.id} is EXPOSED")
            self.infected()
        elif self.state == self.INFECTED:
            #print(f"Agent {self.id} is REMOVED")
            self.remove()

    def expose(self):
        """Person goes from susceptible to exposed"""
        if self.days >= 0:
            # Calculate the infection probability based on the equations
            beta_t = 0.2 
            contact_rate = self.calculate_contact_rate()
            infection_probability = beta_t * contact_rate
            
            #Adjusts infection probability based on mask, if wearing mask reduce probability by 14%
            #1 - 0.14
            if self.mask:
                infection_probability *= 0.30
            
            #If infected neighbor is wearing a mask, reduce infection probability further
            infectedNeighbors = [n for n in self.get_neighbors() if n.state in [self.INFECTED]]

            if infectedNeighbors:
                maskReduction = sum (0.86 for n in infectedNeighbors if n.mask) / len(infectedNeighbors)
                infection_probability *= maskReduction

            if random.random() < infection_probability:
                self.state = self.EXPOSED
                self.days = 0

    def calculate_contact_rate(self):
        # Base contact rate for random interactions (e.g., in public spaces)
        base_contact_rate = 0.4

        # Calculate the contact rate based on the number of infected neighbors
        infected_count = self.count_infected_neighbors()
        neighbor_count = len(list(self.get_neighbors()))  # Count neighbors

        if neighbor_count == 0:
            return base_contact_rate  # Return the base rate if no neighbors

        # Calculate the contact rate as a combination of base rate and network-based rate
        network_contact_rate = infected_count / neighbor_count
        scaling_factor = 1  #Amplifies the network-based rate default 1

        # Combine base rate and network-based rate
        total_contact_rate = base_contact_rate + (scaling_factor * network_contact_rate)

        # Ensure the contact rate does not exceed 1
        return min(total_contact_rate, 1.0)


    def count_infected_neighbors(self):
        # Count the number of infected neighbors
        infected_count = 0
        for neighbor in self.get_neighbors():
            if neighbor.location == self.location and neighbor.state in [self.INFECTED]:
                infected_count += 1
        return infected_count

    def get_neighbors(self):
        # Get the neighbors of the current agent
        return self.network.graph.neighbors(self)

    def get_total_population(self):
        # Get the total population size
        return self.context.size()

    def infected(self):
        """Person goes from exposed to infected after incubation period"""
        incubation = 5
        if self.days_since_infected >= incubation and self.state == self.EXPOSED:
            self.state = self.INFECTED
            self.days = 0
            self.infectious = True

    def remove(self):
        """Person goes from infected to removed (recovered or deceased)"""
        infectious_period = 10  # Average days infectious
        mortality_rate = 0.02  # 2% chance of death
        
        if self.days >= infectious_period and self.state == self.INFECTED:
            if random.random() < mortality_rate:
                self.deceased = True
            self.state = self.REMOVED
            self.infectious = False

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
        initial_infected_count = max(1, int(0.01 * self.populationSize))  # 1% infected to start
        
        for i in range(self.populationSize):
            initial_state = Person.EXPOSED if i < initial_infected_count else Person.SUSCEPTIBLE
            person = Person(i, local_rank=rank, context=self.context, network=self.network, state=initial_state)
            self.context.add(person)
            self.network.add_nodes([person])  
            
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
                    person = Person(node_id, local_rank=rank, context=self.context, network=self.network)  # Pass the MPI rank as local_rank
                    self.context.add(person)
                    self.network.add_nodes([person])  # Add the Person object to the network
                    node_to_agent[node_id] = person  # Map node ID to Person object
                
                # Add edges for each connection
                for neighbor in connections:
                    if neighbor not in node_to_agent:
                        # Create a new Person object for the neighbor
                        neighbor_person = Person(neighbor, local_rank=rank, context=self.context, network=self.network)  # Pass the MPI rank as local_rank
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
                self.update_locations(hour)  #Handle movement within the day
            self.schedule.execute()  #Execute the step once per day
            counts = self.counts()
            #print(f"Day {day + 1}: {counts}")
            print(f"{day + 1}: {counts},")

    #TODO: Locations might be bugged, not changing infection rates
    def update_locations(self, hour):
        for person in self.context.agents():
            if 8 <= hour < 17:  # Work/school hours (8 AM - 5 PM)
                if person.occupation == "worker" and person.state not in [4,3]:
                    person.location = "work"
                elif person.occupation == "student" and person.state not in [4,3]:
                    person.location = "school"
                else:
                    person.location = "home"
            else:
                person.location = "home"

    def counts(self):
        counts = {i: 0 for i in range(4)}
        for person in self.context.agents():
            counts[person.state] += 1
        return counts

def main():
    comm = MPI.COMM_WORLD
    network_file = "network_output.csv"
    model = Model(comm, 1000, network_file)
    model.run(days=100)

if __name__ == "__main__":
    main()