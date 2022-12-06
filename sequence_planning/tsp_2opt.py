import time
import random
# import bpy

# from bpy import context

import numpy as np


def random_route(number_of_points):
    '''Generate a random ordering of indices'''
    return random.sample(range(0, number_of_points), number_of_points)


def calculate_route_length(route, points):
    '''Add up the length of all the edges in the route to get the complete length'''
    distance = 0

    for i in range(len(route) - 1):
        distance += get_distance(route[i], route[(i + 1) % points.shape[0]])
        # distance += np.linalg.norm(points[route[i]] -
        #                            points[route[(i + 1) % points.shape[0]]])

    return distance


def create_edges_from_route(route):
    '''Take a route and return a list of edges for a mesh'''
    edges = []
    for i in range(len(route) - 1):
        edges.append((route[i], route[i+1]))

    edges.append((route[len(route) - 1], route[0]))

    return edges


def swap_2opt(route, point1, point2):
    # Reverse the section indicated by the indices
    new_route = route.copy()
    new_route[(point1 + 1):(point2 + 1)
              ] = route[(point1 + 1):(point2 + 1)][::-1]
    return new_route


def precalculate_distances(points):
    '''Memoizes the distances between all points'''
    global distances

    distances = np.zeros([len(points), len(points)])
    for x in range(len(points)):
        for y in range(len(points)):
            distances[x][y] = np.linalg.norm(points[x] - points[y])

    return distances

def get_distance(point1, point2):
    '''Retrieves the memoized distance'''

    return distances[point1][point2]
