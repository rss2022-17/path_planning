#!/usr/bin/env python
import random

import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped, PoseArray, Point
from nav_msgs.msg import Odometry, OccupancyGrid
import rospkg
import time, os
from utils import LineTrajectory
from tf import transformations as ts
import cv2

class PathPlan(object):
    """ Listens for goal pose published by RViz and uses it to plan a path from
    current car pose.
    """
    def __init__(self):
        self.num_steps = 10000 # can be set as a parameter
        self.end_point_dst_param = 20 # can be set as a parameter
        self.frontier_threshold = 20 # can be set as a parameter

        self.start_time = 0
        self.time_to_plan = 0

        self.odom_topic = rospy.get_param("~odom_topic", "/odom")
        self.map_sub = rospy.Subscriber("/map", OccupancyGrid, self.map_cb)
        self.trajectory = LineTrajectory("/planned_trajectory")
        self.goal_sub = rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.goal_cb, queue_size=10)
        self.traj_pub = rospy.Publisher("/trajectory/current", PoseArray, queue_size=10)
        self.odom_sub = rospy.Subscriber(self.odom_topic, Odometry, self.odom_cb)


    def xy2uv(self, x, y):
        # translates an x, y coordinate to u, v
        res = np.matmul(self.world2map_tfm, np.array([x, y, 0, 1]).T)
        return [int(res[0]/self.map_resolution), int(res[1]/self.map_resolution)]

    def uv2xy(self, u, v):
        # translates an u, v coordinate to x, y
        res = np.matmul(self.map2world_tfm, np.array([u * self.map_resolution, v * self.map_resolution, 0, 1]).T)
        return [res[0], res[1]]

    def map_cb(self, msg):
        # builds a map
        self.map_height = msg.info.height
        self.map_width = msg.info.width
        self.map_resolution = msg.info.resolution
        map_origin_P = msg.info.origin.position
        map_origin_Q = msg.info.origin.orientation
        self.map2world_tfm =  ts.concatenate_matrices(ts.translation_matrix((map_origin_P.x, map_origin_P.y, map_origin_P.z)),
                                                      ts.quaternion_matrix((map_origin_Q.x, map_origin_Q.y, map_origin_Q.z, map_origin_Q.w)))
        self.world2map_tfm = ts.inverse_matrix(self.map2world_tfm)
        self.map_cell_2d = np.reshape(np.array(msg.data, dtype=np.uint8), (msg.info.height, msg.info.width))

    def odom_cb(self, msg):
        # sets current position
        self.start = (msg.pose.pose.position.x, msg.pose.pose.position.y)

    def goal_cb(self, msg):
        # sets goal position
        self.end = (msg.pose.position.x, msg.pose.position.y)
        self.start_time = rospy.get_time()
        self.plan_path(self.start, self.end, self.map_cell_2d)

    class Node:
        # creates an object with a coordinate and parent attribute
        def __init__(self, u, v):
            self.p = [u, v]
            self.parent = None
        # checks for equality
        def __eq__(self, other):
            return self.p[0] == other.p[0] and self.p[1] == other.p[1]

    def random_node(self, map):
        # Returns random node within map that isn't covered by an obstacle
        u = random.randint(0, self.map_width - 1)
        v = random.randint(0, self.map_height - 1)
        if (0 <= u < self.map_width) and (0 <= v < self.map_height) and (map[v, u] == 0):
            return self.Node(u, v)

    def edge_creation(self, node_1, node_2, map):
        # check path collision and returns a boolean of whether edge does not pass through any obstacles
        if node_2 == node_1:
            node_2.parent = node_1.parent
        else:
            # y = mx + b --> b = y - mx
            m = (node_1.p[1] - node_2.p[1]) / (node_1.p[0] - node_2.p[0])
            b = node_1.p[1] - m * node_1.p[0]

            # width coordinates
            start = min(node_1.p[0],node_2.p[0]) 
            stop = max(node_1.p[0],node_2.p[0])

            for x in range(start,stop): # iterate over width (1 pixel at a time)
                y = m*x + b # get corresponding height pixel
                if map[y,x] != 0:
                    return False
            
            node_2.parent = node_1
        return True

    def get_dist(self, node_1, node_2):
        # gets distance between two nodes
        return np.sqrt((node_1.p[0] - node_2.p[0])**2 + (node_1.p[1] - node_2.p[1])**2)

    def get_closest(self, node, nodes):
        # gets the nearest node in the list of nodes
        min_distance = float('inf')
        closest_point = None
        for i in nodes:
            dist = self.get_dist(node, i)
            if dist is not None and min_distance > dist:
                min_distance = dist
                closest_point = i
        return closest_point

    def final_path(self, node):
        # given node, finds a path back to starting node through parent
        # start node has no parent :(
        self.time_to_plan = rospy.get_time() - self.start_time
        
        path = [node.p]
        n = node.parent
        while n is not None:
            path.append(n.p)
            n = n.parent
        path.reverse()

        return path

    def rrt(self, start_point, end_point, map):
        # creates start and end nodes
        end_node = self.Node(*end_point)
        start_node = self.Node(*start_point)

        # list of nodes created and connected
        nodes = [start_node]
        for i in range(self.num_steps):
            within_frontier = False 

            # gets node within a certain distance (frontier_threshold) of a node already in the tree
            while not within_frontier:
                # gets a random node
                rand_node = self.random_node(map)
                # gets the closest node
                closest_node = self.get_closest(rand_node, nodes)

                # get distance between random node and closest node
                rand_closest_dist = self.get_dist(rand_node, closest_node)
                if rand_closest_dist < self.frontier_threshold:
                    within_frontier = True


            if self.edge_creation(closest_node, rand_node, map):
                # if an edge can be created, add to node list
                nodes.append(rand_node)
                dist_from_end = self.get_dist(rand_node, end_node)
                if dist_from_end <= self.end_point_dst_param:
                    # if the random node is within a certain distance from the end point, check if can create an edge
                    # between the two
                    if self.edge_creation(rand_node, end_node, map):
                        # if possible return the path found
                        return self.final_path(end_node)
            
            # if i % 100 == 0:
            #     rospy.loginfo("Still running. We've searched along "+str(i)+ " steps")

        # if no path exists within num_steps, return empty list as unsuccessful
        rospy.loginfo("Couldn't find a path within "+str(self.num_steps)+" steps")
        return []

    def plan_path(self, start_point, end_point, map):
        ## CODE FOR PATH PLANNING ##
        self.trajectory.clear()
        path = self.rrt(self.xy2uv(*start_point), self.xy2uv(*end_point), map)
        # each node is converted back to map coordinates for trajectory
        path_length = 0
        _x, _y = None, None
        for p in path:
            x, y = self.uv2xy(*p)

            if _x is not None: # we're not at the start
                path_length += np.linalg.norm(np.array([_x, _y]) - np.array([x, y]))

            _x, _y = x, y
            self.trajectory.addPoint(Point(x, y, 0))

        rospy.loginfo("Found path with length of: "+str(round(path_length, 2)) + " m")
        rospy.loginfo("Found path in: "+str(round(self.time_to_plan, 4))+ " s")

        self.time_to_plan = 0

        # publish trajectory
        self.traj_pub.publish(self.trajectory.toPoseArray())

        # visualize trajectory Markers
        self.trajectory.publish_viz()

if __name__=="__main__":
    rospy.init_node("path_planning")
    pf = PathPlan()
    rospy.spin()
