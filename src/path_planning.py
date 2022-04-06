#!/usr/bin/env python

from socket import INADDR_BROADCAST
import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped, PoseArray, Point32
from nav_msgs.msg import Odometry, OccupancyGrid
import rospkg
import time, os
from utils import LineTrajectory

class Graph:
    ''' Tree graph that is being generated '''
    def __init__(self,start_point,end_point, x_min, x_max, y_min, y_max):
        self.start_point = start_point
        self.end_point = end_point

        self.adjacency = {start_point:[]}
        self.parents = {start_point: None}

        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
    
    def random_position(self):
        rand_x = np.random.randint(self.x_min, self.x_max+1)
        rand_y = np.random.randint(self.y_min, self.y_max+1)

        return np.array([[rand_y, rand_x]])
    
    def nearest(self, new_point):
        min_dist = float('inf')
        min_point = None

        for point in self.adjacency.keys():
            dist = np.linalg.norm(new_point - point)
            if dist < min_dist:
                min_dist = dist
                min_point = point
        
        return min_point
    
    def edge(self, parent_point, child_point):
        self.adjacency[parent_point].append(child_point)
        self.adjacency[child_point] = []
        self.parents[child_point] = parent_point

    
        
class PathPlan(object):
    """ Listens for goal pose published by RViz and uses it to plan a path from
    current car pose.
    """
    def __init__(self):
        self.odom_topic = rospy.get_param("~odom_topic")
        self.map_sub = rospy.Subscriber("/map", OccupancyGrid, self.map_cb)
        self.trajectory = LineTrajectory("/planned_trajectory")
        self.goal_sub = rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.goal_cb, queue_size=10)
        self.traj_pub = rospy.Publisher("/trajectory/current", PoseArray, queue_size=10)
        self.odom_sub = rospy.Subscriber(self.odom_topic, Odometry, self.odom_cb)

        self.occupancy_cutoff = 0.8
        self.occ_map = None

    def map_cb(self, msg):

        # Store all of the map data into instance variables
        self.occ_map = np.array(msg.data).reshape((msg.info.height, msg.info.width))
        self.resolution = msg.info.resolution
        self.map_orientation = msg.info.origin.orientation
        self.map_position = np.array([[msg.info.origin.position.y, msg.info.origin.position.x]])

        rospy.loginfo("map initialized!!")
        rospy.loginfo("map shape: "+str(self.occ_map.shape))
        rospy.loginfo("map resolution: "+str(self.resolution))
        rospy.loginfo("map orientation: "+str(self.map_orientation))
        rospy.loginfo("map position: "+str(self.map_position))

        rospy.loginfo("nonzero indices in map: "+str(np.nonzero(self.occ_map)))

    def odom_cb(self, msg):
        start_pt = msg.pose.pose.position # Point object

        # if the map isn't set, we can't use it to initialize start points
        if self.occ_map is None: return

        x = start_pt.x
        y = start_pt.y

        # swap x, y then convert to u, v
        start_point = np.array([[y, x]])
        start_point -= self.map_position
        start_point /= self.resolution

        self.start_point = start_point

    def goal_cb(self, msg):
        if self.occ_map is None: return

        self.goal_point = msg.pose.position # Point object

        x = self.goal_point.x
        y = self.goal_point.y

        # swap x, y then convert to u, v
        end_point = np.array([[y, x]])
        end_point -= self.map_position
        end_point /= self.resolution


        # if we're setting the goal, we should already have a start point so plan the path!
        self.plan_path(self.start_point, end_point, self.occ_map)

    def in_obstacle(self,point):
        # collision square?
        occ_val = map[point[0],point[1]]
        
        if occ_val > self.occupancy_cutoff:
            return True
        return False
    
    def through_obstacle(self,point_1, point_2):
        m = (point_1[0,0] - point_2[0,0])/(point_1[0,1] - point_2[0,1])

        # y = mx + b --> y - mx = b

        b = point_1[0,0] - m * point_1[0,1]

        start = min(point_1[0,1], point_2[0,1])
        stop = max(point_1[0,1], point_2[0,1])

        for x in range(np.ceil(start), stop):
            y = m*x + b
            point = np.array([[y, x]])
            if self.in_obstacle(point):
                return True
        return False

    def goal(self, point):
        threshold = 2
        dist = np.linalg.norm(point - self.end_point)
        if dist < threshold:
            return True
        return False
    
    def rrt(self, graph):
        counter = 0
        lim = 100  # hyperparamter

        G = graph

        while counter < lim:
            new_point = G.random_position()

            if self.in_obstacle(new_point):
                continue

            nearest_point = G.nearest(new_point)

            if self.through_obstacle(nearest_point, new_point):
                continue

            G.edge(nearest_point, new_point)

            if G.goal(new_point):
                return (new_point,G)
        
        return (None,G)

    def trace_back(self,graph):
        G = graph

        path = []
        # trace back
        
        reached_start = False
        current_node = end_point
        path.append(end_point)
        while not reached_start:
            current_parent = G.parents[current_node]
            path.append(current_parent)
            current_node = current_parent

            if current_parent == G.start_point:
                reached_start = True
        
        path = np.array(path)
        path = np.flip(path)

        return path

    def plan_path(self, start_point, end_point, map):
        ## CODE FOR PATH PLANNING -- RRT ##

        G = Graph(start_point, end_point, self.map_position[0,1]/self.resolution, self.occ_map.shape[1]/self.resolution, self.map_position[0,0]/self.resolution, self.occ_map.shape[0]/self.resolution)
        
        # get RRT graph !!!
        end_point, path_graph = self.rrt(G)

        if end_point is None:
            rospy.loginfo("Path not found between the given start and goal poses! :(")
        
        else:
            path = self.trace_back(path_graph)

            for p in path:
                # convert [u,v] --> [y,x]
                new_p = p*self.resolution
                new_p += self.map_position

                y = new_p[0,0]
                x = new_p[0,1]

                new_point = Point32(x,y,0)
                self.trajectory.addPoint(new_point)
            
            # add goal point
            self.trajectory.addPoint(self.goal_point)

            # publish trajectory
            self.traj_pub.publish(self.trajectory.toPoseArray())

            # visualize trajectory Markers
            self.trajectory.publish_viz()
        
if __name__=="__main__":
    rospy.init_node("path_planning")
    pf = PathPlan()
    rospy.spin()
