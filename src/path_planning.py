#!/usr/bin/env python

import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped, PoseArray, Point32
from nav_msgs.msg import Odometry, OccupancyGrid
import rospkg
import time, os
from utils import LineTrajectory

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

        self.occupancy_cutoff = 0.9


    def map_cb(self, msg):
        self.occ_map = msg.data
        self.resolution = msg.info.resolution
        self.map_orientation = msg.info.origin.orientation
        self.map_position = msg.info.origin.position
        pass ## REMOVE AND FILL IN ##


    def odom_cb(self, msg):
        pass ## REMOVE AND FILL IN ##
        start_pt = msg.pose.pose.position # Point object

        x = start_pt.x
        y = start_pt.y

        start_point = np.array[[y, x]]
        start_point -= self.map_position
        start_point /= self.resolution

        self.start_point = start_point


    def goal_cb(self, msg):
        pass ## REMOVE AND FILL IN ##
        self.goal_point = msg.pose.position # Point object

        x = self.goal_point.x
        y = self.goal_point.y

        end_point = np.array[[y, x]]
        end_point -= self.map_position
        end_point /= self.resolution

        self.plan_path(self.start_point, end_point, self.occ_map)



    def plan_path(self, start_point, end_point, map):
        ## CODE FOR PATH PLANNING ##

        step_size = 1
        adjacent_squares = np.array([[-1, 1], [0, 1], [1, 1],
                                    [-1, 0],         [1, 0],
                                    [-1, -1],[0, -1],[1, -1]])


        def goal_dist(point):
            # returns float euclidean distance between point and end_point
            # assume point is just indices in the picture
            dist = np.linalg.norm(point - end_point)
            return dist

        def in_collision(point):
            # get the points we want to check sized by the step_size and centered on the given point
            check_points = np.indices(step_size, step_size).reshape(2, -1).T - np.array([int(step_size/2), int(step_size/2)]) + point
            occ_vals = map[check_points]

            # if any grid value is greater than cutoff, we're in collision
            return np.all(occ_vals > self.occupancy_cutoff)

        agenda = [[0, [start_point]]] # reverse sorted
        full_path = None

        while agenda:
            
            # Pull a path and point from the agenda
            last_vertex = agenda.pop(-1)
            cost_so_far = last_vertex[0]
            path_so_far = last_vertex[1]
            last_point = path_so_far[-1]

            if goal_dist(last_point) <= step_size**2:
                # if we're within a __circle__ of radius step_size to the goal
                full_path = path_so_far
                break

            # Create children from that last point
            # treat points as indices in the map
            children = last_point + step_size * adjacent_squares            

            # Prune collisions, update costs, and add to the agenda
            for c in children:
                if not in_collision(c):
                    # we can update cost and add it to the agenda
                    new_cost = cost_so_far + goal_dist(c) + np.linalg.norm(c - last_point)
                    new_path = path_so_far + [c]

                    agenda.append([new_cost, new_path])

            # Reverse sort agenda
            agenda.sort(reverse=True, key=lambda x: x[0])

        if full_path is not None:
            # it would seem that we found a path!
            pose_path = []

            for p in full_path:
                new_p = p * self.resolution
                new_p += self.map_position

                y, x = new_p

                new_point = Point32(x, y, 0)
                self.trajectory.addPoint(new_point)

            self.trajectory.addPoint(self.goal_point)

            # publish trajectory
            self.traj_pub.publish(self.trajectory.toPoseArray())

            # visualize trajectory Markers
            self.trajectory.publish_viz()

        else:
            # No path found
            rospy.loginfo("Path not found between the given start and goal poses!")


if __name__=="__main__":
    rospy.init_node("path_planning")
    pf = PathPlan()
    rospy.spin()
