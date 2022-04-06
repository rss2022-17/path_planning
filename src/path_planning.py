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



    def plan_path(self, start_point, end_point, map):
        ## CODE FOR PATH PLANNING ##

        self.trajectory.clear()
        rospy.loginfo("Starting path planning!!")

        step_size = rospy.get_param("~step_size", 9)
        rospy.loginfo("Using step size of "+str(step_size)+" from ~step_size parameter")

        # Initialize here so we can reuse it
        adjacent_squares = np.array([[-1, 1], [0, 1], [1, 1],
                                    [-1, 0],         [1, 0],
                                    [-1, -1],[0, -1],[1, -1]])

        # search step x step square around child point, initialized here
        collision_squares = np.indices((step_size, step_size)).reshape(2, -1).T - np.rint((step_size-1.0)/2.0)


        def goal_dist(point):
            # returns float euclidean distance between point and end_point
            # assume point is just indices in the picture
            dist = np.linalg.norm(point - end_point)
            return dist

        def in_collision(point):
            # get the points we want to check sized by the step_size and centered on the given point
            check_points = np.rint(collision_squares + np.rint(point)).astype(np.uint16)

            # we don't want any points outside of the occupancy bounds but that probably won't happen?
            # TODO

            # I couldn't figure out how to use numpy indexing/slicing so using a for loop 0.0
            for p in check_points.tolist():
                occ_val = map[p[0], p[1]]

                if occ_val > self.occupancy_cutoff:
                    # if we detect _any_ collision, stop early
                    return True

            return False

            # occ_vals = map[check_points]

            # if occ_vals.shape != check_points.shape:
            #     print("Occ vals shape: "+str(occ_vals.shape))
            #     print("Check points shape: "+str(check_points.shape))

            # assert occ_vals.shape == check_points.shape
            
            # if np.count_nonzero(occ_vals):
            #     print(occ_vals.shape)
            #     print(occ_vals)


            # # if any grid value is greater than cutoff, we're in collision
            # return np.all(occ_vals > self.occupancy_cutoff)


        ## Start the A* Path Planning
        agenda = [[0, [start_point]]] # reverse sorted
        visited = {(start_point[0,0], start_point[0,1])} # visited set of tuple indices
        full_path = None

        while agenda:
            # Pull a path and point from the agenda
            last_vertex = agenda.pop(-1)
            cost_so_far = last_vertex[0]
            path_so_far = last_vertex[1]
            last_point = path_so_far[-1]

            if goal_dist(last_point) <= step_size:
                # if we're within a __circle__ of radius step_size to the goal
                full_path = path_so_far
                rospy.loginfo("Path found")
                break

            # Create children from that last point
            # treat points as indices in the map
            children = last_point + step_size * adjacent_squares            

            # Prune collisions, update costs, and add to the agenda
            for c in children:
                if not in_collision(c):
                    # we can update cost and add it to the agenda
                    tup_vers = (c[0], c[1])
                    if tup_vers not in visited:
                        # we're not in collision or at a point we've visited
                        new_cost = cost_so_far + goal_dist(c) + np.linalg.norm(c - last_point) # apply the A* heuristic
                        new_path = path_so_far + [c.reshape(1,2)] # make the new path

                        # add the new point to the agenda and the visited set
                        agenda.append([new_cost, new_path])
                        visited.add(tup_vers)

            # Reverse sort agenda so we can quickly pop it
            agenda.sort(reverse=True, key=lambda x: x[0])

        # rospy.loginfo("Visited set: "+str(visited))


        ### DRAW MAP BORDERS!
        # n, m = map.shape
        # full_path = []
        # for idx in range(n):
        #     full_path.append(np.array([[idx, 0]]))
        #     full_path.append(np.array([[idx, m-1]]))

        # for idx in range(m):
        #     full_path.append(np.array([[0, idx]]))
        #     full_path.append(np.array([[n-1, idx]]))




        if full_path is not None:
            # we found a path!
            for p in full_path:
                # convert the [u, v] point to [y, x]
                new_p = p * self.resolution
                new_p += self.map_position

                y = new_p[0,0]
                x = new_p[0,1]

                new_point = Point32(x, y, 0)
                self.trajectory.addPoint(new_point)

            # finally, add the goal point
            self.trajectory.addPoint(self.goal_point)

            # publish trajectory
            self.traj_pub.publish(self.trajectory.toPoseArray())

            # visualize trajectory Markers
            self.trajectory.publish_viz()

        else:
            # No path found
            rospy.loginfo("Path not found between the given start and goal poses! :(")


if __name__=="__main__":
    rospy.init_node("path_planning")
    pf = PathPlan()
    rospy.spin()
