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

        self.pose_pub = rospy.Publisher("/planning/popped_pose", PoseStamped, queue_size=10)

        self.start_point = None
        self.goal_point = None

        self.occupancy_cutoff = 0.8
        self.occ_map = None

        self.save_trajs = False
        self.num_paths_made = 0


    def quaternion_rotation_matrix(self, Q):
        """
        Credit to https://automaticaddison.com/how-to-convert-a-quaternion-to-a-rotation-matrix/

        Covert a quaternion into a full three-dimensional rotation matrix.
    
        Input
        :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3) 
    
        Output
        :return: A 3x3 element matrix representing the full 3D rotation matrix. 
                This rotation matrix converts a point in the local reference 
                frame to a point in the global reference frame.
        """
        # Extract the values from Q
        q0 = Q[0] # w
        q1 = Q[1] # x
        q2 = Q[2] # y
        q3 = Q[3] # z
        
        # First row of the rotation matrix
        r00 = 2 * (q0 * q0 + q1 * q1) - 1
        r01 = 2 * (q1 * q2 - q0 * q3)
        r02 = 2 * (q1 * q3 + q0 * q2)
        
        # Second row of the rotation matrix
        r10 = 2 * (q1 * q2 + q0 * q3)
        r11 = 2 * (q0 * q0 + q2 * q2) - 1
        r12 = 2 * (q2 * q3 - q0 * q1)
        
        # Third row of the rotation matrix
        r20 = 2 * (q1 * q3 - q0 * q2)
        r21 = 2 * (q2 * q3 + q0 * q1)
        r22 = 2 * (q0 * q0 + q3 * q3) - 1
        
        # 3x3 rotation matrix
        rot_matrix = np.array([[r00, r01, r02],
                            [r10, r11, r12],
                            [r20, r21, r22]])
                                
        return rot_matrix


    def map_cb(self, msg):

        # Store all of the map data into instance variables
        self.occ_map = np.array(msg.data).reshape((msg.info.height, msg.info.width))

        self.u_max = msg.info.height - 1
        self.v_max = msg.info.width - 1

        mo = msg.info.origin.orientation

        self.resolution = msg.info.resolution
        self.map_orientation = msg.info.origin.orientation
        self.map_position = np.array([[msg.info.origin.position.y, msg.info.origin.position.x]])

        ## Create a rotation matrix for the map's pose
        self.map_rot_mat = self.quaternion_rotation_matrix([mo.w, mo.x, mo.y, mo.z])
        self.map_rot_mat = np.vstack((self.map_rot_mat, np.zeros((1,3)))) # first make it (4 x 3)
        # add the [x, y, z, 1] vector on the right side to make it (4 x 4)
        self.map_rot_mat = np.hstack((self.map_rot_mat, np.array([[msg.info.origin.position.x, msg.info.origin.position.y, msg.info.origin.position.z, 1]]).T))

        self.inv_map_rot_mat = np.linalg.inv(self.map_rot_mat)

        rospy.loginfo("map initialized!!")
        rospy.loginfo("map shape: "+str(self.occ_map.shape))
        rospy.loginfo("map resolution: "+str(self.resolution))
        rospy.loginfo("map orientation: "+str(self.map_orientation))
        rospy.loginfo("map position: "+str(self.map_position))
        rospy.loginfo("map rotation matrix:\n"+str(self.map_rot_mat))

        rospy.loginfo("nonzero indices in map: "+str(np.nonzero(self.occ_map)))

    def odom_cb(self, msg):
        start_pt = msg.pose.pose.position # Point object

        # if the map isn't set, we can't use it to initialize start points
        if self.occ_map is None: return

        np_start = np.array([[start_pt.x, start_pt.y]])
        # x = start_pt.x
        # y = start_pt.y

        # # swap x, y then convert to u, v
        # start_point = (np.array([[y, x]]) - self.map_position) / self.resolution

        self.start_point = np.array([self.convert_x_to_pixels(np_start)])


    def goal_cb(self, msg):
        if self.occ_map is None: return

        if self.start_point is None: 
            rospy.loginfo("Could not plan a trajectory because there's no odom!"+\
                "\n\tMake sure that you're publishing to "+self.odom_topic)
            return

        self.goal_point = msg.pose.position # Point object

        np_end = np.array([[self.goal_point.x, self.goal_point.y]])
        
        # x = self.goal_point.x
        # y = self.goal_point.y

        # swap x, y then convert to u, v
        # end_point = (np.array([[y, x]]) - self.map_position) / self.resolution

        end_point = np.array([self.convert_x_to_pixels(np_end)])

        # if we're setting the goal, we should already have a start point so plan the path!
        self.plan_path(self.start_point, end_point, self.occ_map)


    def convert_pixels_to_x(self, p):
        """
        p is a (1,2) numpy 2darray in pixel space (indices)

        returns tuple (x, y)
        """
        # make [v, u, 0, 1]
        new_p = np.array([[p[0, 1], p[0, 0], 0, 1.0/self.resolution]]).T
        new_p = new_p * self.resolution

        # rotate and translate to map space
        new_p = np.matmul(self.map_rot_mat, new_p)

        # extract out proper x,y (we flip when init new_p)
        x = new_p[0, 0]
        y = new_p[1, 0]

        return x, y


    def convert_x_to_pixels(self, pos):
        """
        pos is a (1,2) numpy 2darray for map space (meters)

        returns tuple (u, v)
        """
        # make [x, y, 0, 1] (not flipped yet) and rotate+translate into pixel frame
        new_p = np.array([[pos[0, 0], pos[0, 1], 0.0, 1.0]]).T
        new_p = np.matmul(self.inv_map_rot_mat, new_p)

        # true divide by resolution to get to pixel resolution
        new_p = np.divide(new_p, self.resolution)

        # extract out proper v, u by flipping
        v = new_p[0, 0]
        u = new_p[1, 0]

        return u, v


    def plan_path(self, start_point, end_point, map):
        ## CODE FOR PATH PLANNING ##

        self.trajectory.clear()
        rospy.loginfo("Starting path planning!!")

        step_size = rospy.get_param("/lab6/step_size", 3)
        rospy.loginfo("Using step size of "+str(step_size)+" from parameter: /lab6/step_size")

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
            _u = np.clip(check_points[:, 0], 0, self.u_max).reshape(-1,1) # u
            _v = np.clip(check_points[:, 1], 0, self.v_max).reshape(-1,1) # v

            true_check_points = np.hstack((_u, _v))

            # I couldn't figure out how to use numpy indexing/slicing so using a for loop 0.0
            for p in true_check_points.tolist():
                # access the occupancy grid and pull out the values
                occ_val = map[p[0], p[1]]

                if occ_val > self.occupancy_cutoff:
                    # if we detect _any_ collision, stop early
                    return True

            return False

        ## Start the A* Path Planning
        agenda = [[goal_dist(start_point), 0, [start_point]]] # reverse sorted -- [estimated cost, cost_so_far, path]
        visited = set() # visited set of tuple indices
        full_path = None

        # used to visualize what point the algo is currently considering (viz search space)
        current_pose = PoseStamped()
        current_pose.header.frame_id = "map"

        while agenda:
            # Pull a path, point, and it's cost from the agenda
            last_vertex = agenda.pop(-1)
            cost_so_far = last_vertex[1]
            path_so_far = last_vertex[2]
            last_point = path_so_far[-1]

            # tuple version bc it's hashable
            tup_vers = (last_point[0,0], last_point[0,1])

            # if we've already pulled off a point, we found a more optimal path to it earlier
            if tup_vers in visited: continue
            visited.add(tup_vers)

            # visualize the current point we're considering
            x, y = self.convert_pixels_to_x(last_point)
            current_pose.pose.position.x = x
            current_pose.pose.position.y = y
            self.pose_pub.publish(current_pose)


            if np.linalg.norm(last_point - end_point) <= step_size:
                # if we're within a __circle__ of radius step_size to the goal
                full_path = path_so_far
                rospy.loginfo("Path found! :)")
                break

            # Create children from that last point; treat points as indices in the map
            children = last_point + step_size * adjacent_squares

            _u = np.clip(children[:, 0], 0, self.u_max).reshape(-1,1) # u
            _v = np.clip(children[:, 1], 0, self.v_max).reshape(-1,1) # v

            true_check_points = np.hstack((_u, _v))          

            # Prune collisions, update costs, and add to the agenda
            for c in true_check_points:
                if not in_collision(c):
                    # we can update cost and add it to the agenda
                    tup_vers = (c[0], c[1])
                    if tup_vers not in visited:
                        # we're not in collision or at a point we've visited
                        dist_to_goal = np.linalg.norm(c - end_point)

                        new_cost_so_far = cost_so_far + np.linalg.norm(c - last_point)
                        new_estimated_cost = new_cost_so_far + dist_to_goal # apply the A* heuristic
                        new_path = path_so_far + [c.reshape(1,2)] # make the new path

                        # add the new point to the agenda
                        agenda.append([new_estimated_cost, new_cost_so_far, new_path])

            # Reverse sort agenda so we can quickly pop it
            agenda.sort(reverse=True, key=lambda x: x[0])


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
                # p numpy (1,2)
                s_x, s_y = self.convert_pixels_to_x(p)

                new_point = Point32(s_x, s_y, 0)
                self.trajectory.addPoint(new_point)

            # finally, add the goal point
            self.trajectory.addPoint(self.goal_point)

            # publish trajectory
            self.traj_pub.publish(self.trajectory.toPoseArray())

            # visualize trajectory Markers
            self.trajectory.publish_viz()

            # save the trajectory
            if self.save_trajs:
                self.num_paths_made += 1
                self.trajectory.save("/home/racecar/racecar_ws/src/path_planning/trajectories/stata_basement_trajectory_"+str(self.num_paths_made)+".traj")

        else:
            # No path found
            rospy.loginfo("Path not found between the given start and goal poses! :(")


if __name__=="__main__":
    rospy.init_node("path_planning")
    pf = PathPlan()
    rospy.spin()
