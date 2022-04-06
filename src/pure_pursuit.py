#!/usr/bin/env python

import rospy
import numpy as np
import time
import utils
import tf

from geometry_msgs.msg import PoseArray, PoseStamped
from visualization_msgs.msg import Marker
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry

class PurePursuit(object):
    """ Implements Pure Pursuit trajectory tracking with a fixed lookahead and speed.
    """
    def __init__(self):
        self.odom_topic       = rospy.get_param("~odom_topic")
        self.lookahead        = # FILL IN #
        self.speed            = # FILL IN #
        self.wrap             = # FILL IN #
        self.wheelbase_length = # FILL IN #
        self.trajectory  = utils.LineTrajectory("/followed_trajectory")
        self.traj_sub = rospy.Subscriber("/trajectory/current", PoseArray, self.trajectory_callback, queue_size=1)
        self.drive_pub = rospy.Publisher("/drive", AckermannDriveStamped, queue_size=1)

        self.odom_sub = rospy.Subscriber("/base_link_pf", Odometry, self.odom_callback, queue_size=1)

    def trajectory_callback(self, msg):
        ''' Clears the currently followed trajectory, and loads the new one from the message
        '''
        print "Receiving new trajectory:", len(msg.poses), "points"
        self.trajectory.clear()
        self.trajectory.fromPoseArray(msg)
        self.trajectory.publish_viz(duration=0.0)

    def odom_callback(self, msg):
        car_x = msg.pose.pose.position.x #step 1, determine current location of vehicle
        car_y = msg.pose.pose.position.y
      
        points = self.trajectory.points
        
        #step 2, find path point closest to vehicle
        distances = []
        for points in points:
            distances.append(((point[0]-car_x)**2 +  (point[1]-car_y)**2)**0.5)
        min_ind = np.argmin(distances) 
        min_point = points(min_ind)
        min_point_dist = self.trajectory.distance_along_trajectory(min_ind)
        #step 3, find goal point
        intersecting_points = []
        Q = [car_x, car_y]
        r = self.lookahead
        for i in range(min_ind, len(points)-1): #-1 because we're looking at segments between points
            P1 = points[i]
            V = points[i+1]
            a = np.dot(V,V)
            b = 2* np.dot(V, P1-Q)
            c = np.dot(P1, P1) + np.dot(Q,Q) - 2*np.dot(P1, Q) - r**2
            disc = b**2 - 4*a*c
            if disc<0:
                continue
            sqrt_disc = np.sqrt(disc)
            t1 = (-b + sqrt_disc)/(2.0*a)
            t2 = (-b - sqrt_disc)/(2.0*a)
            if t1<1 and t1>0:
                intersecting_points.append(P1 + t1*(P2-P1))
            if t2<1 and t2>0:
                intersecting_points.append(P1 + t2*(P2-P1))
        #TODO: figure out how to pick which point is the goal
        

if __name__=="__main__":
    rospy.init_node("pure_pursuit")
    pf = PurePursuit()
    rospy.spin()
