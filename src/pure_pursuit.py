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
        self.lookahead        = 0.5
        self.speed            = 1
        #self.wrap             = # FILL IN #
        self.wheelbase_length = 0.32#
        self.trajectory  = utils.LineTrajectory("/followed_trajectory")
        self.traj_sub = rospy.Subscriber("/trajectory/current", PoseArray, self.trajectory_callback, queue_size=1)
        self.drive_pub = rospy.Publisher("/drive", AckermannDriveStamped, queue_size=1)

        self.odom_sub = rospy.Subscriber("/pf/pose/odom", Odometry, self.odom_callback, queue_size=1)

    def trajectory_callback(self, msg):
        ''' Clears the currently followed trajectory, and loads the new one from the message
        '''
        print("Receiving new trajectory:", len(msg.poses), "points")
        self.trajectory.clear()
        self.trajectory.fromPoseArray(msg)
        self.trajectory.publish_viz(duration=0.0)

    def odom_callback(self, msg):
        try:
            car_x = msg.pose.pose.position.x #step 1, determine current location of vehicle
            car_y = msg.pose.pose.position.y
            car_point = np.array([car_x,car_y])
            points = np.array(self.trajectory.points)
            print(car_point.shape) 
            print(points.shape) 
            #step 2, find path point closest to vehicle
            #TODO: Update this to look at intermediate points on trajectory
            distances = np.linalg.norm(points-np.tile(car_point, (len(self.trajectory.distances),1)))
            #for points in points:
            #    distances.append(np.norm(point-car_point)) #((point[0]-car_x)**2 +  (point[1]-car_y)**2)**0.5)
            min_ind = np.argmin(distances) 
            min_point = points[min_ind]
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

                if t1<1 and t1>0: #t1 is necessarily further along the path, and should take precedence over t2
                    intersecting_points.append(P1 + t1*(V-P1))
                elif t2<1 and t2>0:
                    intersecting_points.append(P1 + t2*(V-P1))
            #TODO: figure out how to pick which point is the goal
            goal = intersecting_points[-1] #take last added point (furthest along path)
            
            #step 4, Transform the goal point to vehicle coordinates
            rot_mat = tf.transformations.quaternion_matrix((msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w))
            trans_mat = tf.transformations.translation_matrix((msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z))
            combined_mat = np.linalg.inv(np.matmul(trans_mat, rot_mat))
            translated_goal = np.transpose(np.array([goal[0], goal[1], 0, 1]))
            translated_goal = np.matmul(combined_mat, translated_goal)
    #        translated_goal = np.array([translated_goal[0], translated_goal[1]])

            #step 5, calculate the curvature
            angle = np.arctan2(translated_goal[1], translated_goal[0])
            L1 = np.linalg.norm(translated_goal[0:1])
            delta = np.arctan2(2*self.wheelbase_length*np.sin(angle), L1) #steering angle

            drive_cmd = AckermannDriveStamped()
            drive_cmd.header.stamp = rospy.Time.now()
            drive_cmd.header.frame_id = "/base_link_pf"
            drive_cmd.drive.steering_angle = delta
            drive_cmd.drive.speed = self.speed
            self.drive_pub.publish(drive_cmd)
        except Exception as err:
            drive_cmd = AckermannDriveStamped()
            drive_cmd.header.stamp = rospy.Time.now()
            drive_cmd.header.frame_id = "/base_link_pf"
            drive_cmd.drive.steering_angle = 0
            drive_cmd.drive.speed = 0
            self.drive_pub.publish(drive_cmd)


if __name__=="__main__":
    rospy.init_node("pure_pursuit")
    pf = PurePursuit()
    rospy.spin()
