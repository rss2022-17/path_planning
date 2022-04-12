#!/usr/bin/env python2
import numpy as np
import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, Pose

if __name__ == "__main__":
    rospy.init_node("autogoalset")
    start_points = [(-3.2, -1.588), (-30.578, -.599)]
    end_points = [(-14.527, 11.937), (-10.491, 16.034)]
    startpub  = rospy.Publisher("/pf/pose/odom", Odometry, queue_size = 10)
    goalpub= rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size = 10)
    test = 0
    start = start_points[test]
    end = end_points[test]
    rospy.sleep(10.)

    done = False
    while not rospy.is_shutdown():
        #Start Pose
        if done == True:
            break
        initpose = Odometry()
        initpose.header.stamp = rospy.Time.now()
        initpose.header.frame_id = "/map"
        initpose.pose.pose.position.x = start[0]
        initpose.pose.pose.position.y = start[1]
        initpose.pose.pose.orientation.w = 1.0
        startpub.publish(initpose)

        #Goal Pose
        goalpose = PoseStamped()
        goalpose.header.stamp = rospy.Time.now()
        goalpose.header.frame_id = "/map"
        goalpose.pose.position.x = end[0]
        goalpose.pose.position.y = end[1]
        goalpose.pose.orientation.w = 1.0
        rospy.loginfo(goalpose)
        goalpub.publish(goalpose)
        done = True
    rospy.spin()


    

