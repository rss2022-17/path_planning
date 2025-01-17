#!/usr/bin/env python2
import numpy as np
import rospy
import sys
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, Pose

if __name__ == "__main__":
    rospy.init_node("autogoalset")
    rospy.loginfo("Node Initialized!")

    
    #Initialize test cases
    test_cases = dict()
    test_cases["straight_path1"] = [(-3.2, -.599),(-30.578, -.599)]
    test_cases["hallway_turn1"] = [(-3.2, -1.588), (-14.527, 11.937)]
    test_cases["hallway_turn2"] = [(-30.578, -.599), (-10.491, 16.034)]
    test_cases["across_map1"] = [(-20.065, 26.127), (-50.196, -0.434)]
    

    startpub  = rospy.Publisher("/initialpose", PoseWithCovarianceStamped, queue_size = 10)
    goalpub= rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size = 10)


    test_case_name = "default"
    if len(sys.argv) > 1:
        test_case_name = sys.argv[1]
    else:
        test_case_name = "straight_path1" #Change if necessary

    test_case = test_cases[test_case_name]



    start = test_case[0]
    end = test_case[1]
    rospy.sleep(0.75)

    rospy.loginfo("Working: after sleep")
    done = False
    
    initpose = PoseWithCovarianceStamped()
    initpose.header.stamp = rospy.Time.now()
    initpose.header.frame_id = "/map"
    initpose.pose.pose.position.x = start[0]
    initpose.pose.pose.position.y = start[1]
    initpose.pose.pose.orientation.w = 1.0
    startpub.publish(initpose)

    rospy.loginfo("Publishing initial pose, sleeping for 2 seconds")
    rospy.sleep(0.75)
    
    #Goal Pose
    goalpose = PoseStamped()
    goalpose.header.stamp = rospy.Time.now()
    goalpose.header.frame_id = "/map"
    goalpose.pose.position.x = end[0]
    goalpose.pose.position.y = end[1]
    goalpose.pose.orientation.w = 1.0
    rospy.loginfo(goalpose)
    goalpub.publish(goalpose)


    

