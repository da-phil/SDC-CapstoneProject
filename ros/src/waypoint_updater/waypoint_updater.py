#!/usr/bin/env python

import rospy
import numpy as np
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, TwistStamped
from styx_msgs.msg import Lane, Waypoint

import math
from scipy.spatial import KDTree

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 100 # Number of waypoints we will publish. You can change this number


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater', log_level=rospy.DEBUG)

        # only start when waypoint publisher started!
        #msg = rospy.wait_for_message('/base_waypoints', Lane)

        self.decel_limit = rospy.get_param('~decel_limit', -5)
        self.accel_limit = rospy.get_param('~accel_limit', 1.)
        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)
        self.waypoints_2d = None
        self.waypoint_tree = None
        self.base_waypoints = None
        self.pose = None
        self.cur_velocity = None
        self.traffic_waypoint_idx = None
        self.obstacle_waypoint_idx = None

        self.stop_commanded = False
        self.cur_stop_waypoints = []
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/current_velocity', TwistStamped, self.velocity_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        rospy.Subscriber('/obstacle_waypoint', Int32, self.obstacle_cb)

        self.start()
        
    def start(self):
        rospy.loginfo('Starting start() function')
        rate = rospy.Rate(25)
        while not rospy.is_shutdown():
            if self.pose and self.base_waypoints:
                # get closest waypoint
                closest_waypoint_idx = self.get_closest_waypoint_idx()
                if closest_waypoint_idx != None:
                    self.publish_waypoints(closest_waypoint_idx)
            rate.sleep()

    def get_closest_waypoint_idx(self):
        if self.waypoint_tree == None:
            return None

        x = self.pose.pose.position.x
        y = self.pose.pose.position.y
        
        closest_waypoint_idx = self.waypoint_tree.query([x, y], 1)[1]

        # check if closest waypoint is ahead or behind the car
        closest_coord = self.waypoints_2d[closest_waypoint_idx]
        prev_coord = self.waypoints_2d[closest_waypoint_idx - 1]

        cur_vec  = np.asarray(closest_coord)
        prev_vec = np.array(prev_coord)
        pos_vec  = np.array([x, y])

        val = np.dot(cur_vec - prev_vec, pos_vec - cur_vec)
        if val > 0:
            closest_waypoint_idx = (closest_waypoint_idx + 1) % len(self.waypoints_2d)
        return closest_waypoint_idx

    def publish_waypoints(self, closest_waypoint_idx):
        lane = Lane()
        farthest_idx   = closest_waypoint_idx+LOOKAHEAD_WPS
        base_waypoints = self.base_waypoints.waypoints[closest_waypoint_idx:farthest_idx]

        if 0 <= self.traffic_waypoint_idx <= farthest_idx:
            #rospy.loginfo("Stop command set!")
            lane.waypoints = self.decelerate_waypoints(base_waypoints, closest_waypoint_idx)
        else:
            #rospy.loginfo("Stop command reset!")
            self.stop_commanded = False
            lane.waypoints = base_waypoints

        self.final_waypoints_pub.publish(lane)

    def decelerate_waypoints(self, waypoints, closest_waypoint_idx):
        
        velocity_init = waypoints[0].twist.twist.linear.x
        stop_idx = max(self.traffic_waypoint_idx - closest_waypoint_idx - int(self.cur_velocity.twist.linear.x * 2), 0)
        #if self.stop_commanded:
        #    return self.cur_stop_waypoints

        """
        stop_idx = self.traffic_waypoint_idx - closest_waypoint_idx - int(self.cur_velocity.twist.linear.x * 1.5)
        # wrap around from 0 to end
        if stop_idx < 0:
            stop_idx += len(waypoints)
        """
        rospy.loginfo("stop_wp: {}, rel_stop_wp before: {}, closest_idx: {}".format(self.traffic_waypoint_idx, stop_idx, closest_waypoint_idx))        
        #rospy.loginfo("cur_wp: {}, stop_wp: {}, len(basic_wp): {}".format(0, stop_idx, len(waypoints)))
        self.cur_stop_waypoints = []
        coeff = velocity_init / (self.distance(waypoints, 0, stop_idx) + 1e-3)
        for i, wp in enumerate(waypoints):
            p = Waypoint()
            p.pose = wp.pose
            dist = self.distance(waypoints, i, stop_idx)
            #velocity_reduced = coeff*dist
            velocity_reduced = math.sqrt(2 * abs(self.decel_limit) * dist)
            if velocity_reduced < 1.0:
                velocity_reduced = 0.0
            #rospy.loginfo("Setting WP {} lower velocity: {} (old: {})".format(i, velocity_reduced, wp.twist.twist.linear.x))
            p.twist.twist.linear.x = min(velocity_reduced, wp.twist.twist.linear.x)
            self.cur_stop_waypoints.append(p)
        #rospy.loginfo("All new WPs:\n{}".format(self.cur_stop_waypoints))
        self.stop_commanded = True
        return self.cur_stop_waypoints

    def pose_cb(self, msg):
        self.pose = msg

    def velocity_cb(self, msg):
        self.cur_velocity = msg

    def waypoints_cb(self, waypoints):
        #self.final_waypoints_pub.publish(waypoints)
        self.base_waypoints = waypoints
        if self.waypoints_2d == None:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y]
                                 for waypoint in waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        self.traffic_waypoint_idx = int(msg.data)

    def obstacle_cb(self, msg):
        self.obstacle_waypoint_idx = int(msg.data)

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
