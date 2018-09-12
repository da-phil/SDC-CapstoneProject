#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
import numpy as np
from scipy.spatial import KDTree


STATE_COUNT_THRESHOLD = 3

class TLDetector(object):
    def __init__(self):
        self.initialized = False        
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.waypoints_tree = None
        self.camera_image = None
        self.lights = []
        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        config_string = rospy.get_param("/traffic_light_config")
        threshold = rospy.get_param('~threshold', 0.2)
        hw_ratio = rospy.get_param('~hw_ratio', 1.5)
        self.sim_testing = bool(rospy.get_param("~sim_testing", True))

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        self.config = yaml.load(config_string)
        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)
        self.light_image_bbox = rospy.Publisher('/image_color_bbox', Image, queue_size=1)
        self.bridge = CvBridge()

        self.light_classifier = TLClassifier(threshold, hw_ratio, self.sim_testing)
        self.listener = tf.TransformListener()

        self.initialized = True
        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints
        waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y]
                         for waypoint in waypoints.waypoints]
        self.waypoints_tree = KDTree(waypoints_2d)

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        if not self.initialized:
            return

        self.has_image = True
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    def get_closest_waypoint(self, position_xy):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            position_xy (list): position in x and y to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        assert len(position_xy) == 2, "position_xy must have a length of 2"
        return self.waypoints_tree.query(position_xy, 1)[1]

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if not self.has_image:
            self.prev_light_loc = None
            return False

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "rgb8")

        # Get classification
        signal, bbox = self.light_classifier.get_classification(cv_image)

        # Publish detected bounding box area of traffic light for debug purpose, but only if there was a detection
        if not np.array_equal(bbox, np.zeros(4)):
            try:
                ros_img = self.bridge.cv2_to_imgmsg(np.asarray(cv_image)[bbox[0]:bbox[2], bbox[1]:bbox[3]], encoding="rgb8")
                self.light_image_bbox.publish(ros_img)
            except CvBridgeError as e:
                print(e)

        return signal

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        closest_light = None
        light_wp_idx = None

        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        if self.pose and self.waypoints:
            car_wp_idx = self.get_closest_waypoint([self.pose.pose.position.x, self.pose.pose.position.y])

            # find the closest visible traffic light (if one exists)
            diff = len(self.waypoints.waypoints)
            for i, light in enumerate(self.lights):
                # get stop line waypoint idx
                line = stop_line_positions[i]
                temp_wp_idx = self.get_closest_waypoint(line)
                # find closest stop line waypoint idx
                d = temp_wp_idx - car_wp_idx
                if 0 <= d < diff:
                    diff = d
                    closest_light = light
                    light_wp_idx = temp_wp_idx

        if closest_light:
            state = self.get_light_state(closest_light)

            # show verbose output 100 waypoints before traffic lights
            if car_wp_idx+100 > light_wp_idx:
                rospy.logwarn('Found traffic light with state {} at waypoint_idx {} (car_cur_wp: {})'.format(self.light_classifier.category_index[state]["name"],
                                                                                                             light_wp_idx, car_wp_idx))
            return light_wp_idx, state

        #self.waypoints = None
        return closest_light, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
