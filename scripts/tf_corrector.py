#!/usr/bin/env python
import rospy
from nav_msgs.msg import Odometry
from tf2_ros import TFMessage
from geometry_msgs.msg import TransformStamped

import numpy as np

last_odom_msg = None
def odom_msg_callback(msg):
    global last_odom_msg
    last_odom_msg = msg

def main():
    global last_odom_msg

    rospy.init_node('tf_corrector', anonymous=True)

    static_x = rospy.get_param("~static_x")
    static_y = rospy.get_param("~static_y")
    static_z = rospy.get_param("~static_z")
    frame_id = rospy.get_param("~frame_id")
    rospy.Subscriber("~input_odom", Odometry, odom_msg_callback)
    tf_pub = rospy.Publisher("/tf", TFMessage, queue_size=100)

    rate = rospy.Rate(1000)
    while not rospy.is_shutdown():
        if last_odom_msg is None:
            rospy.logwarn_throttle(1.0, "waiting for odom data")
            try:
                rate.sleep()
            except:
                pass
            continue

        tf = TransformStamped()
        tf.header.frame_id = last_odom_msg.header.frame_id
        tf.header.stamp = last_odom_msg.header.stamp
        tf.child_frame_id = frame_id
        tf.transform.translation.x = static_x
        tf.transform.translation.y = static_y
        tf.transform.translation.z = static_z
        tf.transform.rotation = last_odom_msg.pose.pose.orientation
        tf_msg = TFMessage()
        tf_msg.transforms.append(tf)
        tf_pub.publish(tf_msg)
        last_odom_msg = None


if __name__ == '__main__':
    main()
