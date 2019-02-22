#!/usr/bin/env python
import rospy
# import tf
# from tf.transformations import quaternion_from_euler
import rosbag
from image_geometry import PinholeCameraModel
from tf2_ros import TFMessage
from tf2_ros import Buffer
from tf2_ros import TransformListener
from tf2_geometry_msgs import PointStamped
from rosgraph_msgs.msg import Clock

import pickle
import numpy as np
from numpy import cos
from numpy import sin
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import csv
import copy

import cv2
from cv_bridge import CvBridge

def load_csv_data(csv_fname):
    rospy.loginfo("Using CSV file {:s}".format(csv_fname))

    n_pos = sum(1 for line in open(csv_fname)) - 1
    positions = np.zeros((n_pos, 3))
    times = np.zeros((n_pos,))
    it = 0
    with open(csv_fname, 'r') as fhandle:
        first_loaded = False
        csvreader = csv.reader(fhandle, delimiter=',')
        for row in csvreader:
            if not first_loaded:
                first_loaded = True
                continue
            positions[it, :] = np.array([float(row[0]), float(row[1]), float(row[2])])
            times[it] = float(row[3])
            it += 1
    return (positions, times)

def msg_to_pos(tf_buffer, msg, to_frame_id):
    if msg is None:
        return None

    ps = PointStamped()
    ps.header = msg.header
    ps.point.x = msg.pose.pose.position.x
    ps.point.y = msg.pose.pose.position.y
    ps.point.z = msg.pose.pose.position.z

    try:
        ps2 = tf_buffer.transform(ps, to_frame_id)
    except Exception as e:
        rospy.logwarn('Exception during TF from {:s} to {:s}: {:s}'.format(msg.header.frame_id, to_frame_id, e))
        return None
    return np.array([ps2.point.x, ps2.point.y, ps2.point.z])

def pos_to_pxpos(pos, cmodel, offset=(0, 0)):
    if pos is None:
        return None
    u, v = cmodel.project3dToPixel(pos)
    u = int(round(u + offset[0]))
    v = int(round(v + offset[1]))
    return (u, v)

def load_rosbag_cinfo(bag_fname, topic):
    rospy.loginfo("Using rosbag {:s}".format(bag_fname))
    bag = rosbag.Bag(bag_fname)
    n_msgs = bag.get_message_count(topic_filters=topic)
    if n_msgs == 0:
        rospy.logerr("No messages from topic {:s} in bag".format(topic))
    else:
        rospy.loginfo("Loading {:d} messages".format(n_msgs))
    cinfo = None
    for topic, msg, cur_stamp in bag.read_messages(topics=topic):
        cinfo = msg
        break
    return cinfo

def load_rosbag_msgs(bag_fname, topic, skip_time=0, skip_time_end=0, start_time=None, end_time=None):
    rospy.loginfo("Using rosbag {:s}".format(bag_fname))
    bag = rosbag.Bag(bag_fname)
    n_msgs = bag.get_message_count(topic_filters=topic)
    if n_msgs == 0:
        rospy.logerr("No messages from topic {:s} in bag".format(topic))
    else:
        rospy.loginfo("Loading {:d} messages".format(n_msgs))
    msgs = n_msgs*[None]

    skip = rospy.Duration.from_sec(skip_time)
    if start_time is None:
        start_time = rospy.Time.from_sec(bag.get_start_time()) + skip
    if end_time is None:
        skip_end = rospy.Duration.from_sec(skip_time_end)
        end_time = rospy.Time.from_sec(bag.get_end_time()) - skip_end
    it = 0
    for topic, msg, cur_stamp in bag.read_messages(topics=topic, start_time=start_time, end_time=end_time):
        if rospy.is_shutdown():
            break
        msgs[it] = msg
        it += 1
    return msgs[0:it]

class msg:
    def __init__(self, time, positions):
        self.time = time
        self.positions = positions

def process_msgs(msgs, cinfo, shift):
    ret = len(msgs)*[None]

    cam_model = PinholeCameraModel()
    if cinfo is not None:
        cam_model.fromCameraInfo(cinfo)

    offset = None
    for it in range(0, len(msgs)):
        cur_msg = msgs[it]
        out_msg = msg(cur_msg.header.stamp.to_sec(), list())
        if cinfo is not None:
            # rospy.loginfo("using cinfo")
            for det in cur_msg.points:
                xyz = np.array([det.x, det.y, det.z])
                if offset is None:
                    offset = xyz
                xyz = xyz - offset + shift
                # xyz = np.dot(xyz, R)
                out_msg.positions.append(xyz)
        else:
            # rospy.loginfo("not using cinfo")
            x = cur_msg.pose.pose.position.x
            y = cur_msg.pose.pose.position.y
            z = cur_msg.pose.pose.position.z
            out_msg.positions.append(np.array([x, y, z]))
        ret[it] = out_msg
    return ret


def find_closest(stamp, msgs, max_dt=float('Inf')):
    closest_msg = None
    closest_diff = float('Inf')
    for msg in msgs:
        cur_stamp = msg.header.stamp
        cur_diff = abs((stamp - cur_stamp).to_sec())
        if cur_diff <= closest_diff:
            closest_msg = msg
            closest_diff = cur_diff
        else:
            break
    if cur_diff > max_dt:
        closest_msg = None
    return closest_msg


def find_closest_pos(stamp, poss, stamps, frame_id, to_frame_id, tf_buffer):
    closest_pos = None
    closest_diff = float('Inf')
    for it in range(0, len(poss)):
        cur_stamp = rospy.Time.from_sec(stamps[it])
        cur_diff = abs((stamp - cur_stamp).to_sec())
        if cur_diff <= closest_diff:
            closest_pos = poss[it]
            closest_diff = cur_diff
        else:
            break

    ps = PointStamped()
    ps.header.stamp = stamp
    ps.header.frame_id = frame_id
    ps.point.x = closest_pos[0]
    ps.point.y = closest_pos[1]
    ps.point.z = closest_pos[2]

    try:
        ps2 = tf_buffer.transform(ps, to_frame_id)
    except Exception as e:
        rospy.logwarn('Exception during TF from {:s} to {:s}: {:s}'.format(frame_id, to_frame_id, e))
        return None

    return np.array([ps2.point.x, ps2.point.y, ps2.point.z])


# def calc_statistics(positions1, times1, msgs, FP_error):
#     TPs = 0
#     TNs = 0
#     FPs = 0
#     FNs = 0

#     max_dt = 0.05
#     errors = len(positions1)*[None]
#     for it in range(0, len(positions1)):
#         time1 = times1[it]
#         (closest_it, closest_diff) = find_closest(time1, msgs)
#         if closest_diff > max_dt:
#             FNs += 1
#             continue

#         if len(msgs[closest_it].positions) > 0:
#             closest_pos = find_closest_pos(positions1[it, :], msgs[closest_it].positions)
#             cur_err = np.linalg.norm(positions1[it, :] - closest_pos)
#             if closest_pos[1] > 5 or cur_err > FP_error:
#                 FPs += len(msgs[closest_it].positions)
#             else:
#                 FPs += len(msgs[closest_it].positions) - 1
#                 TPs += 1
#                 errors[it] = np.linalg.norm(positions1[it, :] - closest_pos)
#         else:
#             FNs += 1

#     errors = np.array(errors, dtype=float)
#     nn_errors = errors[~np.isnan(errors)]
#     maxerr = float("NaN")
#     meanerr = float("NaN")
#     stderr = float("NaN")
#     if len(nn_errors) > 0:
#         maxerr = np.max(nn_errors)
#         meanerr = np.mean(nn_errors)
#         stderr = np.std(nn_errors)
#     rospy.loginfo("Max. error: {:f}".format(maxerr))
#     rospy.loginfo("Mean error: {:f}, std.: {:f}".format(meanerr, stderr))
#     return (TPs, TNs, FPs, FNs)

def publish_clock(pub, stamp):
    clk_msg = Clock()
    clk_msg.clock = stamp
    pub.publish(clk_msg)

def publish_tfs(pub, stamp, msgs, last_it, override_stamp=None):
    if last_it >= len(msgs):
        return last_it
    cur_msg = msgs[last_it]
    while cur_msg.transforms[0].header.stamp < stamp and last_it < len(msgs):
        pub.publish(cur_msg)
        cur_msg = msgs[last_it]
        last_it += 1
    return last_it

def publish_tfs_static(pub, msgs, stamp):
    for msg in msgs:
        # print("-----------------------------------------------------------------------------")
        # print("TF before:")
        # print(msg.transforms[0].header)
        # print("child_frame_id: \"{:s}\"".format(msg.transforms[0].child_frame_id))
        # for tf in msg.transforms:
        #     tf.header.stamp = stamp
        # print("TF after:")
        # print(msg.transforms[0].header)
        # print("child_frame_id: \"{:s}\"".format(msg.transforms[0].child_frame_id))
        pub.publish(msg)

def main():
    rospy.init_node('localization_evaluator', anonymous=True)

    ## LOAD ROS PARAMETERS
    loc_cnn_bag = rospy.get_param("~loc_cnn_bag")
    loc_depth_bag = rospy.get_param("~loc_depth_bag")
    gt_csv = rospy.get_param("~gt_csv")
    cinfo_cnn_bag = rospy.get_param("~cinfo_cnn_bag")
    cinfo_depth_bag = rospy.get_param("~cinfo_depth_bag")
    img_bag = rospy.get_param("~img_bag")
    tf_bag = rospy.get_param("~tf_bag")

    loc_cnn_topic = rospy.get_param("~loc_cnn_topic")
    loc_depth_topic = rospy.get_param("~loc_depth_topic")
    cinfo_cnn_topic = rospy.get_param("~cinfo_cnn_topic")
    cinfo_depth_topic = rospy.get_param("~cinfo_depth_topic")
    img_topic = rospy.get_param("~img_topic")

    img_path = rospy.get_param("~out_img_path")

    ground_truth_fname = rospy.get_param("~ground_truth_fname")

    ## PREPARE DUMMY PUB FOR TF
    clock_pub = rospy.Publisher("/clock", Clock, queue_size=100)
    tf_pub = rospy.Publisher("/tf", TFMessage, queue_size=100)
    tf_static_pub = rospy.Publisher("/tf_static", TFMessage, queue_size=100)
    tf_buffer = Buffer()
    listener = TransformListener(tf_buffer)

    ## START DATA LOADING
    rosbag_skip_time = 12
    rosbag_skip_time_end = 60
    loc_cnn_msgs = load_rosbag_msgs(loc_cnn_bag, loc_cnn_topic, skip_time=rosbag_skip_time, skip_time_end=rosbag_skip_time_end)
    start_t = loc_cnn_msgs[0].header.stamp
    end_t = loc_cnn_msgs[-1].header.stamp
    loc_depth_msgs = load_rosbag_msgs(loc_depth_bag, loc_depth_topic, start_time=start_t, end_time=end_t)
    cnn_cinfo = load_rosbag_msgs(cinfo_cnn_bag, cinfo_cnn_topic)[0]
    depth_cinfo = load_rosbag_msgs(cinfo_depth_bag, cinfo_depth_topic)[0]
    image_msgs = load_rosbag_msgs(img_bag, img_topic, start_time=start_t, end_time=end_t)
    tf_msgs = load_rosbag_msgs(tf_bag, "/tf", start_time=0, end_time=end_t)
    tf_static_msgs = load_rosbag_msgs(tf_bag, "/tf_static")
    # print(tf_static_msgs)

    # # Load GT positions from CSV
    gt_positions, gt_times = load_csv_data(gt_csv)

    cmodel = PinholeCameraModel()
    cmodel.fromCameraInfo(cnn_cinfo)

    (depth_TPs, depth_TNs, depth_FPs, depth_FNs) = (0, 0, 0, 0)
    (cnn_TPs, cnn_TNs, cnn_FPs, cnn_FNs) = (0, 0, 0, 0)

    ## INITIALIZATION COMPLETE - START SHOWING IMAGES
    winname = "localizations"
    cv2.namedWindow(winname, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    bridge = CvBridge()
    max_dt = 0.2
    last_tf_it = 0
    publish_clock(clock_pub, start_t)
    publish_tfs_static(tf_static_pub, tf_static_msgs, start_t)
    n_saved = 0
    for msg in image_msgs:
        if rospy.is_shutdown():
            break

        cur_stamp = msg.header.stamp

        # #{ PUBLISH TFs and Clock
        
        publish_clock(clock_pub, cur_stamp)
        last_tf_it = publish_tfs(tf_pub, cur_stamp + rospy.Duration.from_sec(max_dt), tf_msgs, last_tf_it)
        publish_tfs_static(tf_static_pub, tf_static_msgs, cur_stamp + rospy.Duration.from_sec(max_dt))
        
        # #} end of PUBLISH TFs

        img_orig = bridge.imgmsg_to_cv2(msg, "bgr8")
        img = img_orig.copy()

        depth_loc_msg = find_closest(cur_stamp, loc_depth_msgs, max_dt)
        cnn_loc_msg = find_closest(cur_stamp, loc_cnn_msgs, max_dt)

        depth_loc = msg_to_pos(tf_buffer, depth_loc_msg, msg.header.frame_id)
        cnn_loc = msg_to_pos(tf_buffer, cnn_loc_msg, msg.header.frame_id)
        gt_loc = find_closest_pos(cur_stamp, gt_positions, gt_times, "local_origin", msg.header.frame_id, tf_buffer)

        depth_pxpos = pos_to_pxpos(depth_loc, cmodel, offset=(0, -5))
        cnn_pxpos = pos_to_pxpos(cnn_loc, cmodel)
        gt_pxpos = pos_to_pxpos(gt_loc, cmodel)

        if depth_pxpos is not None:
            cv2.circle(img, depth_pxpos, 20, (255, 0, 0), 2)
        if cnn_pxpos is not None:
            cv2.circle(img, cnn_pxpos, 20, (0, 0, 255), 2)
        # cv2.circle(img, gt_pxpos, 20, (0, 0, 0), 2)

        if cnn_loc_msg is None:
            cnn_FNs += 1
        elif cnn_loc_msg.pose.pose.position.y > 5:
            cnn_FPs += 1
        else:
            cnn_TPs += 1

        if depth_loc_msg is None:
            depth_FNs += 1
        else:
            depth_TPs += 1

        depth_txt = "Our detector; TPs: {:6d}, TNs: {:6d}, FPs: {:6d}, FNs: {:6d}".format(depth_TPs, depth_TNs, depth_FPs, depth_FNs)
        cnn_txt =   "CNN detector; TPs: {:6d}, TNs: {:6d}, FPs: {:6d}, FNs: {:6d}".format(cnn_TPs, cnn_TNs, cnn_FPs, cnn_FNs)

        cv2.putText(img, depth_txt, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), )
        cv2.putText(img, cnn_txt, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), )

        cv2.imshow(winname, img)
        key = cv2.waitKey(100)
        if key == ord("s"):
            img = img_orig.copy()
            fname = "{:s}/saveimg{:d}.png".format(img_path, n_saved)
            raw_fname = "{:s}/raw_saveimg{:d}.png".format(img_path, n_saved)
            n_saved += 1
            cv2.imwrite(raw_fname, img)
            if depth_pxpos is not None:
                cv2.circle(img, depth_pxpos, 20, (255, 0, 0), 2)
            if cnn_pxpos is not None:
                cv2.circle(img, cnn_pxpos, 20, (0, 0, 255), 2)
            cv2.imwrite(fname, img)
            rospy.loginfo('Images {:s} and {:s} saved'.format(fname, raw_fname))
            

    depth_precision = depth_TPs/float(depth_TPs + depth_FPs)
    depth_recall = depth_TPs/float(depth_TPs + depth_FNs)
    rospy.loginfo(depth_txt)
    rospy.loginfo("recall: {:f}, precision: {:f}".format(depth_recall, depth_precision))

    cnn_precision = cnn_TPs/float(cnn_TPs + cnn_FPs)
    cnn_recall = cnn_TPs/float(cnn_TPs + cnn_FNs)
    rospy.loginfo("recall: {:f}, precision: {:f}".format(cnn_recall, cnn_precision))
    rospy.loginfo(cnn_txt)


if __name__ == '__main__':
    main()
