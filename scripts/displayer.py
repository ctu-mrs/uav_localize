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
from matplotlib import transforms
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import csv
import copy

from PIL import Image, ImageFont, ImageDraw
import cv2
from cv_bridge import CvBridge

# #{ CUT BY TIME FUNCTIONS

def cut_to(msgs, end_time):
    end_it = 0
    for msg in msgs:
        if msg.header.stamp < end_time:
            end_it += 1
        else:
            break
    return msgs[0:end_it]

def cut_from(msgs, start_time):
    start_it= 0
    for msg in msgs:
        if msg.header.stamp < start_time:
            start_it += 1
        else:
            break
    return msgs[start_it:-1]

# #} end of CUT BY TIME FUNCTIONS

# #{ load_csv_data function

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

# #} end of load_csv_data function

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

def load_rosbag_stream(bag_fname, topic, skip_time=0, skip_time_end=0, start_time=None, end_time=None, bag=None):
    rospy.loginfo("Using rosbag {:s}".format(bag_fname))
    if bag is None:
        bag = rosbag.Bag(bag_fname)
    skip = rospy.Duration.from_sec(skip_time)
    if start_time is None:
        start_time = rospy.Time.from_sec(bag.get_start_time()) + skip
    if end_time is None:
        skip_end = rospy.Duration.from_sec(skip_time_end)
        end_time = rospy.Time.from_sec(bag.get_end_time()) - skip_end
    return bag.read_messages(topics=topic, start_time=start_time, end_time=end_time)

def load_rosbag_msgs(bag_fname, topic, skip_time=0, skip_time_end=0, start_time=None, end_time=None):
    bag = rosbag.Bag(bag_fname)
    n_msgs = bag.get_message_count(topic_filters=topic)
    if n_msgs == 0:
        rospy.logerr("No messages from topic {:s} in bag".format(topic))
    else:
        rospy.loginfo("Loading {:d} messages".format(n_msgs))
    msgs = n_msgs*[None]

    msg_stream = load_rosbag_stream(bag_fname, topic, skip_time=skip_time, skip_time_end=skip_time_end, start_time=start_time, end_time=end_time, bag=bag)

    it = 0
    for topic, msg, cur_stamp in msg_stream:
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


def find_closest_msg(stamp, msgs, max_dt=float('Inf')):
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


def find_closest(stamp, stamps):
    closest_it = None
    closest_diff = float('Inf')
    for it in range(0, len(stamps)):
        cur_stamp = stamps[it]
        cur_diff = abs(stamp - cur_stamp)
        if cur_diff <= closest_diff:
            closest_it = it
            closest_diff = cur_diff
        else:
            break
    return closest_it

def time_align(times1, positions2, times2):
    max_dt = 0.1
    positions_time_aligned = np.zeros((len(times1), 3))
    for it in range(0, len(times1)):
        time1 = times1[it]
        closest_it = find_closest(time1, times2)
        if abs(times2[closest_it] - time1) > max_dt:
            positions_time_aligned[it, :] = np.array([None, None, None])
        else:
            positions_time_aligned[it, :] = positions2[closest_it, :]
    return positions_time_aligned

def transform_to(pos, stamp, frame_id, to_frame_id, tf_buffer):
    if pos is None:
        return None
    ps = PointStamped()
    ps.header.stamp = stamp
    ps.header.frame_id = frame_id
    ps.point.x = pos[0]
    ps.point.y = pos[1]
    ps.point.z = pos[2]

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

# #{ fig2rgb_array

def fig2rgb_array(fig):
    fig.canvas.draw()
    buf = np.array(fig.canvas.print_to_buffer()[0])
    # buf = fig.canvas.tostring_argb()
    ncols, nrows = fig.canvas.get_width_height()
    buf = buf.reshape(nrows, ncols, 4)
    ret = buf[:, :, [2, 1, 0]]
    # ret[:, :, [1, 3]] = ret[:, :, [3, 1]]
    return ret

def fig2argb_array(fig):
    fig.canvas.draw()
    buf = np.array(fig.canvas.print_to_buffer()[0])
    # buf = fig.canvas.tostring_argb()
    ncols, nrows = fig.canvas.get_width_height()
    ret = np.fromstring(buf, dtype=np.uint8).reshape(nrows, ncols, 4)
    ret[:, :, [0, 1, 2, 3]] = ret[:, :, [3, 2, 1, 0]]
    # ret[:, :, [1, 3]] = ret[:, :, [3, 1]]
    return ret

# #} end of fig2argb_array

def scale_by_alpha(img, a):
    img[:, :, 0] = img[:, :, 0]/255.0*a
    img[:, :, 1] = img[:, :, 1]/255.0*a
    img[:, :, 2] = img[:, :, 2]/255.0*a

def add_with_offset(img, rgb, x=0, y=0):
    # res = cv2.addWeighted(img[0:fig_data.shape[0], 0:fig_data.shape[1]], 0, fig_data, 1, 0.0)
    img[y:(y+rgb.shape[0]), x:(x+rgb.shape[1]), :] += rgb

def add_with_alpha(img, argb, x=0, y=0):
    # res = cv2.addWeighted(img[0:fig_data.shape[0], 0:fig_data.shape[1]], 0, fig_data, 1, 0.0)
    rgb = argb[:, :, 1:4]
    a = argb[:, :, 0]
    scale_by_alpha(rgb, a)
    scale_by_alpha(img[y:(y+argb.shape[0]), x:(x+argb.shape[1])], (255 - a))
    add_with_offset(img, rgb, x, y)

def main():
    rospy.init_node('localization_evaluator', anonymous=True)

    ## LOAD ROS PARAMETERS
    loc_cnn_bag = rospy.get_param("~loc_cnn_bag")
    loc_depth_bag = rospy.get_param("~loc_depth_bag")
    depth_csv = rospy.get_param("~depth_csv")
    cnn_csv = rospy.get_param("~cnn_csv")
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
    depth_topic = rospy.get_param("~depth_topic")

    img_path = rospy.get_param("~out_img_path")

    ## PREPARE DUMMY PUB FOR TF
    clock_pub = rospy.Publisher("/clock", Clock, queue_size=100)
    tf_pub = rospy.Publisher("/tf", TFMessage, queue_size=100)
    tf_static_pub = rospy.Publisher("/tf_static", TFMessage, queue_size=100)
    tf_buffer = Buffer()
    listener = TransformListener(tf_buffer)

    ## START DATA LOADING
    rosbag_skip_time = 10
    rosbag_skip_time_end = 0
    loc_cnn_msgs = load_rosbag_msgs(loc_cnn_bag, loc_cnn_topic, skip_time=rosbag_skip_time, skip_time_end=rosbag_skip_time_end)
    start_t = loc_cnn_msgs[0].header.stamp
    end_t = loc_cnn_msgs[-1].header.stamp
    loc_depth_msgs = load_rosbag_msgs(loc_depth_bag, loc_depth_topic, start_time=start_t, end_time=end_t)
    cnn_cinfo = load_rosbag_msgs(cinfo_cnn_bag, cinfo_cnn_topic)[0]
    depth_cinfo = load_rosbag_msgs(cinfo_depth_bag, cinfo_depth_topic)[0]
    # image_msgs = load_rosbag_msgs(img_bag, img_topic, start_time=start_t, end_time=end_t)
    # depth_msgs = load_rosbag_msgs(img_bag, depth_topic, start_time=start_t, end_time=end_t)
    image_msgs_stream = load_rosbag_stream(img_bag, [img_topic, depth_topic], start_time=start_t, end_time=end_t)
    tf_msgs = load_rosbag_msgs(tf_bag, "/tf", start_time=0, end_time=end_t)
    tf_static_msgs = load_rosbag_msgs(tf_bag, "/tf_static")
    # print(tf_static_msgs)

    # # Load positions from CSV
    # depth_positions, depth_times = load_csv_data(depth_csv)
    # cnn_positions, cnn_times = load_csv_data(cnn_csv)
    gt_positions, gt_times = load_csv_data(gt_csv)
    # depth_positions = time_align(gt_times, depth_positions, depth_times)
    # cnn_positions = time_align(gt_times, cnn_positions, cnn_times)
    start_time = gt_times[0]
    end_time = gt_times[-1]
    duration = end_time - start_time

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
    w = 640
    h = 480
    dpi = 200

    fig = plt.figure(figsize=(w/float(dpi), h/float(dpi)), dpi=dpi, facecolor='w')
    # fig.patch.set_alpha(0.0)
    # ax = fig.add_subplot(111, projection='3d')
    ax = fig.add_subplot(111)

    fig2 = plt.figure(figsize=(w/float(dpi), h/float(dpi)/2), dpi=dpi, facecolor='w')
    # fig2.patch.set_alpha(0.0)
    # ax = fig.add_subplot(111, projection='3d')
    ax2 = fig2.add_subplot(111)

    mono_font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", 26)

    time_hist = list()

    # depth_TP_hist = list()
    # depth_TN_hist = list()
    # depth_FP_hist = list()
    # depth_FN_hist = list()

    # cnn_TP_hist = list()
    # cnn_TN_hist = list()
    # cnn_FP_hist = list()
    # cnn_FN_hist = list()

    depth_err_hist = list()
    cnn_err_hist = list()
    img_depth = None
    img = None

    for topic, msg, cur_stamp in image_msgs_stream:
        if rospy.is_shutdown():
            break

        # #{ obtain images
        
        if topic == depth_topic:
            img_depth = bridge.imgmsg_to_cv2(msg, "mono16")
            # img_depth = cv2.resize(img_depth, (w, h))
            h2, w2 = img_depth.shape
            dh = h2 - h
            dw = w2 - w
            img_depth = img_depth[dh/2:dh/2+h, dw/2:dw/2+w]
            img_depth = cv2.convertScaleAbs(img_depth, alpha=1/256.0)
            img_depth = cv2.cvtColor(img_depth, cv2.COLOR_GRAY2BGR)
            # img_depth = cv2.applyColorMap(img_depth, cv2.COLORMAP_JET)
        else:
            img = bridge.imgmsg_to_cv2(msg, "bgr8")
        if img_depth is None or img is None:
            continue
        
        # #} end of obtain images

        # #{ PUBLISH TFs and Clock
        
        publish_clock(clock_pub, cur_stamp)
        last_tf_it = publish_tfs(tf_pub, cur_stamp + rospy.Duration.from_sec(max_dt), tf_msgs, last_tf_it)
        publish_tfs_static(tf_static_pub, tf_static_msgs, cur_stamp + rospy.Duration.from_sec(max_dt))
        
        # #} end of PUBLISH TFs

        # #{ obtain data
        
        img_total = 255*np.ones((2*h, 2*w, 3), dtype=np.uint8)
        
        depth_loc_msg = find_closest_msg(cur_stamp, loc_depth_msgs, max_dt)
        cnn_loc_msg = find_closest_msg(cur_stamp, loc_cnn_msgs, max_dt)
        
        depth_loc = msg_to_pos(tf_buffer, depth_loc_msg, msg.header.frame_id)
        depth_loc_wf = transform_to(depth_loc, cur_stamp, msg.header.frame_id, "local_origin", tf_buffer)
        cnn_loc = msg_to_pos(tf_buffer, cnn_loc_msg, msg.header.frame_id)
        cnn_loc_wf = transform_to(cnn_loc, cur_stamp, msg.header.frame_id, "local_origin", tf_buffer)
        closest_it = find_closest(cur_stamp.to_sec(), gt_times)
        gt_loc_wf = gt_positions[closest_it]
        # depth_loc_wf = depth_positions[closest_it]
        # cnn_loc_wf = cnn_positions[closest_it]
        gt_loc = transform_to(gt_loc_wf, cur_stamp, "local_origin", msg.header.frame_id, tf_buffer)
        
        # #} end of obtain data

        # #{ draw localizations
        
        depth_pxpos = pos_to_pxpos(depth_loc, cmodel, offset=(0, -5))
        cnn_pxpos = pos_to_pxpos(cnn_loc, cmodel)
        gt_pxpos = pos_to_pxpos(gt_loc, cmodel)
        
        if depth_pxpos is not None:
            cv2.circle(img, depth_pxpos, 20, (255, 0, 0), 2)
        if cnn_pxpos is not None:
            cv2.circle(img, cnn_pxpos, 20, (0, 0, 255), 2)
            # cv2.rect(img, cnn_pxpos, (cnn_loc_msg.width, cnn_loc_msg.height), 20, (0, 0, 255), 2)
        # cv2.circle(img, gt_pxpos, 20, (0, 0, 0), 2)
        
        # #} end of draw localizations

        # #{ update statistics

        depth_err = None
        if depth_loc is not None:
            depth_err = np.linalg.norm(gt_loc-depth_loc)

        cnn_err = None
        if cnn_loc is not None and cnn_loc_wf[1] < 5:
            cnn_err = np.linalg.norm(gt_loc-cnn_loc)
        
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
        
        # #} end of update statistics

        # #{ unused
        
        # depth_TP_hist.append(depth_TPs)
        # depth_TN_hist.append(depth_TNs)
        # depth_FP_hist.append(depth_FPs)
        # depth_FN_hist.append(depth_FNs)
        
        # cnn_TP_hist.append(cnn_TPs)
        # cnn_TN_hist.append(cnn_TNs)
        # cnn_FP_hist.append(cnn_FPs)
        # cnn_FN_hist.append(cnn_FNs)
        
        # #} end of unused
        
        time_hist.append(cur_stamp.to_sec() - start_time)
        depth_err_hist.append(depth_err)
        cnn_err_hist.append(cnn_err)

        # #{ PLOT THE POSITIONS

        ax.clear()
        # ax.patch.set_alpha(0.0)
        ax.set_xlim((-10, 60))
        ax.set_ylim((-20, 20))
        ax.set_xlabel('x (x)')
        ax.set_ylabel('y (m)')
        ax.set_aspect('equal')
        ax.set_title('top view')
        ax.yaxis.set_label_coords(-0.10, 0.5)
        ax.xaxis.set_label_coords(0.5, -0.15)
        ax.grid()
        ax.set_xticks(np.arange(-10, 61, 10))
        ax.set_yticks(np.arange(-20, 21, 10))
        if depth_loc is not None:
            # ax.plot([depth_loc[0]], [depth_loc[1]], [depth_loc[2]], 'bx')
            ax.plot([depth_loc_wf[0]], [depth_loc_wf[1]], 'bx')
        if cnn_loc is not None:
            # ax.plot([cnn_loc[0]], [cnn_loc[1]], [cnn_loc[2]], 'rx')
            ax.plot([cnn_loc_wf[0]], [cnn_loc_wf[1]], 'rx')
        # ax.plot([gt_loc[0]], [gt_loc[1]], [gt_loc[2]], 'x')
        ax.plot([gt_loc_wf[0]], [gt_loc_wf[1]], '.', color='black')
        fig_data = fig2rgb_array(fig)
        # add_with_alpha(img, fig_data, 0, 0)
        img_total[h:2*h, 0:w, :] = fig_data
        
        # #} end of PLOT THE POSITIONS

        # #{ PLOT THE ERORS

        ax2.clear()
        # ax2.patch.set_alpha(0.0)
        ax2.set_xlim((0, 60))
        ax2.set_ylim((0, 20))
        ax2.set_xlabel('time (s)')
        # ax2.set_ylabel('RMSE (m)')
        ax2.set_title('localization error')
        ax2.set_aspect(1/1.5)
        ax2.set_yticks(np.arange(0, 10, 20))
        ax2.yaxis.set_label_coords(-0.15, 0.5)
        ax2.xaxis.set_label_coords(0.5, -0.25)
        ax2.grid()

        # ax2.plot(time_hist, depth_TP_hist)
        # ax2.plot(time_hist, depth_FN_hist)
        ax2.plot(time_hist, depth_err_hist, 'b')
        ax2.plot(time_hist, cnn_err_hist, 'r')
        fig_data = fig2rgb_array(fig2)
        # add_with_alpha(img, fig_data, 320, 0)
        img_total[int(1.5*h):2*h, w:2*w, :] = fig_data
        
        # #} end of PLOT THE ERROR

        # #{ CREATE TEXT IMAGE
        
        img_text = 255*np.ones((h/2, w, 3), dtype=np.uint8)
        img_p = Image.fromarray(img_text)
      
        draw = ImageDraw.Draw(img_p)
        header_txt= "Detector |  TPs |  TNs |  FPs |  FNs"
        depth_txt = "ours     | {:4d} | {:4d} | {:4d} | {:4d}".format(depth_TPs, depth_TNs, depth_FPs, depth_FNs)
        cnn_txt =   "CNN      | {:4d} | {:4d} | {:4d} | {:4d}".format(cnn_TPs, cnn_TNs, cnn_FPs, cnn_FNs)
        
        draw.text((20, 26), header_txt, (0,0,0), font=mono_font)
        draw.text((20, 60), depth_txt, (255,0,0), font=mono_font)
        draw.text((20, 90), cnn_txt, (0,0,255), font=mono_font)

        img_text = np.array(img_p)
        cv2.line(img_text, (0, 58), (w, 58), (0,0,0))
        img_total[h:int(1.5*h), w:2*w, :] = img_text
        
        # #} end of unused

        img_total[0:h, 0:w, :] = img
        img_total[0:h, w:2*w, :] = img_depth
        cv2.imshow(winname, img_total)
        key = cv2.waitKey(1)
        # if key == ord("s"):
        #     img = img_orig.copy()
        #     fname = "{:s}/saveimg{:d}.png".format(img_path, n_saved)
        #     raw_fname = "{:s}/raw_saveimg{:d}.png".format(img_path, n_saved)
        #     n_saved += 1
        #     cv2.imwrite(raw_fname, img)
        #     if depth_pxpos is not None:
        #         cv2.circle(img, depth_pxpos, 20, (255, 0, 0), 2)
        #     if cnn_pxpos is not None:
        #         cv2.circle(img, cnn_pxpos, 20, (0, 0, 255), 2)
        #     cv2.imwrite(fname, img)
        #     rospy.loginfo('Images {:s} and {:s} saved'.format(fname, raw_fname))
            

    # depth_precision = depth_TPs/float(depth_TPs + depth_FPs)
    # depth_recall = depth_TPs/float(depth_TPs + depth_FNs)
    # rospy.loginfo(depth_txt)
    # rospy.loginfo("recall: {:f}, precision: {:f}".format(depth_recall, depth_precision))

    # cnn_precision = cnn_TPs/float(cnn_TPs + cnn_FPs)
    # cnn_recall = cnn_TPs/float(cnn_TPs + cnn_FNs)
    # rospy.loginfo(cnn_txt)
    # rospy.loginfo("recall: {:f}, precision: {:f}".format(cnn_recall, cnn_precision))


if __name__ == '__main__':
    main()
