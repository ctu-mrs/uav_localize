#!/usr/bin/env python

# #{ header

import rospy
import rosbag

from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64

import pickle
import numpy as np
from numpy import cos
from numpy import sin
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import sys
import csv
import time

# #} end of header

# #{ 

def msg_to_pos(msg):
    cur_pos = msg.pose.pose.position
    position = np.array((cur_pos.x, cur_pos.y, cur_pos.z))
    return position

def msgs_to_pos(msgs):
    positions = np.zeros((len(msgs), 3))
    for it in range(0, len(msgs)):
        cur_pos = msgs[it].pose.pose.position
        positions[it, :] = (cur_pos.x, cur_pos.y, cur_pos.z)
    return positions

def msgs_to_times(msgs):
    times = np.zeros((len(msgs),))
    for it in range(0, len(msgs)):
        cur_stamp = msgs[it].header.stamp
        times[it] = cur_stamp.to_sec()
    return times

def pcl_msgs_to_pos(msgs):
    n_positions = 0
    for it in range(0, len(msgs)):
        for pt in msgs[it].points:
            n_positions += 1
    positions = np.zeros((n_positions, 3))
    pos_it = 0
    for it in range(0, len(msgs)):
        for pt in msgs[it].points:
            positions[pos_it, :] = (pt.x, pt.y, pt.z)
            pos_it += 1
    return positions

def pcl_msgs_to_times(msgs):
    n_positions = 0
    for it in range(0, len(msgs)):
        for pt in msgs[it].points:
            n_positions += 1
    times = np.zeros((n_positions,))
    pos_it = 0
    for it in range(0, len(msgs)):
        cur_stamp = msgs[it].header.stamp
        for pt in msgs[it].points:
            times[pos_it] = cur_stamp.to_sec()
            pos_it += 1
    return times

def load_rosbag_msgs(bag_fname, topic, skip_time=0, skip_time_end=0):
    rospy.loginfo("Using rosbag {:s}".format(bag_fname))
    bag = rosbag.Bag(bag_fname)
    n_msgs = bag.get_message_count(topic_filters=topic)
    if n_msgs == 0:
        rospy.logerr("No messages from topic {:s} in bag".format(topic))
    else:
        rospy.loginfo("Loading {:d} messages".format(n_msgs))
    msgs = n_msgs*[None]

    skip = rospy.Duration.from_sec(skip_time)
    start_time = rospy.Time.from_sec(bag.get_start_time()) + skip
    skip_end = rospy.Duration.from_sec(skip_time_end)
    end_time = rospy.Time.from_sec(bag.get_end_time()) - skip_end
    it = 0
    for topic, msg, cur_stamp in bag.read_messages(topics=topic, start_time=start_time, end_time=end_time):
        if rospy.is_shutdown():
            break
        msgs[it] = msg
        it += 1
    return msgs[0:it]

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

def load_csv_tf(csv_fname):
    rospy.loginfo("Using CSV file {:s}".format(csv_fname))
    with open(csv_fname, 'r') as fhandle:
        first_loaded = False
        csvreader = csv.reader(fhandle, delimiter=',')
        for row in csvreader:
            if not first_loaded:
                first_loaded = True
                continue
            tf = np.array([float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6]), float(row[7]), float(row[8])])
            return tf

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

def rotation(theta):
   # tx,ty,tz = theta
   tx = theta[0]
   ty = theta[1]
   tz = theta[2]

   Rx = np.array([[1,0,0], [0, cos(tx), -sin(tx)], [0, sin(tx), cos(tx)]])
   Ry = np.array([[cos(ty), 0, -sin(ty)], [0, 1, 0], [sin(ty), 0, cos(ty)]])
   Rz = np.array([[cos(tz), -sin(tz), 0], [sin(tz), cos(tz), 0], [0,0,1]])

   return np.dot(Rx, np.dot(Ry, Rz)) 

def transform_gt(gt_pos, params, inverse=False):
    angles = params[0:3]
    shift = params[3:6]
    R = rotation(angles)
    if inverse:
        R = R.transpose()
    # shift = np.array([20, -20, -1]).transpose()
    rotated = np.dot(gt_pos, R)
    rotated += shift
    return rotated

def find_closest(time, times):
    closest_it = 0
    closest_diff = abs(time - times[closest_it])
    for it in range(0, len(times)):
        cur_diff = abs(time - times[it])
        # print(cur_diff, closest_diff)
        if cur_diff <= closest_diff:
            closest_it = it
            closest_diff = cur_diff
        else:
            break
    return closest_it

def get_positions_time(positions1, times1, positions2, times2):
    max_dt = 0.1
    tposs = np.zeros_like(positions1)
    for it in range(0, len(positions1)):
        time1 = times1[it]
        closest_it = find_closest(time1, times2)
        # sys.stdout.write("{:d}: {:f}s X {:f}s  :: ".format(it, time1, times2[closest_it]))
        # if abs(times2[closest_it] - time1) > max_dt or positions2[closest_it][1] > 6:
        if abs(times2[closest_it] - time1) > max_dt:
            tposs[it, :] = np.array([np.nan, np.nan, np.nan])
            # print("FP")
            continue
        # print("TP")
        tposs[it, :] = positions2[closest_it, :]
    return tposs

# def calc_errors_time(positions1, tposs):
#     max_dt = 0.1
#     return np.linalg.norm(tposs-positions1, axis=1)

def calc_errors(positions1, positions2):
    errors = np.linalg.norm(positions1 - positions2, axis=1)
    return errors

def calc_error(positions1, positions2, FP_error):
    errors = calc_errors(positions1, positions2)
    tot_error = 0
    N = 0
    for err in errors:
        if err is not None and err > 0 and err < FP_error:
            tot_error += err
            N += 1
    return (tot_error, N)

def calc_avg_error_diff(positions1, positions2, FP_error):
    E1 = np.zeros((6,))
    dtransf = 0.1
    # max_y = -15
    max_y = float('Inf')
    for it in range(0, 6):
        params = np.zeros((6,))
        params[it] = dtransf
        pts = transform_gt(positions1, params)
        idxs = positions2[:, 1] < max_y
        tot_err, N = calc_error(pts[idxs, :], positions2[idxs, :], FP_error)
        E1[it] = tot_err/float(N)
    E2 = np.zeros((6,))
    for it in range(0, 6):
        params = np.zeros((6,))
        params[it] = -dtransf
        pts = transform_gt(positions1, params)
        idxs = positions2[:, 1] < max_y
        tot_err, N = calc_error(pts[idxs, :], positions2[idxs, :], FP_error)
        E2[it] = tot_err/float(N)
    dE = (E2 - E1)/(2*dtransf)
    return dE


def find_min_tf(positions, to_positions, FP_error, only_rot=False):
    lam_ang = 2e-1
    lam_pos = 2e-3
    if only_rot:
        lam_pos = 0
    # N_SAMPLES = np.min([10000, len(positions)])
    transf_tot = np.zeros((6,))
    for it in range(0, 260):
        if rospy.is_shutdown():
            break
        print("iteration {:d}:".format(it))
        # idxs = np.random.choice(len(positions)-1, N_SAMPLES, replace=False)
        # rand_poss = positions[idxs, :]
        # rand_tims = times[idxs]
        rand_poss = positions
        tot_err, N = calc_error(rand_poss, to_positions, FP_error)
        E0 = tot_err/float(N)
        print("avg. error: {:f} (from {:d} points)".format(E0, N))
        t1 = time.time()
        dE = calc_avg_error_diff(rand_poss, to_positions, FP_error)
        dt = time.time() - t1
        print("avg. error diff [{:f}, {:f}, {:f}, {:f}, {:f}, {:f}] (time: {:f}s)".format(dE[0], dE[1], dE[2], dE[3], dE[4], dE[5], dt))
        print("cur tf [{:f}, {:f}, {:f}, {:f}, {:f}, {:f}])".format(transf_tot[0], transf_tot[1], transf_tot[2], transf_tot[3], transf_tot[4], transf_tot[5]))
        transf = dE
        transf[0:3] *= lam_ang
        transf[3:6] *= lam_pos
        positions = transform_gt(positions, transf)
        transf_tot += transf
        if it % 40 == 0:
            lam_pos /= 2
            lam_ang /= 2
    return transf_tot

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

def put_to_file(header, fname):
    with open(fname, 'w') as ofhandle:
        ofhandle.write(header)

def append_to_file(dist, error, fname):
    with open(fname, 'a') as ofhandle:
        ofhandle.write("{:f},{:f}\n".format(dist, error))

def put_tf_to_file(tf, fname):
    with open(fname, 'w') as ofhandle:
        ofhandle.write("yaw,pitch,roll,x,y,z,rotx,roty,rotz\n")
        ofhandle.write("{:f},{:f},{:f},{:f},{:f},{:f},{:f},{:f},{:f}\n".format(tf[0], tf[1], tf[2], tf[3], tf[4], tf[5], tf[6], tf[7], tf[8]))

def put_errs_to_file(dists, errors, dist_errors, fname):
    with open(fname, 'w') as ofhandle:
        ofhandle.write("distance,error,distErrors\n")
        for it in range(0, len(dists)):
            ofhandle.write("{:f},{:f},{:f}\n".format(dists[it], errors[it], dist_errors[it]))

def calc_statistics(positions1, tposs, FP_error):
    errors = calc_errors(positions1, tposs)
    TPs = 0
    TNs = 0
    FPs = 0
    FNs = 0

    for err in errors:
        if err is None:
            FNs += 1
        elif err > FP_error or err < 0:
            FNs += 1
            FPs += 1
        else:
            TPs += 1

    errors = np.array(errors, dtype=float)
    return (TPs, TNs, FPs, FNs, errors)

def calc_probs(TPs, GTs):
    probs = np.zeros((len(TPs),))
    for it in range(0, len(TPs)):
        probs[it] = np.sum(TPs[0:it+1])/np.sum(GTs[0:it+1])
    return probs

# def filter_gt_pos(gt_positions, max_diff=1):
#     times = [
#         [14.6, 15.0, -1.2],
#         [60.6, 61.0, -1.2],

#     prev_pos = gt_positions[0, :]
#     for it in range(0, gt_positions.shape[0]):
#         pos = gt_positions[it, :]
#         print(it, ":", pos, " XXX ", prev_pos)
#         for it2 in range(0, len(pos)):
#             diff = pos[it2] - prev_pos[it2]
#             while np.abs(diff) > max_diff:
#                 pos[it2] -= np.sign(diff)*max_diff
#                 diff = pos[it2] - prev_pos[it2]
#                 print(it, ":", pos, " XXX ", prev_pos, ", diff: ", diff)
#         prev_pos = pos
#         gt_positions[it, :] = pos
#     return gt_positions

def subtract(from_pos, from_times, sub_pos, sub_times):
    last_sub_it = 0
    for it in range(0, len(from_pos)):
        cur_t = from_times[it]
        next_sub_t = sub_times[last_sub_it+1]
        while next_sub_t < cur_t and last_sub_it+2 < len(sub_times):
            last_sub_it += 1
            next_sub_t = sub_times[last_sub_it+1]

        cur_pos = from_pos[it]
        cur_sub_pos = sub_pos[last_sub_it]
        from_pos[it] = cur_pos - cur_sub_pos


loc_msg = None
def cbk_loc(msg):
    global loc_msg
    loc_msg = msg

tgt_grt_msg = None
def cbk_tgt_grt(msg):
    global tgt_grt_msg
    tgt_grt_msg = msg

obs_grt_msg = None
def cbk_obs_grt(msg):
    global obs_grt_msg
    obs_grt_msg = msg

obs_odo_msg = None
def cbk_obs_odo(msg):
    global obs_odo_msg
    obs_odo_msg = msg

# #} end of 


def main():
    global loc_msg
    global tgt_grt_msg
    global obs_grt_msg
    global obs_odo_msg
    rospy.init_node('localization_evaluator', anonymous=True)

    out_fname = rospy.get_param('~out_filename')

    sub_loc = rospy.Subscriber('~localization', PoseWithCovarianceStamped, cbk_loc)
    sub_tgt_grt = rospy.Subscriber('~target_ground_truth', Odometry, cbk_tgt_grt)
    sub_obs_grt = rospy.Subscriber('~observer_ground_truth', Odometry, cbk_obs_grt)
    sub_obs_odo = rospy.Subscriber('~observer_odometry', Odometry, cbk_obs_odo)

    pub_dist = rospy.Publisher('~target_distance', Float64, queue_size=10)
    pub_err = rospy.Publisher('~localization_error', Float64, queue_size=10)

    put_to_file("dist,err\n", out_fname)
    dists = list()
    est_dists = list()
    errs = list()
    r = rospy.Rate(200)
    while not rospy.is_shutdown():
        if loc_msg is not None and tgt_grt_msg is not None and obs_grt_msg is not None and obs_odo_msg is not None:
            rospy.loginfo_throttle(1.0, "processing data")

            tgt_pos = msg_to_pos(tgt_grt_msg)
            obs_pos = msg_to_pos(obs_grt_msg)
            obs_odo = msg_to_pos(obs_odo_msg)
            tgt_dir = tgt_pos - obs_pos
            tgt_dist = np.linalg.norm(tgt_dir)

            loc_pos = msg_to_pos(loc_msg)
            loc_dir = loc_pos - obs_odo
            loc_dir = loc_dir - 0.5*loc_dir/np.linalg.norm(loc_dir)
            loc_dist = np.linalg.norm(loc_dir)
            # loc_pos_corr = loc_pos - obs_pos
            loc_err = np.linalg.norm(loc_dir - tgt_dir)

            pub_dist.publish(tgt_dist)
            pub_err.publish(loc_err)

            print("tgt_pos: ", tgt_pos)
            print("obs_pos: ", obs_pos)
            print("loc_pos: ", loc_pos)
            print("tgt_dir", tgt_dir)
            print("loc_dir", loc_dir)

            print("tgt_dist", tgt_dist)
            print("loc_err", loc_err)
            print("loc_dist", loc_dist)
            
            print("-----------------------------------")

            # append_to_file(tgt_dist, loc_err, out_fname)
            dists.append(tgt_dist)
            est_dists.append(loc_dist)
            errs.append(loc_err)

            loc_msg = None
        else:
            rospy.loginfo_throttle(1.0, "waiting for data")
        r.sleep()

    dists = np.array(dists)
    est_dists = np.array(est_dists)
    dist_errs = dists - est_dists
    print("average distance error: {:2f}".format(np.mean(dist_errs)))
    errs = np.array(errs)
    print("average error: {:2f}".format(np.mean(errs)))

    _, err_avg_edges = np.histogram(dists, bins=15)
    err_bin_width = err_avg_edges[1] - err_avg_edges[0]
    err_avg_centers = err_avg_edges[0:-1] + err_bin_width/2
    err_avg = len(err_avg_centers)*[None]
    disterr_avg = len(err_avg_centers)*[None]
    for it in range(0, len(err_avg_edges)-1):
        low = err_avg_edges[it]
        high = err_avg_edges[it+1]
        idxs = np.logical_and(dists > low, dists < high)
        cur_errs = errs[idxs]
        err_avg[it] = np.mean(cur_errs)
        cur_disterrs = dist_errs[idxs]
        disterr_avg[it] = np.mean(cur_disterrs)

    put_errs_to_file(err_avg_centers, err_avg, disterr_avg, out_fname)

    fig = plt.figure()
    plt.plot(err_avg_centers, err_avg, 'r')
    plt.plot(dists, dist_errs, 'bx')
    plt.plot(dists, est_dists, 'rx')
    plt.plot(dists, dists, 'black')
    plt.title("Localization error over distance")
    plt.show()

if __name__ == '__main__':
    main()
