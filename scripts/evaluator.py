#!/usr/bin/env python
import rospy
import rosbag

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

# def load_pickle(in_fname):
#     msgs = None
#     if os.path.isfile(in_fname):
#         with open(in_fname, 'r') as in_fhandle:
#             msgs = pickle.load(in_fhandle)
#     return msgs

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

def put_to_file(positions, times, fname):
    with open(fname, 'w') as ofhandle:
        ofhandle.write("x,y,z,time\n")
        for it in range(0, len(positions)):
            ofhandle.write("{:f},{:f},{:f},{:f}\n".format(positions[it, 0], positions[it, 1], positions[it, 2], times[it]))

def put_tf_to_file(tf, fname):
    with open(fname, 'w') as ofhandle:
        ofhandle.write("yaw,pitch,roll,x,y,z,rotx,roty,rotz\n")
        ofhandle.write("{:f},{:f},{:f},{:f},{:f},{:f},{:f},{:f},{:f}\n".format(tf[0], tf[1], tf[2], tf[3], tf[4], tf[5], tf[6], tf[7], tf[8]))

def put_errs_to_file(dists, errors, fname):
    with open(fname, 'w') as ofhandle:
        ofhandle.write("error,distance\n")
        for it in range(0, len(dists)):
            ofhandle.write("{:f},{:f}\n".format(dists[it], errors[it]))

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

def main():
    rospy.init_node('localization_evaluator', anonymous=True)
    # out_fname = rospy.get_param('~output_filename')
    # in_fname = rospy.get_param('~input_filename')
    loc_bag_fname = rospy.get_param('~localization_bag_name')
    gt_bag_fname = rospy.get_param('~ground_truth_bag_name')
    obs_bag_fname = rospy.get_param('~observer_bag_name')

    loc_topic_name = rospy.get_param('~localization_topic_name')
    gt_topic_name = rospy.get_param('~ground_truth_topic_name')
    obs_topic_name = rospy.get_param('~observer_ground_truth_topic_name')

    loc_out_fname = rospy.get_param('~localization_out_fname')
    gt_out_fname = rospy.get_param('~ground_truth_out_fname')
    # loc_out_fname2 = rospy.get_param('~localization_out_fname2')
    # gt_out_fname2 = rospy.get_param('~ground_truth_out_fname2')
    # loc_out_fname3 = rospy.get_param('~localization_out_fname3')
    # gt_out_fname3 = rospy.get_param('~ground_truth_out_fname3')
    tf_out_fname = rospy.get_param('~tf_out_fname')
    dist_err_out_fname = rospy.get_param('~distance_error_out_fname')
    TP_prob_out_fname = rospy.get_param('~TP_probability_out_fname')

    # msgs = load_pickle(in_fname)
    FP_error = 30.0 # meters
    # FP_error = float('Inf')

    rosbag_skip_time = 35
    rosbag_skip_time_end = 65

    tf = None
    if os.path.isfile(tf_out_fname):
        rospy.loginfo("File {:s} found, loading it".format(tf_out_fname))
        tf = load_csv_tf(tf_out_fname)
        rospy.loginfo("Loaded a tf [{:f}, {:f}, {:f}, {:f}, {:f}, {:f}], [{:f}, {:f}, {:f}] from {:s}".format(tf[0], tf[1], tf[2], tf[3], tf[4], tf[5], tf[6], tf[7], tf[8], tf_out_fname))
        tf_frombag = False
    else:
        tf_frombag = True

    bag = rosbag.Bag(loc_bag_fname)

    obs_positions = None
    obs_times = None
    if True:
        rospy.loginfo("Loading data from rosbag {:s}".format(loc_bag_fname))
        # if msgs is None:
        # rospy.loginfo("Input file not valid, loading from rosbag")
        obs_msgs = load_rosbag_msgs(obs_bag_fname, obs_topic_name, skip_time=rosbag_skip_time, skip_time_end=rosbag_skip_time_end)
        if obs_msgs is None:
            exit(1)

        obs_positions = msgs_to_pos(obs_msgs)
        obs_times = msgs_to_times(obs_msgs)
        obs_idxs = np.argsort(obs_times)
        obs_positions = obs_positions[obs_idxs]
        obs_times = obs_times[obs_idxs]
    rospy.loginfo("Loaded {:d} observer positions".format(len(obs_positions)))

    # #{ load localization
    
    # start_time = 1567515863.202346
    if os.path.isfile(loc_out_fname):
        rospy.loginfo("File {:s} found, loading it".format(loc_out_fname))
        loc_positions, loc_times = load_csv_data(loc_out_fname)
        rospy.loginfo("Loaded {:d} positions from {:s}".format(len(loc_positions), loc_out_fname))
        # loc_positions2, loc_times2 = load_csv_data(loc_out_fname2)
        # rospy.loginfo("Loaded {:d} positions from {:s}".format(len(loc_positions2), loc_out_fname2))
        # loc_positions3, loc_times3 = load_csv_data(loc_out_fname3)
        # rospy.loginfo("Loaded {:d} positions from {:s}".format(len(loc_positions3), loc_out_fname3))
        # loc_positions = np.vstack((loc_positions, loc_positions2, loc_positions3))
        # loc_times = np.hstack((loc_times, loc_times2, loc_times3))
        # loc_positions = np.vstack((loc_positions1, loc_positions2))
        # loc_times = np.hstack((loc_times1, loc_times2))
        loc_frombag = False
    else:
        rospy.loginfo("Loading data from rosbag {:s}".format(loc_bag_fname))
        # if msgs is None:
        # rospy.loginfo("Input file not valid, loading from rosbag")
        loc_msgs = load_rosbag_msgs(loc_bag_fname, loc_topic_name, skip_time=rosbag_skip_time, skip_time_end=rosbag_skip_time_end)
        if loc_msgs is None:
            exit(1)
    
        # loc_positions = pcl_msgs_to_pos(loc_msgs)
        # loc_times = pcl_msgs_to_times(loc_msgs)
        loc_positions = msgs_to_pos(loc_msgs)
        loc_times = msgs_to_times(loc_msgs)
        loc_idxs = np.argsort(loc_times)
        loc_positions = loc_positions[loc_idxs]
        loc_times = loc_times[loc_idxs]
        # loc_positions[:, 2] += 191
        loc_frombag = True

    rospy.loginfo("Loaded {:d} localization positions".format(len(loc_positions)))
    
    # #} end of load localization

    # #{ load ground truth
    
    if os.path.isfile(gt_out_fname):
        rospy.loginfo("File {:s} found, loading it".format(gt_out_fname))
        min_positions, gt_times = load_csv_data(gt_out_fname)
        rospy.loginfo("Loaded {:d} positions from {:s}".format(len(min_positions), gt_out_fname))
        # min_positions2, gt_times2 = load_csv_data(gt_out_fname2)
        # rospy.loginfo("Loaded {:d} positions from {:s}".format(len(min_positions2), gt_out_fname2))
        # min_positions3, gt_times3 = load_csv_data(gt_out_fname3)
        # rospy.loginfo("Loaded {:d} positions from {:s}".format(len(min_positions3), gt_out_fname3))
        # min_positions = np.vstack((min_positions, min_positions2, min_positions3))
        # gt_times = np.hstack((gt_times, gt_times2, gt_times3))
        # loc_positions = transform_gt(loc_positions, [0, 0, -0.52, 0, 0, 0])
        # loc_positions = loc_positions - loc_positions[0, :] + min_positions[0, :]
        gt_frombag = False
    else:
        gt_msgs = load_rosbag_msgs(gt_bag_fname, gt_topic_name, skip_time=0, skip_time_end=0)
        # else:
        #     rospy.loginfo("Input file loaded, processing")
        if gt_msgs is None:
            exit(1)
    
        start_time = loc_times[0]
        end_time = loc_times[-1]
        print("loc:", start_time, end_time)
        print("gt:", gt_msgs[0].header.stamp.to_sec(), gt_msgs[-1].header.stamp.to_sec())
        gt_msgs = cut_from(gt_msgs, rospy.Time.from_sec(start_time))
        gt_msgs = cut_to(gt_msgs, rospy.Time.from_sec(end_time))
        print("loc:", start_time, end_time)
        print("gt:", gt_msgs[0].header.stamp.to_sec(), gt_msgs[-1].header.stamp.to_sec())
    
        gt_positions = msgs_to_pos(gt_msgs)
        gt_times = msgs_to_times(gt_msgs)
        gt_idxs = np.argsort(gt_times)
        gt_positions = gt_positions[gt_idxs]
        gt_times = gt_times[gt_idxs]
        # rospy.logwarn("filtering GT positions")
    
        # plt.plot(gt_times - start_time, gt_positions[:, 2], 'rx')
        # gt_positions = filter_gt_pos(gt_positions)
        # plt.plot(gt_times - start_time, gt_positions[:, 2], 'gx')
        # plt.show()
        # print("gt:", gt_times[0], gt_times[-1])
        # loc_positions = transform_gt(loc_positions, [1.57, 3.14, 1.57, 0, 0, 0], inverse=True)
        # rot_positions = transform_gt(gt_positions, [0, 0, -1.17, 0, 0, 0])
        # rot_positions = transform_gt(gt_positions, [0, 0.06, 0, 0, 0, 0])
    
        min_positions = gt_positions
    
        if tf_frombag:
            zero_pos = loc_positions[0, :]
            loc_positions = loc_positions - zero_pos
            gt_zero_pos = gt_positions[0, :]
            gt_positions = gt_positions - gt_zero_pos
            # Find the transformed positions of GT which minimize RMSE with the localization
            loc_positions_time_aligned = time_align(gt_times, loc_positions, loc_times)
            TP_mask = ~np.isnan(loc_positions_time_aligned[:, 0])
            # TP_mask = np.logical_and(TP_mask, loc_positions_time_aligned[:, 0] < 6.0)
            # TP_mask = np.logical_and(TP_mask, loc_positions_time_aligned[:, 1] > -100.0)
            FP_mask1 = np.logical_and.reduce((
                                        loc_positions_time_aligned[:, 1] > 2.0,
                                        loc_positions_time_aligned[:, 2] < -3.0
            ))
            FP_mask2 = np.logical_and.reduce((
                                        loc_positions_time_aligned[:, 0] < -2.0,
                                        loc_positions_time_aligned[:, 2] < -3.0
            ))
            # FP_mask2 = np.logical_and.reduce((
            #                             loc_positions_time_aligned[:, 2] < -2.9,
            #                             loc_positions_time_aligned[:, 0] > 2.0, loc_positions_time_aligned[:, 0] < 4.0,
            #                             loc_positions_time_aligned[:, 1] > 1.5, loc_positions_time_aligned[:, 1] < 4.0
            # ))
            # FP_mask3 = np.logical_and.reduce((
            #                             loc_positions_time_aligned[:, 2] > 15.0,
            # ))
            FP_mask = np.logical_or.reduce((FP_mask1, FP_mask2))
            TP_mask = np.logical_and(TP_mask, ~FP_mask)
            loc_pos = loc_positions_time_aligned[TP_mask, :]
    
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.grid()
            ax.plot(loc_pos[:, 0], loc_pos[:, 1], loc_pos[:, 2], 'bo')
            # ax.plot(loc_positions_time_aligned[~TP_mask, 0], loc_positions_time_aligned[~TP_mask, 1], loc_positions_time_aligned[~TP_mask, 2], 'gx')
            ax.plot(gt_positions[:, 0], gt_positions[:, 1], gt_positions[:, 2], 'rx')
            plt.show()

            gt_pos = gt_positions[TP_mask, :]
            tf = find_min_tf(gt_pos, loc_pos, FP_error, only_rot=False)
            min_positions = transform_gt(gt_positions, tf)
            min_positions += zero_pos
            loc_positions += zero_pos
            tf[3:6] += zero_pos
            tf = np.hstack((tf, gt_zero_pos))
        else:
            rospy.logwarn("using cached TF:")
            print(tf)
            rot_pos = tf[6:9]
            tf_tmp = tf[0:6]
            gt_positions -= rot_pos
            min_positions = transform_gt(gt_positions, tf_tmp)
            # min_positions += rot_pos
    
        # min_positions = min_positions - min_positions[0, :] + loc_positions[0, :]
        # min_positions = rot_positions
        gt_frombag = True
    
    rospy.loginfo("Loaded {:d} ground truth positions".format(len(min_positions)))
    
    # #} end of load ground truth

    if True:
        rot_pos = tf[6:9]
        tf_tmp = tf[0:6]
        obs_positions -= rot_pos
        obs_positions = transform_gt(obs_positions, tf_tmp)
        # obs_positions += rot_pos

    if loc_frombag:
        subtract(loc_positions, loc_times, obs_positions, obs_times)

    if gt_frombag:
        subtract(min_positions, gt_times, obs_positions, obs_times)
    
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.grid()
    # ax.plot(loc_positions[:, 0], loc_positions[:, 1], loc_positions[:, 2], 'bo')
    # ax.plot(obs_positions[:, 0], obs_positions[:, 1], obs_positions[:, 2], 'gx')
    # ax.plot(min_positions[:, 0], min_positions[:, 1], min_positions[:, 2], 'rx')
    # plt.show()

    # rot_pos = tf[6:9]
    # tf_tmp = tf[0:6]
    # obs_positions -= rot_pos
    # obs_positions = transform_gt(obs_positions, tf_tmp)

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.grid()
    # ax.plot(min_positions[:, 0], min_positions[:, 1], min_positions[:, 2], 'rx')
    # ax.plot(obs_positions[:, 0], obs_positions[:, 1], obs_positions[:, 2], 'gx')
    # ax.plot(loc_positions[:, 0], loc_positions[:, 1], loc_positions[:, 2], 'bo')
    # plt.show()
    
    print("loc:", loc_times[0], loc_times[-1], loc_positions[0, :])
    print("gt:", gt_times[0], gt_times[-1], min_positions[0, :])

    tposs = get_positions_time(min_positions, gt_times, loc_positions, loc_times)
    TPs, TNs, FPs, FNs, errors = calc_statistics(min_positions, tposs, FP_error)
    rospy.loginfo("TPs, TNs, FPs, FNs: {:d}, {:d}, {:d}, {:d}".format(TPs, TNs, FPs, FNs))

    precision = TPs/float(TPs + FPs)
    recall = TPs/float(TPs + FNs)
    rospy.loginfo("precision, recall: {:f}, {:f}".format(precision, recall))

    # error_over_distance = error/min_positions[:, 2]
    TP_mask = errors < FP_error
    TP_mask = np.logical_and(TP_mask, tposs[:, 0] < 6.0)
    TP_mask = np.logical_and(TP_mask, tposs[:, 1] > -100.0)
    TP_mask = np.logical_and(TP_mask, tposs[:, 2] > -2.0)
    TP_errors = errors[TP_mask]
    rospy.loginfo("Max. error: {:f}".format(np.max(TP_errors)))
    rospy.loginfo("Mean error: {:f}, std.: {:f}".format(np.mean(TP_errors), np.std(TP_errors)))
    # TP_mask = np.ones((len(errors),), dtype=bool)
    dists = np.linalg.norm(min_positions, axis=1)
    est_dists = np.linalg.norm(tposs[TP_mask], axis=1)
    TP_dists = dists[TP_mask]

    plt.plot(TP_dists, TP_errors)
    plt.show()

    # _, pos_hist_edges = np.histogram(dists, bins=20)
    # pos_hist, _ = np.histogram(dists, bins=pos_hist_edges)
    # pos_hist = np.array(pos_hist, dtype=float)
    # TP_hist, _  = np.histogram(TP_dists, bins=pos_hist_edges)
    # TP_hist = np.array(TP_hist, dtype=float)
    # TP_probs = TP_hist/pos_hist
    # # TP_probs = calc_probs(TP_hist, pos_hist)
    # bin_width = pos_hist_edges[1] - pos_hist_edges[0]
    # pos_hist_centers = pos_hist_edges[0:-1] + bin_width/2
    # # dist = np.linspace(np.min(dists), np.max(dists), len(errors))
    
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.grid()
    # ax.plot(tposs[TP_mask, 0], tposs[TP_mask, 1], tposs[TP_mask, 2], 'bo')
    # ax.plot(obs_positions[:, 0], obs_positions[:, 1], obs_positions[:, 2], 'gx')
    # ax.plot(min_positions[:, 0], min_positions[:, 1], min_positions[:, 2], 'rx')
    # plt.show()

    # dists = dists[dists <= 18.0]
    _, err_avg_edges = np.histogram(dists, bins=15)
    err_bin_width = err_avg_edges[1] - err_avg_edges[0]
    err_avg_centers = err_avg_edges[0:-1] + err_bin_width/2
    err_avg = len(err_avg_centers)*[None]
    for it in range(0, len(err_avg_edges)-1):
        low = err_avg_edges[it]
        high = err_avg_edges[it+1]
        idxs = np.logical_and(TP_dists > low, TP_dists < high)
        err_avg[it] = np.mean(TP_errors[idxs])

    # fig = plt.figure()
    # ax = fig.gca()
    # ax.set_aspect('equal')
    # plt.plot(TP_dists, est_dists, 'rx')
    # plt.plot(TP_dists, TP_dists, 'g')
    # plt.title("estimated distance over distance")
    # plt.show()

    fig = plt.figure()
    plt.plot(err_avg_centers, err_avg, 'r')
    # plt.plot(TP_dists, TP_errors, 'rx')
    # plt.plot(pos_hist_centers, TP_probs, 'b')
    # plt.plot(TP_dists, TP_errors, 'g.')
    # # plt.plot(pos_hist_centers, TP_hist, 'g.')
    # # plt.plot(pos_hist_centers, pos_hist, '.')
    plt.title("Localization error over distance")
    plt.show()
    put_errs_to_file(err_avg_centers, err_avg, dist_err_out_fname)
    # put_errs_to_file(pos_hist_centers, TP_probs, TP_prob_out_fname)

    # # n_FPs = np.sum(np.isnan(errors))
    # # print(n_FPs)
    # # TP_mask = ~np.isnan(errors)
    # # TP_mask = np.ones(min_positions.shape[0], dtype=bool)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot(loc_positions[:, 0], loc_positions[:, 1], loc_positions[:, 2], 'g.')
    # ax.plot([loc_positions[0, 0]], [loc_positions[0, 1]], [loc_positions[0, 2]], 'gx')
    # ax.plot([loc_positions[-1, 0]], [loc_positions[-1, 1]], [loc_positions[-1, 2]], 'go')

    # rospy.loginfo("Done loading positions")
    # loc_positions = transform_gt(gt_positions, [0, 0, -1.17, 0, 0, 0])
    # loc_positions = loc_positions - loc_positions[0, :] + min_positions[0, :] + np.array([1, -1.1, 0])
    if tf_frombag:
        rospy.loginfo('Saving TF [{:f}, {:f}, {:f}, {:f}, {:f}, {:f}], [{:f}, {:f}, {:f}] to CSV: {:s}'.format(tf[0], tf[1], tf[2], tf[3], tf[4], tf[5], tf[6], tf[7], tf[8], tf_out_fname))
        put_tf_to_file(tf, tf_out_fname)
    if loc_frombag:
        rospy.loginfo('Saving localizations to CSV: {:s}'.format(loc_out_fname))
        put_to_file(loc_positions, loc_times, loc_out_fname)
    if gt_frombag:
        rospy.loginfo('Saving ground truths to CSV: {:s}'.format(gt_out_fname))
        put_to_file(min_positions, gt_times, gt_out_fname)

    # # ax.plot(min_positions[:, 0], min_positions[:, 1], min_positions[:, 2], 'r')
    # # ax.plot(min_positions[:, 0], min_positions[:, 1], min_positions[:, 2], 'b.')
    # ax.plot(min_positions[TP_mask, 0], min_positions[TP_mask, 1], min_positions[TP_mask, 2], 'b.')
    # ax.plot(min_positions[~TP_mask, 0], min_positions[~TP_mask, 1], min_positions[~TP_mask, 2], 'rx')
    # ax.plot([min_positions[0, 0]], [min_positions[0, 1]], [min_positions[0, 2]], 'rx')
    # ax.plot([min_positions[-1, 0]], [min_positions[-1, 1]], [min_positions[-1, 2]], 'ro')
    # ax.set_xlabel('x (m)')
    # ax.set_ylabel('y (m)')
    # ax.set_zlabel('z (m)')
    # ax.set_aspect('equal')
    # plt.xlim([-20, 20])
    # plt.ylim([-30, 10])
    # plt.show()

if __name__ == '__main__':
    main()
