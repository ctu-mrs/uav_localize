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
import csv

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

def transform_gt(gt_pos, params):
    angles = params[0:3]
    shift = params[3:6]
    R = rotation(angles)
    # shift = np.array([20, -20, -1]).transpose()
    rotated = np.dot(gt_pos, R)
    rotated += shift
    return rotated

def find_closest(time, times):
    closest_it = 0
    for it in range(0, len(times)):
        cur_diff = abs(time - times[it])
        closest_diff = abs(time - times[closest_it])
        # print(cur_diff, closest_diff)
        if cur_diff <= closest_diff:
            closest_it = it
        else:
            break
    return closest_it

def calc_errors(positions1, times1, positions2, times2):
    max_dt = 0.05
    errors = len(positions1)*[None]
    for it in range(0, len(positions1)):
        time1 = times1[it]
        closest_it = find_closest(time1, times2)
        if abs(times2[closest_it] - time1) > max_dt:
            continue
        errors[it] = np.linalg.norm(positions1[it, :] - positions2[closest_it, :])
    return errors

def calc_error(positions1, times1, positions2, times2, FP_error):
    errors = calc_errors(positions1, times1, positions2, times2)
    tot_error = 0
    N = 0
    for err in errors:
        if err is not None and err < FP_error:
            tot_error += err
            N += 1
    return (tot_error, N)

def calc_avg_error_diff(positions1, times1, positions2, times2, FP_error):
    E1 = np.zeros((6,))
    dtransf = 0.1
    for it in range(0, 6):
        params = np.zeros((6,))
        params[it] = dtransf
        pts = transform_gt(positions1, params)
        tot_err, N = calc_error(pts, times1, positions2, times2, FP_error)
        E1[it] = tot_err/float(N)
    E2 = np.zeros((6,))
    for it in range(0, 6):
        params = np.zeros((6,))
        params[it] = -dtransf
        pts = transform_gt(positions1, params)
        tot_err, N = calc_error(pts, times1, positions2, times2, FP_error)
        E2[it] = tot_err/float(N)
    dE = (E2 - E1)/(2*dtransf)
    return dE


def find_min_positions(positions, times, to_positions, to_times, FP_error):
    lam_pos = 2e-3
    lam_ang = 2e-2
    for it in range(0, 160):
        if rospy.is_shutdown():
            break
        print("iteration {:d}:".format(it))
        tot_err, N = calc_error(positions, times, to_positions, to_times, FP_error)
        E0 = tot_err/float(N)
        print("avg. error: {:f} (from {:d} points)".format(E0, N))
        dE = calc_avg_error_diff(positions, times, to_positions, to_times, FP_error)
        print("avg. error diff [{:f}, {:f}, {:f}, {:f}, {:f}, {:f}]:".format(dE[0], dE[1], dE[2], dE[3], dE[4], dE[5]))
        transf = dE
        transf[0:3] *= lam_pos
        transf[3:6] *= lam_ang
        positions = transform_gt(positions, transf)
        if it % 40 == 0:
            lam_pos /= 2
            lam_ang /= 2
    return positions
        
def put_to_file(positions, times, fname):
    with open(fname, 'w') as ofhandle:
        ofhandle.write("x,y,z,time\n")
        for it in range(0, len(positions)):
            ofhandle.write("{:f},{:f},{:f},{:f}\n".format(positions[it, 0], positions[it, 1], positions[it, 2], times[it]))

def calc_statistics(positions1, times1, positions2, times2, FP_error):
    errors = calc_errors(positions1, times1, positions2, times2)
    TPs = 0
    TNs = 0
    FPs = 0
    FNs = 0

    for err in errors:
        if err is None:
            FNs += 1
        elif err > FP_error:
            FNs += 1
            FPs += 1
        else:
            TPs += 1

    errors = np.array(errors, dtype=float)
    nn_errors = errors[~np.isnan(errors)]
    rospy.loginfo("Max. error: {:f}".format(np.max(nn_errors)))
    rospy.loginfo("Mean error: {:f}, std.: {:f}".format(np.mean(nn_errors), np.std(nn_errors)))
    return (TPs, TNs, FPs, FNs)

def main():
    rospy.init_node('localization_evaluator', anonymous=True)
    # out_fname = rospy.get_param('~output_filename')
    # in_fname = rospy.get_param('~input_filename')
    loc_bag_fname = rospy.get_param('~localization_bag_name')
    gt_bag_fname = rospy.get_param('~ground_truth_bag_name')
    loc_topic_name = rospy.get_param('~localization_topic_name')
    gt_topic_name = rospy.get_param('~ground_truth_topic_name')
    loc_out_fname = rospy.get_param('~localization_out_fname')
    gt_out_fname = rospy.get_param('~ground_truth_out_fname')

    # msgs = load_pickle(in_fname)
    FP_error = 7.0 # meters

    if os.path.isfile(loc_out_fname) and os.path.isfile(gt_out_fname):
        rospy.loginfo("Files {:s} and {:s} found, loading them".format(loc_out_fname, gt_out_fname))
        loc_positions, loc_times = load_csv_data(loc_out_fname)
        min_positions, gt_times = load_csv_data(gt_out_fname)
    else:
        rospy.loginfo("Loading data from rosbags {:s} and {:s}".format(loc_bag_fname, gt_bag_fname))
        # if msgs is None:
        skip_time = 20
        skip_time_end = 60
        # rospy.loginfo("Input file not valid, loading from rosbag")
        loc_msgs = load_rosbag_msgs(loc_bag_fname, loc_topic_name, skip_time=skip_time, skip_time_end=skip_time_end)
        gt_msgs = load_rosbag_msgs(gt_bag_fname, gt_topic_name, skip_time=skip_time, skip_time_end=skip_time_end)
        # else:
        #     rospy.loginfo("Input file loaded, processing")
        if loc_msgs is None or gt_msgs is None:
            exit(1)

        start_time = loc_msgs[0].header.stamp
        end_time = loc_msgs[-1].header.stamp
        gt_msgs = cut_from(gt_msgs, start_time)
        gt_msgs = cut_to(gt_msgs, end_time)

        rospy.loginfo("Loaded {:d} localization messages".format(len(loc_msgs)))
        rospy.loginfo("Loaded {:d} ground truth messages".format(len(gt_msgs)))
    
        loc_positions = msgs_to_pos(loc_msgs)
        loc_times = msgs_to_times(loc_msgs)
        gt_positions = msgs_to_pos(gt_msgs)
        gt_times = msgs_to_times(gt_msgs)
        rot_positions = transform_gt(gt_positions, [0, 0, -1.17, 0, 0, 0])
        rot_positions = rot_positions - rot_positions[0, :] + loc_positions[0, :]
        # Find the transformed positions of GT which minimize RMSE with the localization
        min_positions = find_min_positions(rot_positions, gt_times, loc_positions, loc_times, FP_error)

        put_to_file(loc_positions, loc_times, loc_out_fname)
        put_to_file(min_positions, gt_times, gt_out_fname)

    rospy.loginfo("Done loading positions")

    TPs, TNs, FPs, FNs = calc_statistics(min_positions, gt_times, loc_positions, loc_times, FP_error)
    rospy.loginfo("TPs, TNs, FPs, FNs: {:d}, {:d}, {:d}, {:d}".format(TPs, TNs, FPs, FNs))

    precision = TPs/float(TPs + FPs)
    recall = TPs/float(TPs + FNs)
    rospy.loginfo("precision, recall: {:f}, {:f}".format(precision, recall))

    error = np.array(calc_errors(min_positions, gt_times, loc_positions, loc_times), dtype=float)
    # error_over_distance = error/min_positions[:, 2]
    # dist = np.linspace(np.min(min_positions[:, 2]), np.max(min_positions[:, 2]), len(error))
    # plt.plot(dist, error)
    # plt.show()

    TP_mask = error < FP_error
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(loc_positions[:, 0], loc_positions[:, 1], loc_positions[:, 2], 'g.')
    ax.plot([loc_positions[0, 0]], [loc_positions[0, 1]], [loc_positions[0, 2]], 'gx')
    ax.plot([loc_positions[-1, 0]], [loc_positions[-1, 1]], [loc_positions[-1, 2]], 'go')

    # ax.plot(min_positions[:, 0], min_positions[:, 1], min_positions[:, 2], 'r')
    ax.plot(min_positions[TP_mask, 0], min_positions[TP_mask, 1], min_positions[TP_mask, 2], 'b.')
    ax.plot(min_positions[~TP_mask, 0], min_positions[~TP_mask, 1], min_positions[~TP_mask, 2], 'rx')
    ax.plot([min_positions[0, 0]], [min_positions[0, 1]], [min_positions[0, 2]], 'rx')
    ax.plot([min_positions[-1, 0]], [min_positions[-1, 1]], [min_positions[-1, 2]], 'ro')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    ax.set_aspect('equal')
    plt.show()

if __name__ == '__main__':
    main()
