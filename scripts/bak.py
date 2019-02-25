
class diff_calculator:
    def __init__(self, positions1, times1, positions2, times2, FP_error):
        self.positions1 = positions1
        self.times1 = times1
        self.positions2 = positions2
        self.times2 = times2
        self.FP_error = FP_error

    def calc_one_diff(self, it):
        params = np.zeros((6,))
        params[it] = self.dtransf
        pts = transform_gt(self.positions1, params)
        tot_err, N = calc_error(pts, self.times1, self.positions2, self.times2, self.FP_error)
        return tot_err/float(N)

    def calc_avg_error_diff(self):
        self.dtransf = 0.1
        E1 = list()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for out in executor.map(self.calc_one_diff, range(0, 6)):
                # put results into correct output list
                E1.append(out)
        E1 = np.array(E1)

        self.dtransf = -0.1
        E2 = list()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for out in executor.map(self.calc_one_diff, range(0, 6)):
                # put results into correct output list
                E2.append(out)
        E2 = np.array(E1)

        dE = (E2 - E1)/(2*self.dtransf)
        return dE


def find_min_positions(positions, times, to_positions, to_times, FP_error):
    lam_pos = 2e-3
    lam_ang = 2e-2
    # N_SAMPLES = np.min([10000, len(positions)])
    for it in range(0, 160):
        if rospy.is_shutdown():
            break
        print("iteration {:d}:".format(it))
        # idxs = np.random.choice(len(positions)-1, N_SAMPLES, replace=False)
        # rand_poss = positions[idxs, :]
        # rand_tims = times[idxs]
        rand_poss = positions
        rand_tims = times
        tot_err, N = calc_error(rand_poss, rand_tims, to_positions, to_times, FP_error)
        E0 = tot_err/float(N)
        print("avg. error: {:f} (from {:d} points)".format(E0, N))
        DC = diff_calculator(rand_poss, rand_tims, to_positions, to_times, FP_error)
        dE = DC.calc_avg_error_diff()
