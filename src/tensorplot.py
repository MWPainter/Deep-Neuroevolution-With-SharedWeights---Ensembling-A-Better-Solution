import os
import sys
import numpy as np
import seaborn as sns
import pandas as pd
import tensorflow as tf
import matplotlib
from matplotlib import pyplot as plt
from scipy import signal

# This is needed to have TrueType fonts.
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

"""
TensorPlot: converting TensorBoard summaries into nice plots with SeaBoarn.
"""

"""
Section: Generic helpers
"""


def _listify(val):
    """
    If val is not a list, return the list [val], otherwise, return val.
    """
    if type(val) is str or not hasattr(val, '__len__'):
        return [val]
    return val


def _si(arr, i, j=None):
    """
    Saturated indexinf of an array.
    :param arr: An array to index.
    :param i: An index that may be out of bounds.
    :param j: A second index that isn't bounds checked.
    :return: arr[i], or arr[0] if i < 0 or arr[n-1] if i >= n.
    """
    if i < 0:
        i = 0
    elif i >= len(arr):
        i = len(arr) - 1
    if j is not None:
        return arr[i, j]
    return arr[i]


def _log_scale_(curve):
    """Puts curve on log x scale, assumes curve is a np.array of shape (N,2)"""
    curve[:, 0] = np.log10(curve[:, 0])
    return curve


def _crop_points(points, xmin, xmax, ymin, ymax):
    """Considers a set of points 'points', with shape (N,2) and crops them to 
    the ranges [xmin,xmax] and [ymin,ymax]. Treats the points as points and not 
    as a curve. If this eliminates points in the middle of a curve it would 
    change how it looks (consider cropping a sine curve to [0,1] on yaxis)."""
    indxs = ((points[:, 0] >= xmin)
             & (points[:, 0] <= xmax)
             & (points[:, 1] >= ymin)
             & (points[:, 1] <= ymax))
    return points[indxs]


"""
Section: wrappers for reading TensorBoard summaries and for plotting with Seaboarn.
"""


def _clean_sequence(logdir, scalar_name):
    """
    For a given scalar with name 'scalar_name' we clean the values in the summaries in 'logdir'. Clean means that we
    remove any out of sequence values recorded for the scalar.

    :param logdir: The filepath of the TB log directory
    :param scalar_name: The scalar that we wish to clean values for in this log
    """
    pass


def _remove_discontinuity(logdir, scalar_name):
    """
    For a given scalar with name 'scalar_name' we remove a discontinuity from the plot. Sometime in restarting plots
    values are reset. This method assumes that the value is monotonically increasing, and that the value should never
    drop. If it does drop by some value v, then it shifts the rest of the curve upwards by 'v'.

    :param logdir: The filepath of the TB log directory
    :param scalar_name: The scalar that we wish to clean values for in this log
    """
    pass


def _read_sequence(logdir, scalar_name):
    """
    Reads from the TensorBoard summery to produce a list of

    :param logdir: The filepath of the TB log directory
    :param scalar_name: The scalar that we wish to get the values for
    :return: np.array with shape (N,2) for the sequence in
    """

    """
    summary_iterators = [EventAccumulator(logfile).Reload()]
    print (summary_iterators)
    out = []

    steps = [[e.step] for e in summary_iterators[0].Scalars(scalar_name)]

    for events in zip(*[acc.Scalars(scalar_name) for acc in summary_iterators]):

        assert len(set(e.step for e in events)) == 1
        print(e.value)
        out.append([e.value for e in events])

    result = np.array(np.concatenate([np.array(steps), np.array(out)], axis=1))
    print (result)
    """

    out = []
    for (dirname, _, filenames) in os.walk(logdir):
        for filename in filenames:
            logfile = os.path.join(dirname, filename)
            for event in tf.train.summary_iterator(logfile):
                for value in event.summary.value:
                    if value.tag == scalar_name:
                        out.append([value.simple_value])

    steps = [[i] for i in range(len(out))]
    result = np.array(np.concatenate([np.array(steps), np.array(out)], axis=1))
    return result


def _read_sequence_csv(csv_file):
    points = []
    with open(csv_file, "r") as f:
        for line in f:
            try:
                _, xstr, ystr = line.split(",")
                points.append([float(xstr), float(ystr)])
            except:
                pass  # first line in the file... try and ask forgiveness...
    result = np.array(points)
    return result


def _add_moving_averages(sequence, window_size):
    df = pd.DataFrame(data=sequence, columns=["x", "y"])
    rolling_mean = df.rolling(window=window_size, on='x').mean()  # .ewm(alpha=0.2)
    rolling_std = df.rolling(window=window_size, on='x').std()

    new_df = pd.DataFrame({'x': df['x'], 'y': df['y'],
                           'y_rmean': rolling_mean['y'].values,
                           'y_rstd': rolling_std['y'].values})

    new_df['y_ewma'] = new_df['y'].ewm(span=window_size, adjust=False).mean()

    return new_df


def _plot_sequence(df, linestyle, label, avg='rolling'):
    """
    Adds the plot 'sequence' with color 'color' to the current Seaboarn figure.

    :param sequence: A np.array with shape (N,2) for the sequence we wish to add to the current figure.
    :param color: The color that we wish to plot with.
    """
    # df = pd.DataFrame(data=sequence, columns=["x","y"])

    # cis = np.asarray((df['y_rmean']-df['y_rstd']*1000,
    #                   df['y_rmean']+df['y_rstd']*1000))
    if (avg == 'rolling'):
        sns.lineplot(x='x', y='y_rmean', data=df)
    else:
        sns.lineplot(x='x', y='y_ewma', data=df)  # , linestyle=linestyle)
    # plt.plot(sequence[:,0], sequence[:,1], linestyle=linestyle, label=label)


def _save_current_fig(out_filename):
    """
    Saves the current seaboarn figure in the file with name 'out_filename'.
    """
    plt.savefig(out_filename)


"""
Section: TensorPlot processing.
"""


def _compute_parametric_curve(seq1, seq2):
    """
    Produces a paramteric sequence. If seq1 is [(x1,y1), ..., ] and seq2 is [(x'1, y'1), ..., ], then we compute a
    sequence [(a1, b1), ..., ]. Each pair (ai, bi) is equal to some (yi,y'j) where xi=x'j.

    We assume that x1 <= x2 <= x3 <= ... and that x'1 <= x'2 <= x'3 <= ...

    As the above rule can lead to some data being missed, we allow for the data to be made complete using one of three
    options:
    - ignore = we should ignore any values in y that don't have a corresponding y'
    - use_last = assuming that we're parameterised by time,
    - linear_interpolate = use a linear interpolation (or the same as use_last if we have run out of values to
                interpolate with)

    :param seq1: The first sequence to produce the parametric curve. A np.array with shape (N,2).
    :param seq2: The second sequence to produce the parametric curve. A np.array with shape (M,2).
    :param missing_value_completion: How we should complete any missing values for the latent parameters
    :return: The parametric curve, an np.array with shape (N',2)
    """
    N = seq1.shape[0]
    M = seq2.shape[0]
    i = j = 0
    parametric_points = []

    # rescale
    tx = seq1[-1, 0]
    ty = seq2[-1, 0]
    seq2[:, 0] = seq2[:, 0] * tx / ty

    j = 0
    for j in range(M):
        while i < N and _si(seq1, i + 1, 0) < _si(seq2, j, 0):
            i += 1
        new_point = [_si(seq1, i, 1), _si(seq2, j, 1)]
        parametric_points.append(new_point)
    return np.array(parametric_points)

    # while i < M and j < N:
    #     if _si(seq1,i,0) == _si(seq2,j,0):
    #         new_point = (_si(seq1,i,1), _si(seq2,j,1))
    #         parametric_points.append(new_point)
    #         i += 1
    #         j += 1
    #
    #     elif _si(seq1, i, 0) < _si(seq2, j, 0):
    #         new_point = None
    #         if missing_value_completion == "use_last":
    #             new_point = (_si(seq1,i,1), _si(seq2,j-1,1))
    #         elif missing_value_completion != "linear_interpolate":
    #             new_point = (_si(seq1,i,1), _interpolate_value(_si(seq2,j-1), _si(seq2,j), _si(seq1,i,0)))
    #         if new_point is not None:
    #             parametric_points.append(new_point)
    #         i += 1
    #
    #     else: # seq1[i,0] > seq2[j,0]:
    #         new_point = None
    #         if missing_value_completion == "use_last":
    #             new_point = (_si(seq1,i-1,1), _si(seq2,j,1))
    #         elif missing_value_completion != "linear_interpolate":
    #             new_point = (_interpolate_value(_si(seq1,i-1), _si(seq1,i), _si(seq2,j,0)), _si(seq2,j,1))
    #         if new_point is not None:
    #             parametric_points.append(new_point)
    #         j += 1
    #
    # print(np.array(parametric_points)[0,0])
    # return np.array(parametric_points)


def _interpolate_value(p1, p2, u):
    """
    If p1 = (u1,v1) and p2 = (u2,v2) then return v = v1 + (u-u1)/(u2-u1) * (v2-v1)
    """
    u1, v1 = p1
    u2, v2 = p2
    return v1 + (u - u1) / (u2 - u1) * (v2 - v1)


"""
Section: TensorPlot interface
"""


def normal_plots(out_filename, event_filename_scalar_pair, xaxis_name, yaxis_name, linestyles, labels, window_size=500,
                 xmin=-np.inf, xmax=np.inf, ymin=-np.inf, ymax=np.inf, log_x_axis=False):
    """
       Reads in the sequence values recorded for each scalar in 'scalar_names' and plots it on the same graph.

       :param logdirs: The filepath(s) of the TB log directory
       :param out_filename: The name of the file to save an image of the figure at
       :param sclar_names: The scalars to plot
       """
    plt.figure()
    sns.set(style="darkgrid")


    scalar_filenames = _listify(event_filename_scalar_pair)
    linestyles = _listify(linestyles)
    labels = _listify(labels)
    # logdirs = _listify(logdirs)
    # if len(logdirs) == 1:
    #     logdirs = logdirs * len(scalar_names)
    # assert len(logdirs) == len(scalar_names), "Invalid number of logdirs provided"

    for i in range(len(event_filename_scalar_pair)):
        # logdir = logdirs[i]
        event_filename = event_filename_scalar_pair[i][0]
        scalar = event_filename_scalar_pair[i][1]
        scalar_sequence = _read_sequence(event_filename, scalar)
        scalar_sequence = _crop_points(scalar_sequence, xmin, xmax, ymin, ymax)
        scalar_sequence = _add_moving_averages(scalar_sequence, window_size=window_size)
        if log_x_axis:
            scalar_sequence = _log_scale_(scalar_sequence)
        _plot_sequence(scalar_sequence, linestyles[i], labels[i], avg='ewma')
    plt.xlabel(xaxis_name)
    plt.ylabel(yaxis_name)
    plt.gca().legend(loc='lower right')
    _save_current_fig(out_filename)


# def normal_plots(out_filename, scalar_filenames, xaxis_name, yaxis_name, linestyles, labels):
#     """
#     Reads in the sequence values recorded for each scalar in 'scalar_names' and plots it on the same graph.

#     :param logdirs: The filepath(s) of the TB log directory
#     :param out_filename: The name of the file to save an image of the figure at
#     :param sclar_names: The scalars to plot
#     """
#     plt.figure()
#     sns.set(style="darkgrid")

#     scalar_filenames = _listify(scalar_filenames)
#     linestyles = _listify(linestyles)
#     labels = _listify(labels)
#     # logdirs = _listify(logdirs)
#     # if len(logdirs) == 1:
#     #     logdirs = logdirs * len(scalar_names)
#     # assert len(logdirs) == len(scalar_names), "Invalid number of logdirs provided"

#     for i in range(len(scalar_filenames)):
#         # logdir = logdirs[i]
#         scalar_filename = scalar_filenames[i]
#         scalar_sequence = _read_sequence_csv(scalar_filename)
#         _plot_sequence(scalar_sequence, linestyles[i], labels[i])
#     plt.xlabel(xaxis_name)
#     plt.ylabel(yaxis_name)
#     plt.gca().legend(loc='lower right')
#     _save_current_fig(out_filename)


def parametric_plots(out_filename, event_filename_scalar_pair_one, event_filename_scalar_pair_two, xaxis_name,
                     yaxis_name, linestyles, labels, num_points=None, window_size=500,
                     xmin=-np.inf, xmax=np.inf, ymin=-np.inf, ymax=np.inf, log_x_axis=False):
    """
    Produces a paramteric plot. If scalar1's plot is a sequence [(x1,y1), ..., ] and scalar2's plot is a sequence
    [(x'1, y'1), ..., ], then we compute a sequence [(a1, b1), ..., ] from the two scalars. Each pair (ai, bi) is
    equal to some (yi,y'j) where xi=x'j.

    As the above rule can lead to some data being missed, we allow for the data to be made complete using one of three
    options:
    - ignore = we should ignore any values in y that don't have a corresponding y'
    - use_last = assuming that we're parameterised by time,
    - linear_interpolate = use a linear interpolation (or the same as use_last if we have run out of values to
                interpolate with)

    Typical usage of this should be that we have two values y and y' that are recorded over a period of time, and
    rather than plotting y vs time and y' vs time, we wish to plot y vs y'.

    :param logdirs: The filepath of the TB log directory(s)
    :param out_filename: The name of the file to save an image of the figure at
    :param scalar_names_one: The first scalar from TB to use in the parametric plot
    :param scalar_names_two: The second scalar from TB to use in the parametric plot
    :param missing_value_completion: How we should complete any missing values for the latent parameters
    """
    plt.figure()
    sns.set(style="darkgrid")

    event_filename_scalar_pair_one = _listify(event_filename_scalar_pair_one)
    event_filename_scalar_pair_two = _listify(event_filename_scalar_pair_two)
    linestyles = _listify(linestyles)
    labels = _listify(labels)
    # logdirs = _listify(logdirs)
    # if len(logdirs) == 1:
    #     logdirs = logdirs * len(scalar_names_one)
    assert len(event_filename_scalar_pair_one) == len(
        event_filename_scalar_pair_two), "Invalid input to plot multiple parametric curves on a single axis."
    # assert len(logdirs) == len(scalar_names_two), "Invalid number of logdirs provided"

    for i in range(len(event_filename_scalar_pair_one)):
        # logdir = logdirs[i]
        event_filename_one = event_filename_scalar_pair_one[i][0]
        scalar_one = event_filename_scalar_pair_one[i][1]

        event_filename_two = event_filename_scalar_pair_two[i][0]
        scalar_two = event_filename_scalar_pair_two[i][1]

        scalar_sequence_one = _read_sequence(event_filename_one, scalar_one)
        scalar_sequence_two = _read_sequence(event_filename_two, scalar_two)

        parametric_sequence = _compute_parametric_curve(scalar_sequence_one, scalar_sequence_two)

        parametric_sequence = _crop_points(parametric_sequence, xmin, xmax, ymin, ymax)
        parametric_sequence = _add_moving_averages(parametric_sequence, window_size=window_size)

        if log_x_axis:
            parametric_sequence = _log_scale_(parametric_sequence)

        _plot_sequence(parametric_sequence, linestyles[i], labels[i], avg='ewma')
    plt.xlabel(xaxis_name)
    plt.ylabel(yaxis_name)
    plt.gca().legend(loc='lower right')
    _save_current_fig(out_filename)


# def parametric_plots(out_filename, scalar_filenames_one, scalar_filenames_two, xaxis_name, yaxis_name, linestyles, labels, num_points):
#     """
#     Produces a paramteric plot. If scalar1's plot is a sequence [(x1,y1), ..., ] and scalar2's plot is a sequence
#     [(x'1, y'1), ..., ], then we compute a sequence [(a1, b1), ..., ] from the two scalars. Each pair (ai, bi) is
#     equal to some (yi,y'j) where xi=x'j.

#     As the above rule can lead to some data being missed, we allow for the data to be made complete using one of three
#     options:
#     - ignore = we should ignore any values in y that don't have a corresponding y'
#     - use_last = assuming that we're parameterised by time,
#     - linear_interpolate = use a linear interpolation (or the same as use_last if we have run out of values to
#                 interpolate with)

#     Typical usage of this should be that we have two values y and y' that are recorded over a period of time, and
#     rather than plotting y vs time and y' vs time, we wish to plot y vs y'.

#     :param logdirs: The filepath of the TB log directory(s)
#     :param out_filename: The name of the file to save an image of the figure at
#     :param scalar_names_one: The first scalar from TB to use in the parametric plot
#     :param scalar_names_two: The second scalar from TB to use in the parametric plot
#     :param missing_value_completion: How we should complete any missing values for the latent parameters
#     """
#     plt.figure()
#     sns.set(style="darkgrid")

#     scalar_filenames_one = _listify(scalar_filenames_one)
#     scalar_filenames_two = _listify(scalar_filenames_two)
#     linestyles = _listify(linestyles)
#     labels = _listify(labels)
#     # logdirs = _listify(logdirs)
#     # if len(logdirs) == 1:
#     #     logdirs = logdirs * len(scalar_names_one)
#     assert len(scalar_filenames_one) == len(scalar_filenames_two), "Invalid input to plot multiple parametric curves on a single axis."
#     # assert len(logdirs) == len(scalar_names_two), "Invalid number of logdirs provided"

#     for i in range(len(scalar_filenames_one)):
#         # logdir = logdirs[i]
#         scalar_filename_one = scalar_filenames_one[i]
#         scalar_filename_two = scalar_filenames_two[i]
#         scalar_sequence_one = _read_sequence_csv(scalar_filename_one)
#         scalar_sequence_two = _read_sequence_csv(scalar_filename_two)
#         parametric_sequence = _compute_parametric_curve(scalar_sequence_one, scalar_sequence_two)
#         if num_points is not None:
#             parametric_sequence = parametric_sequence[:num_points]
#         _plot_sequence(parametric_sequence, linestyles[i], labels[i])
#     plt.xlabel(xaxis_name)
#     plt.ylabel(yaxis_name)
#     plt.gca().legend(loc='lower right')
#     _save_current_fig(out_filename)


def repair_discontinuity(logdir, scalar_name):
    """
    For a given scalar with name 'scalar_name' we remove a discontinuity from the plot. Sometime in restarting plots
    values are reset. This method assumes that the value is monotonically increasing, and that the value should never
    drop. If it does drop by some value v, then it shifts the rest of the curve upwards by 'v'.

    :param logdir: The filepath of the TB log directory
    :param scalar_name: The scalar that we wish to clean values for in this log
    """
    raise NotImplementedError()


def clean_scalar(logdir, scalar_name):
    """
    For a given scalar with name 'scalar_name' we clean the values in the summaries in 'logdir'. Clean means that we
    remove any out of sequence values recorded for the scalar.

    :param logdir: The filepath of the TB log directory
    :param scalar_name: The scalar that we wish to clean values for in this log
    """
    raise NotImplementedError()


if __name__ == "__main__":
    base_dir = sys.argv[1]

    # matplotlib.rcParams['ps.useafm'] = True
    # matplotlib.rcParams['pdf.use14corefonts'] = True
    # matplotlib.rcParams['text.usetex'] = True

    # # Test normal plots
    # imgfile = os.path.join(base_dir, "test_normal.pdf")

    # event_filename_scalar_pair = [
    #     ('tb_logs/tblogs/last_ednw_default_tb_log/Completely_Random_Init_Net2Net/events.out.tfevents.1552469288.dgj410', 'iter/train/valacc_1'),
    #     ('tb_logs/tblogs/last_ednw_default_tb_log/R2R_student/events.out.tfevents.1552424267.dgj410','iter/train/valacc_1')]

    # xaxis = "Epochs"
    # yaxis = "Validation Accuracy"
    # linestyles = ['-','-','-','-','-','-','-']
    # labels = ['Net2WiderNet', 'R2WiderR', 'RandomPad', 'ResNetCifar18(1/6)', 'Teacher', 'Teacher-NoResidual', 'NetMorph']
    # normal_plots_new(imgfile, event_filename_scalar_pair, xaxis, yaxis, linestyles, labels)

    # # Test parametric plots
    # imgfile = os.path.join(base_dir, "test_parametric.pdf")

    # event_filename_scalar_pair_one = [
    #     ('tb_logs/tblogs/last_ednw_default_tb_log/Completely_Random_Init_Net2Net/events.out.tfevents.1552469288.dgj410',
    #      'iter/train/valacc_1'),
    #     ('tb_logs/tblogs/last_ednw_default_tb_log/R2R_student/events.out.tfevents.1552424267.dgj410',
    #      'iter/train/valacc_1')]

    # event_filename_scalar_pair_two = [
    #     ('tb_logs/tblogs/last_ednw_default_tb_log/R2R_student/events.out.tfevents.1552424267.dgj410',
    #      'iter/train/valacc_1'),
    #     ('tb_logs/tblogs/last_ednw_default_tb_log/Completely_Random_Init_Net2Net/events.out.tfevents.1552469288.dgj410',
    #      'iter/train/valacc_1')]

    # xaxis = "Epochs"
    # yaxis = "Validation Accuracy"
    # linestyles = ['-', '-', '-', '-', '-']
    # labels = ['Net2WiderNet', 'R2WiderR', 'RandomPad', 'ResNetCifar18(1/6)', 'Teacher', 'Teacher-NoResidual',
    #           'NetMorph']
    # parametric_plots_new(imgfile, event_filename_scalar_pair_one, event_filename_scalar_pair_two,
    #                      xaxis, yaxis, linestyles, labels, num_points=100)

    # Expr 1 - SVHN / net2widernet tests, validation curves
    imgfile = os.path.join(base_dir, "svhn_net2widernet.pdf")

    #exp_dir = os.path.join(os.getenv("HOME"),
    #                       "Desktop/Neuroevolution/Deep-Neuroevolution-With-SharedWeights---Ensembling-A-Better-Solution/src/tb_logs/jade_tb_logs/tb_logs/paper_ewnw_default_tb_log")
    exp_dir = os.path.join(os.getenv("HOME"), "r2r/jade_tb_logs/tb_logs/paper_ewnw_default_tb_log")
    n2n_file = os.path.join(exp_dir, "Net2Net_student")
    r2r_file = os.path.join(exp_dir, "R2R_student")
    netmorph_file = os.path.join(exp_dir, "NetMorph_student")
    rand_pad_file = os.path.join(exp_dir, "RandomPadding_student")
    rand_init_file = os.path.join(exp_dir, "Completely_Random_Init")
    teacher_file = os.path.join(exp_dir, "teacher_w_residual")
    teacher_nr_file = os.path.join(exp_dir, "teacher_w_out_residual")

    scalars = [
        (n2n_file, 'iter/train/valacc_1'),
        (r2r_file, 'iter/train/valacc_1'),
        (netmorph_file, 'iter/train/valacc_1'),
        (rand_pad_file, 'iter/train/valacc_1'),
        (rand_init_file, 'iter/train/valacc_1'),
        (teacher_file, 'iter/train/valacc_1'),
        (teacher_nr_file, 'iter/train/valacc_1'),
    ]
    labels = ['Net2WiderNet', 'R2WiderR', 'NetMorph', 'RandomPad', 'ResNetCifar18(3/8)', 'ResNetCifar18(1/4)',
              'ResNetCifar18(1/4)(NoRes)']
    xaxis = "Iterations"
    yaxis = "Validation Accuracy"
    linestyles = ['-', '-', '-', '-', '-', '-', '-']
    normal_plots(imgfile, scalars, xaxis, yaxis, linestyles, labels)

    # Expr 2 - SVHN / r2widerr tests, validation curves
    imgfile = os.path.join(base_dir, "svhn_r2widerr_flops.pdf")

    #exp_dir = os.path.join(os.getenv("HOME"),
     #                      "Desktop/Neuroevolution/Deep-Neuroevolution-With-SharedWeights---Ensembling-A-Better-Solution/src/tb_logs/jade_tb_logs/tb_logs/paper_ewnw_default_tb_log")
    exp_dir = os.path.join(os.getenv("HOME"), "r2r/jade_tb_logs/tb_logs/paper_ewrw_default_tb_log")
    n2n_file = os.path.join(exp_dir, "Net2Net_student")
    r2r_file = os.path.join(exp_dir, "R2R_student")
    netmorph_file = os.path.join(exp_dir, "NetMorph_student")
    rand_pad_file = os.path.join(exp_dir, "RandomPadding_student")
    rand_init_file = os.path.join(exp_dir, "Completely_Random_Init")
    teacher_file = os.path.join(exp_dir, "teacher_w_residual")
    teacher_nr_file = os.path.join(exp_dir, "teacher_w_out_residual")

    scalars_xvals = [
        (n2n_file, 'iter/train/total_flops'),
        (r2r_file, 'iter/train/total_flops'),
        (netmorph_file, 'iter/train/total_flops'),
        (rand_pad_file, 'iter/train/total_flops'),
        (rand_init_file, 'iter/train/total_flops'),
        (teacher_file, 'iter/train/total_flops'),
        (teacher_nr_file, 'iter/train/total_flops'),
    ]

    scalars_yvals = [
        (n2n_file, 'iter/train/valacc_1'),
        (r2r_file, 'iter/train/valacc_1'),
        (netmorph_file, 'iter/train/valacc_1'),
        (rand_pad_file, 'iter/train/valacc_1'),
        (rand_init_file, 'iter/train/valacc_1'),
        (teacher_file, 'iter/train/valacc_1'),
        (teacher_nr_file, 'iter/train/valacc_1'),
    ]
    labels = ['Net2WiderNet', 'R2WiderR', 'NetMorph', 'RandomPad', 'ResNetCifar18(1)', 'ResNetCifar18(2/3)',
              'ResNetCifar18(2/3)(NoRes)']
    xaxis = "Iterations"
    yaxis = "Validation Accuracy"
    linestyles = ['-', '-', '-', '-', '-', '-', '-']
    parametric_plots(imgfile, scalars_xvals, scalars_yvals, xaxis, yaxis, linestyles, labels, log_x_axis=False)

    # # Fig 5 / SVHN / net2widernet tests, validation curves
    # imgfile = os.path.join(base_dir, "svhn_net2widernet.pdf")

    # scalars = [
    #     ('n2n_file': 'iter/train/valacc_1'),
    #     ('r2r_file': 'iter/train/valacc_1'),
    #     ('netmorph_file': 'iter/train/valacc_1'),
    #     ('rand_pad_file': 'iter/train/valacc_1'),
    #     ('rand_init_file': 'iter/train/valacc_1'),
    #     ('teacher_file'): 'iter/train/valacc_1'),
    # ]
    # labels = ['Net2WiderNet', 'R2WiderR', 'NetMorph', 'RandomPad', 'ResNetCifar18(1)', 'ResnetCifar18(2/3)']
    # xaxis = "Iterations"
    # yaxis = "Validation Accuracy"
    # linestyles = ['-', '-', '-', '-', '-']
    # normal_plots_new(imgfile, scalars, xaxis, yaxis, linestyles, labels)

    # # Fig 6 / CIFAR / net2deepernet tests, validation curves
    # imgfile = os.path.join(base_dir, "cifar_net2deepernet.pdf")

    # scalars = [
    #     ('n2n_file': 'iter/train/valacc_1'),
    #     ('r2r_file': 'iter/train/valacc_1'),
    #     ('rand_pad_file': 'iter/train/valacc_1'),
    #     ('rand_init_file': 'iter/train/valacc_1'),
    #     ('teacher_file'): 'iter/train/valacc_1'),
    # ]
    # labels = ['Net2DeeperNet', 'R2DeeperR', 'RandomPad', 'ResNetCifar18(3/8)', 'ResNetCifar12(3/8)']
    # xaxis = "Iterations"
    # yaxis = "Validation Accuracy"
    # linestyles = ['-', '-', '-', '-']
    # normal_plots_new(imgfile, scalars, xaxis, yaxis, linestyles, labels)

    # # Fig 6 / SVHN / net2deepernet tests, validation curves
    # imgfile = os.path.join(base_dir, "svhn_net2deepernet.pdf")

    # scalars = [
    #     ('n2n_file': 'iter/train/valacc_1'),
    #     ('r2r_file': 'iter/train/valacc_1'),
    #     ('rand_pad_file': 'iter/train/valacc_1'),
    #     ('rand_init_file': 'iter/train/valacc_1'),
    #     ('teacher_file'): 'iter/train/valacc_1'),
    # ]
    # labels = ['Net2DeeperNet', 'R2DeeperR', 'RandomPad', 'ResNetCifar18(1)', 'ResNetCifar12(1)']
    # xaxis = "Iterations"
    # yaxis = "Validation Accuracy"
    # linestyles = ['-', '-', '-', '-']
    # normal_plots_new(imgfile, scalars, xaxis, yaxis, linestyles, labels)

    # # Fig 7 / CIFAR / overfitting
    # imgfile = os.path.join(base_dir, "cifar_overfit.pdf")

    # scalars = [
    #     ('overfitting_teacher_file': 'iter/train/accuracy_1'),
    #     ('overfitting_teacher_file': 'iter/train/valacc_1'),
    #     ('overfitting_student_file': 'iter/train/accuracy_1'),
    #     ('overfitting_student_file': 'iter/train/valacc_1'),
    #     ('rand_init_file': 'iter/train/accuracy_1'),
    #     ('rand_init_file': 'iter/train/valacc_1'),
    # ]
    # labels = ['Teacher-Train', 'Teacher-Test', 'Student-Train', 'Student-Test', 'ResNetCifar18(3/8)-Train', 'ResNetCifar18(3/8)-Test']
    # xaxis = "Iterations"
    # yaxis = "Training/Validation Accuracy"
    # linestyles = ['--', '-', '--', '-', '--', '-']
    # normal_plots_new(imgfile, scalars, xaxis, yaxis, linestyles, labels)

    # # Fig 8 / CIFAR / Weight init
    # imgfile = os.path.join(base_dir, "cifar_large_init.pdf")

    # scalars = [
    #     ('largewieghts_file': 'iter/train/accuracy_1'),
    #     ('matchstd_file': 'iter/train/accuracy_1'),
    # ]
    # labels = ['He Init', 'StdDev Matching']
    # xaxis = "Iterations"
    # yaxis = "Training/Validation Accuracy"
    # linestyles = ['--', '--']
    # normal_plots_new(imgfile, scalars, xaxis, yaxis, linestyles, labels)

    # # Fig 9.1 / CIFAR / R2WiderR tests
    # imgfile = os.path.join(base_dir, "cifar_r2widerr.pdf")

    # scalars = [
    #     ('n2n_file': 'iter/train/valacc_1'),
    #     ('r2r_file': 'iter/train/valacc_1'),
    #     ('netmorph_file': 'iter/train/valacc_1'),
    #     ('rand_pad_file': 'iter/train/valacc_1'),
    #     ('rand_init_file': 'iter/train/valacc_1'),
    #     ('teacher_file'): 'iter/train/valacc_1'),
    # ]
    # labels = ['Net2WiderNet', 'R2WiderR', 'NetMorph', 'RandomPad', 'ResNetCifar18(3/8)', 'ResNetCifar18(1/4)']
    # xaxis = "Iterations"
    # yaxis = "Validation Accuracy"
    # linestyles = ['-', '-', '-', '-', '-']
    # normal_plots_new(imgfile, scalars, xaxis, yaxis, linestyles, labels)

    # # Fig 9.1 / SVHN / R2WiderR tests
    # imgfile = os.path.join(base_dir, "svhn_r2widerr.pdf")

    # scalars = [
    #     ('n2n_file': 'iter/train/valacc_1'),
    #     ('r2r_file': 'iter/train/valacc_1'),
    #     ('netmorph_file': 'iter/train/valacc_1'),
    #     ('rand_pad_file': 'iter/train/valacc_1'),
    #     ('rand_init_file': 'iter/train/valacc_1'),
    #     ('teacher_file'): 'iter/train/valacc_1'),
    # ]
    # labels = ['Net2WiderNet', 'R2WiderR', 'NetMorph', 'RandomPad', 'ResNetCifar18(1)', 'ResNetCifar18(2/3)']
    # xaxis = "Iterations"
    # yaxis = "Validation Accuracy"
    # linestyles = ['-', '-', '-', '-', '-']
    # normal_plots_new(imgfile, scalars, xaxis, yaxis, linestyles, labels)

    # # Fig 9.1 / CIFAR / R2DeeprR tests
    # imgfile = os.path.join(base_dir, "cifar_r2deeperr.pdf")

    # scalars = [
    #     ('n2n_file': 'iter/train/valacc_1'),
    #     ('r2r_file': 'iter/train/valacc_1'),
    #     ('rand_pad_file': 'iter/train/valacc_1'),
    #     ('rand_init_file': 'iter/train/valacc_1'),
    #     ('teacher_file'): 'iter/train/valacc_1'),
    # ]
    # labels = ['Net2DeeperNet', 'R2DeeperR', 'RandomPad', 'ResNetCifar18(3/8)', 'ResNetCifar12(3/8)']
    # xaxis = "Iterations"
    # yaxis = "Validation Accuracy"
    # linestyles = ['-', '-', '-', '-', '-']
    # normal_plots_new(imgfile, scalars, xaxis, yaxis, linestyles, labels)

    # # Fig 9.1 / SVHN / R2DeeprR tests
    # imgfile = os.path.join(base_dir, "svhn_r2deeperr.pdf")

    # scalars = [
    #     ('n2n_file': 'iter/train/valacc_1'),
    #     ('r2r_file': 'iter/train/valacc_1'),
    #     ('rand_pad_file': 'iter/train/valacc_1'),
    #     ('rand_init_file': 'iter/train/valacc_1'),
    #     ('teacher_file'): 'iter/train/valacc_1'),
    # ]
    # labels = ['Net2DeeperNet', 'R2DeeperR', 'RandomPad', 'ResNetCifar18(1)', 'ResNetCifar12(1)']
    # xaxis = "Iterations"
    # yaxis = "Validation Accuracy"
    # linestyles = ['-', '-', '-', '-', '-']
    # normal_plots_new(imgfile, scalars, xaxis, yaxis, linestyles, labels)

    # # Fig 9.2 / CIFAR / R2WiderR tests
    # imgfile = os.path.join(base_dir, "cifar_r2widerr_flops.pdf")

    # scalars_xvals = [
    #     ('n2n_file': 'iter/train/total_flops'),
    #     ('r2r_file': 'iter/train/total_flops'),
    #     ('netmorph_file': 'iter/train/total_flops'),
    #     ('rand_pad_file': 'iter/train/total_flops'),
    #     ('rand_init_file': 'iter/train/total_flops'),
    #     ('teacher_file'): 'iter/train/total_flops'),
    # ]

    # scalars_yvals = [
    #     ('n2n_file': 'iter/train/valacc_1'),
    #     ('r2r_file': 'iter/train/valacc_1'),
    #     ('netmorph_file': 'iter/train/valacc_1'),
    #     ('rand_pad_file': 'iter/train/valacc_1'),
    #     ('rand_init_file': 'iter/train/valacc_1'),
    #     ('teacher_file'): 'iter/train/valacc_1'),
    # ]
    # labels = ['Net2WiderNet', 'R2WiderR', 'NetMorph', 'RandomPad', 'ResNetCifar18(3/8)', 'ResNetCifar18(1/4)']
    # xaxis = "Iterations"
    # yaxis = "Validation Accuracy"
    # linestyles = ['-', '-', '-', '-', '-']
    # parametric_plots(imgfile, scalars_xvals, scalars_yvals, xaxis, yaxis, linestyles, labels, log_x_axis=True)

    # # Fig 9.2 / SVHN / R2WiderR tests
    # imgfile = os.path.join(base_dir, "svhn_r2widerr_flops.pdf")

    # scalars_xvals = [
    #     ('n2n_file': 'iter/train/total_flops'),
    #     ('r2r_file': 'iter/train/total_flops'),
    #     ('netmorph_file': 'iter/train/total_flops'),
    #     ('rand_pad_file': 'iter/train/total_flops'),
    #     ('rand_init_file': 'iter/train/total_flops'),
    #     ('teacher_file'): 'iter/train/total_flops'),
    # ]

    # scalars_yvals = [
    #     ('n2n_file': 'iter/train/valacc_1'),
    #     ('r2r_file': 'iter/train/valacc_1'),
    #     ('netmorph_file': 'iter/train/valacc_1'),
    #     ('rand_pad_file': 'iter/train/valacc_1'),
    #     ('rand_init_file': 'iter/train/valacc_1'),
    #     ('teacher_file'): 'iter/train/valacc_1'),
    # ]
    # labels = ['Net2WiderNet', 'R2WiderR', 'NetMorph', 'RandomPad', 'ResNetCifar18(1)', 'ResNetCifar18(2/3)']
    # xaxis = "Iterations"
    # yaxis = "Validation Accuracy"
    # linestyles = ['-', '-', '-', '-', '-']
    # parametric_plots(imgfile, scalars_xvals, scalars_yvals, xaxis, yaxis, linestyles, labels, log_x_axis=True)

    # # Fig 9.2 / CIFAR / R2DeeprR tests
    # imgfile = os.path.join(base_dir, "cifar_r2deeperr_flops.pdf")

    # scalars_xvals = [
    #     ('n2n_file': 'iter/train/total_flops'),
    #     ('r2r_file': 'iter/train/total_flops'),
    #     ('rand_pad_file': 'iter/train/total_flops'),
    #     ('rand_init_file': 'iter/train/total_flops'),
    #     ('teacher_file'): 'iter/train/total_flops'),
    # ]

    # scalars_yvals = [
    #     ('n2n_file': 'iter/train/valacc_1'),
    #     ('r2r_file': 'iter/train/valacc_1'),
    #     ('rand_pad_file': 'iter/train/valacc_1'),
    #     ('rand_init_file': 'iter/train/valacc_1'),
    #     ('teacher_file'): 'iter/train/valacc_1'),
    # ]
    # labels = ['Net2DeeperNet', 'R2DeeperR', 'RandomPad', 'ResNetCifar18(3/8)', 'ResNetCifar12(3/8)']
    # xaxis = "Iterations"
    # yaxis = "Validation Accuracy"
    # linestyles = ['-', '-', '-', '-', '-']
    # parametric_plots(imgfile, scalars_xvals, scalars_yvals, xaxis, yaxis, linestyles, labels, log_x_axis=True)

    # # Fig 9.2 / SVHN / R2DeeprR tests
    # imgfile = os.path.join(base_dir, "svhn_r2deeperr_flops.pdf")

    # scalars_xvals = [
    #     ('n2n_file': 'iter/train/total_flops'),
    #     ('r2r_file': 'iter/train/total_flops'),
    #     ('rand_pad_file': 'iter/train/total_flops'),
    #     ('rand_init_file': 'iter/train/total_flops'),
    #     ('teacher_file': 'iter/train/total_flops'),
    # ]

    # scalars_yvals = [
    #     ('n2n_file': 'iter/train/valacc_1'),
    #     ('r2r_file': 'iter/train/valacc_1'),
    #     ('rand_pad_file': 'iter/train/valacc_1'),
    #     ('rand_init_file': 'iter/train/valacc_1'),
    #     ('teacher_file'): 'iter/train/valacc_1'),
    # ]
    # labels = ['Net2DeeperNet', 'R2DeeperR', 'RandomPad', 'ResNetCifar18(1)', 'ResNetCifar12(1)']
    # xaxis = "Iterations"
    # yaxis = "Validation Accuracy"
    # linestyles = ['-', '-', '-', '-', '-']
    # parametric_plots(imgfile, scalars_xvals, scalars_yvals, xaxis, yaxis, linestyles, labels, log_x_axis=True)

    """
    # Net2WiderNet results
    imgfile = os.path.join(base_dir, "n2wn.png")
    nw_n2n  = os.path.join(base_dir, "nw_n2n.csv")
    nw_r2r  = os.path.join(base_dir, "nw_r2r.csv")
    nw_rp   = os.path.join(base_dir, "nw_rp.csv")
    nw_init = os.path.join(base_dir, "nw_init.csv")
    nw_r2rt = os.path.join(base_dir, "nw_r2rt.csv")
    nw_n2nt = os.path.join(base_dir, "nw_n2nt.csv")
    nw_nm   = os.path.join(base_dir, "nw_nm.csv")
    yfiles = [nw_n2n,
              nw_r2r,
              nw_rp,
              nw_init,
              nw_r2rt,
              nw_n2nt,
              nw_nm]
    xaxis = "Epochs"
    yaxis = "Validation Accuracy"
    linestyles = ['-','-','-','-','-','-','-']
    labels = ['Net2WiderNet', 'R2WiderR', 'RandomPad', 'ResNetCifar18(1/6)', 'Teacher', 'Teacher-NoResidual', 'NetMorph']
    normal_plots(imgfile, yfiles, xaxis, yaxis, linestyles, labels, 100)

    # Net2DeeperNet results
    imgfile = os.path.join(base_dir, "n2dn.png")
    nd_n2n  = os.path.join(base_dir, "nd_n2n.csv")
    nd_r2r  = os.path.join(base_dir, "nd_r2r.csv")
    nd_rp   = os.path.join(base_dir, "nd_rp.csv")
    nd_init = os.path.join(base_dir, "nd_init.csv")
    nd_r2rt = os.path.join(base_dir, "nd_r2rt.csv")
    nd_n2nt = os.path.join(base_dir, "nd_n2nt.csv")
    yfiles = [nd_n2n,
              nd_r2r,
              nd_rp,
              nd_init,
              nd_r2rt,
              nd_n2nt]
    xaxis = "Epochs"
    yaxis = "Validation Accuracy"
    linestyles = ['-','-','-','-','-','-','-']
    labels = ['Net2DeeperNet', 'R2DeeperR', 'RandomPad', 'ResNetCifar18(1/6)', 'Teacher', 'Teacher-NoResidual']
    normal_plots(imgfile, yfiles, xaxis, yaxis, linestyles, labels, 100)

    # R2WiderR results
    imgfile = os.path.join(base_dir, "r2wr.png")
    imgfile_f = os.path.join(base_dir, "r2wr_f.png")
    rw_n2n  = os.path.join(base_dir, "rw_n2n.csv")
    rw_r2r  = os.path.join(base_dir, "rw_r2r.csv")
    rw_rp   = os.path.join(base_dir, "rw_rp.csv")
    rw_init = os.path.join(base_dir, "rw_init.csv")
    rw_r2rt = os.path.join(base_dir, "rw_r2rt.csv")
    rw_n2nt = os.path.join(base_dir, "rw_n2nt.csv")
    rw_nm   = os.path.join(base_dir, "rw_nm.csv")
    yfiles = [nw_n2n,
              nw_r2r,
              nw_rp,
              nw_init,
              nw_r2rt,
              nw_n2nt,
              nw_nm]
    rw_n2n_f  = os.path.join(base_dir, "rw_n2n_f.csv")
    rw_r2r_f  = os.path.join(base_dir, "rw_r2r_f.csv")
    rw_rp_f   = os.path.join(base_dir, "rw_rp_f.csv")
    rw_init_f = os.path.join(base_dir, "rw_init_f.csv")
    rw_r2rt_f = os.path.join(base_dir, "rw_r2rt_f.csv")
    rw_n2nt_f = os.path.join(base_dir, "rw_n2nt_f.csv")
    rw_nm_f   = os.path.join(base_dir, "rw_nm_f.csv")
    xfiles = [rw_n2n_f,
              rw_r2r_f,
              rw_rp_f,
              rw_init_f,
              rw_r2rt_f,
              rw_n2nt_f,
              rw_nm_f]
    xaxis = "Epochs"
    yaxis = "Validation Accuracy"
    linestyles = ['-','-','-','-','-','-','-']
    labels = ['Net2WiderNet', 'R2WiderR', 'RandomPad', 'ResNetCifar16(1/6)', 'Teacher', 'Teacher-NoResidual', 'NetMorph']
    normal_plots(imgfile, yfiles, xaxis, yaxis, linestyles, labels, 100)
    xaxis = "FLOPs"
    parametric_plots(imgfile_f, xfiles, yfiles, xaxis, yaxis, linestyles, labels, 100)


    # R2DeeperR results
    imgfile = os.path.join(base_dir, "r2dr.png")
    imgfile_f = os.path.join(base_dir, "r2dr_f.png")
    rd_n2n  = os.path.join(base_dir, "rd_n2n.csv")
    rd_r2r  = os.path.join(base_dir, "rd_r2r.csv")
    rd_rp   = os.path.join(base_dir, "rd_rp.csv")
    rd_init = os.path.join(base_dir, "rd_init.csv")
    rd_r2rt = os.path.join(base_dir, "rd_r2rt.csv")
    rd_n2nt = os.path.join(base_dir, "rd_n2nt.csv")
    rd_nm   = os.path.join(base_dir, "rd_nm.csv")
    yfiles = [nw_n2n,
              nw_r2r,
              nw_rp,
              nw_init,
              nw_r2rt,
              nw_n2nt,
              nw_nm]
    rd_n2n_f  = os.path.join(base_dir, "rd_n2n_f.csv")
    rd_r2r_f  = os.path.join(base_dir, "rd_r2r_f.csv")
    rd_rp_f   = os.path.join(base_dir, "rd_rp_f.csv")
    rd_init_f = os.path.join(base_dir, "rd_init_f.csv")
    rd_r2rt_f = os.path.join(base_dir, "rd_r2rt_f.csv")
    rd_n2nt_f = os.path.join(base_dir, "rd_n2nt_f.csv")
    rd_nm_f   = os.path.join(base_dir, "rd_nm_f.csv")
    xfiles = [rw_n2n_f,
              rw_r2r_f,
              rw_rp_f,
              rw_init_f,
              rw_r2rt_f,
              rw_n2nt_f,
              rw_nm_f]
    xaxis = "Epochs"
    yaxis = "Validation Accuracy"
    linestyles = ['-','-','-','-','-','-','-']
    labels = ['Net2WiderNet', 'R2WiderR', 'RandomPad', 'ResNetCifar(1/6)', 'Teacher', 'Teacher-NoResidual', 'NetMorph']
    normal_plots(imgfile, yfiles, xaxis, yaxis, linestyles, labels, 100)
    xaxis = "FLOPs"
    parametric_plots(imgfile_f, xfiles, yfiles, xaxis, yaxis, linestyles, labels, 100)
    """
