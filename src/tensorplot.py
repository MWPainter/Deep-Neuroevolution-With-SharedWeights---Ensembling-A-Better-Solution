



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
    if not hasattr(val, '__len__'):
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
        return arr[i,j]
    return arr[i]





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




def _read_sequence(logdir, scalar_name):
    """
    Reads from the TensorBoard summery to produce a list of

    :param logdir: The filepath of the TB log directory
    :param scalar_name: The scalar that we wish to get the values for
    :return: np.array with shape (N,2) for the sequence in
    """
    pass




def _plot_sequence(sequence, color):
    """
    Adds the plot 'sequence' with color 'color' to the current Seaboarn figure.

    :param sequence: A np.array with shape (N,2) for the sequence we wish to add to the current figure.
    :param color: The color that we wish to plot with.
    """
    pass




def _get_color_palet(palet_name):
    """
    :return: The seaboarn color pallet with name 'palet_name'
    """
    pass




def _save_current_fig(out_filename):
    """
    Saves the current seaboarn figure in the file with name 'out_filename'.
    """




"""
Section: TensorPlot processing.
"""




def _compute_parametric_curve(seq1, seq2, missing_value_completion):
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

    while i < M and j < N:
        if _si(seq1,i,0) == _si(seq2,j,0):
            new_point = (_si(seq1,i,1), _si(seq2,j,1))
            parametric_points.append(new_point)
            i += 1
            j += 1

        elif _si(seq1, i, 0) < _si(seq2, j, 0):
            new_point = None
            if missing_value_completion == "use_last":
                new_point = (_si(seq1,i,1), _si(seq2,j-1,1))
            elif missing_value_completion != "linear_interpolate":
                new_point = (_si(seq1,i,1), _interpolate_value(_si(seq2,j-1), _si(seq2,j), _si(seq1,i,0)))
            if new_point is not None:
                parametric_points.append(new_point)
            i += 1

        else: # seq1[i,0] > seq2[j,0]:
            new_point = None
            if missing_value_completion == "use_last":
                new_point = (_si(seq1,i-1,1), _si(seq2,j,1))
            elif missing_value_completion != "linear_interpolate":
                new_point = (_interpolate_value(_si(seq1,i-1), _si(seq1,i), _si(seq2,j,0)), _si(seq2,j,1))
            if new_point is not None:
                parametric_points.append(new_point)
            j += 1

    return np.array(parametric_points)





def _interpolate_value(p1, p2, u):
    """
    If p1 = (u1,v1) and p2 = (u2,v2) then return v = v1 + (u-u1)/(u2-u1) * (v2-v1)
    """
    u1, v1 = p1
    u2, v2 = p2
    return v1 + (u-u1)/(u2-u1) * (v2-v1)







"""
Section: TensorPlot interface
"""




def normal_plots(logdir, out_filename, sclar_names):
    """
    Reads in the sequence values recorded for each scalar in 'scalar_names' and plots it on the same graph.

    :param logdir: The filepath of the TB log directory
    :param out_filename: The name of the file to save an image of the figure at
    :param sclar_names: The scalars to plot
    """
    sclar_names = _listify(sclar_names)
    for scalar_name in sclar_names:
        scalar_sequence = _read_sequence(logdir, scalar_name)
        _plot_sequence(scalar_sequence, ...)
    _save_current_fig(out_filename)




def parametric_plots(logdir, out_filename, scalar_names_one, scalar_names_two, missing_value_completion="use_last"):
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

    :param logdir: The filepath of the TB log directory
    :param out_filename: The name of the file to save an image of the figure at
    :param scalar_names_one: The first scalar from TB to use in the parametric plot
    :param scalar_names_two: The second scalar from TB to use in the parametric plot
    :param missing_value_completion: How we should complete any missing values for the latent parameters
    """
    scalar_names_one = _listify(scalar_names_one)
    scalar_names_two = _listify(scalar_names_two)
    assert(len(scalar_names_one) == len(scalar_names_two), "Invalid input to plot multiple parametric curves on a single axis.")
    for i in range(len(scalar_names_one)):
        scalar_name_one = scalar_names_one[i]
        scalar_name_two = scalar_names_two[i]
        scalar_sequence_one = _read_sequence(logdir, scalar_name_one)
        scalar_sequence_two = _read_sequence(logdir, scalar_name_two)
        parametric_sequence = _compute_parametric_curve(scalar_sequence_one, scalar_sequence_two, missing_value_completion)
        _plot_sequence(parametric_sequence, ...)
    _save_current_fig(out_filename)