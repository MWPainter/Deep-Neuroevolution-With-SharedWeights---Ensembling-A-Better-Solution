import math





def _round_up_multiply(a, b, m):
    """
    Performs a*b and rounds up to the nearest m. Note that (x+m-1)//m, is a divide by m rounded up
    """
    prod = int(math.ceil(((a*b + m - 1) // m) * m))
    return prod





class Residual_Connection(object):
    """
    This file defines a class that wraps a residual connection. As a residual connection will propogate (by addition) a
    hidden volume to further up the network, we need to make residual connections aware of the widening's that are
    taking place.

    Examples of where residual connections will break function preserving transforms, unless handled carefully, are:
    1. Widening volume v (from a single conv module) to [v; w; -w], where [v] is later used in a residual connection
        upstream in the network.
    2. If the volume forming the residual connection is a concatenation of two volumes, say [v1; v2], and it is then
        widened to [v1; w1; -w1; v2; w2; -w2].

    In 1, we observe that if we just copy all of [v; w; -w] in a residual connection that it will break function
    preserving transforms. This is because to widen volume v in a function preserving way, we alter the conv module that
    outputs v, and any conv modules that take v as input. However, consider a volume x, who's input includes a residual
    connection from v, i.e. x += v, which then becomes x += [v; w; -w] immediately after the widening. The modules forming
    the input and output of x will not have been changed, and therefore, the w and -w will change the output.

    One potential (unsatisfactory and incorrect) solution would be to limit the number of channels in the residual
    connection. For this solution, we should consider that happens in example 2. Let c_v1 be the number of channels in
    the voluem v1 and so on. And suppose that we limit the number of channels to k: "x[:,:k] += residual_volume[:,:k]".
    Either we have k <= c_v1, which means that the capacity of the residual channel is extremely limited, or, if we try to
    have k > c_v1, then the residual connection breaks the residual transform, as described in the paragraph above.

    Therefore, working with example 2, we consider conceptually replacing this psuedocode:
    res = [a; b; c; d; e; f] # A widened volume, from example 2
    ...
    x += res


    for the following:
    res = [a; b; c; d; e; f] # A widened volume, from example 2
    ...
    idx = [c_v1, c_v1+c_v2, c_v1+c_v2+c_w1, c_v1+c_v2+c_w1+c_w2]
    x[:idx[0]] += a
    x[idx[0]:idx[1]] += d
    x[idx[1]:idx[2]] += b
    x[idx[1]:idx[2]] -= c
    x[idx[2]:idx[3]] += e
    x[idx[2]:idx[3]] -= f


    The generalize logic for the above solution is precisely the logic that this class contains, so that actually, we can
    write in code:
    res = [a; b; c; d; e; f] # A widened volume, from example 2
    ...
    assert isinstance(res_con, Residual_Connection)
    x = res_con(x, res).


    To be able to do this, the class will keep track of a "slice_map" which we define in the following way:
    if slice_map[(a,b)] = (k, (c,d)), and out = res_con(x, res), then we must have out[c:d] = x[c:d] + k * res[a:b].
    """

    def __init__(self, initial_residual_slice_indices=None):
        """
        Creates a residual conncection object, where we can optionally initialize a slice map (described in the class
        description).

        If no initial slicing is provided, then, we attempt to lazily initialize the slice mapping.

        :param initial_residual_slice_indices: The indices that define the edges of the initial slices of the volume being
            used over the residual connection. This should be a list of the form [0, num_channels] if there is a single
            conv module that outputs the volume used in the residual connection.
        """
        self.residual_slice_map = None
        if initial_residual_slice_indices is not None:
            self._initialize_slice_map(initial_residual_slice_indices)






    def _initialize_slice_map(self, initial_residual_slice_indices):
        """
        Initializes the slice map, where the following holds:
        if slice_map[(a,b)] = (k, (c,d)), and out = res_con(x, res), then we must have out[c:d] = x[c:d] + k * res[a:b].

        We need to give the initial slicings into the residual connection, that is, if the volume which is being used for
        the residual connection is the concatenation of 3 volumes, with 10, 20 and 15 channels respectively, then we need
        to initialize with indices [0, 10, 30, 45].

        :param initial_residual_slice_indices: The indices that define the edges of the initial slices of the volume being
            used over the residual connection. This should be a list of the form [0, num_channels] if there is a single
            conv module that outputs the volume used in the residual connection.
        """
        self.residual_slice_map = {}
        for i in range(1, len(initial_residual_slice_indices)):
            residual_slice = (initial_residual_slice_indices[i-1], initial_residual_slice_indices[i])
            self.residual_slice_map[residual_slice] = (1.0, residual_slice)
        self.next_output_index = initial_residual_slice_indices[-1]





    def _widen_(self, volume_slice_indices, extra_channels, multiplicative_widen, alpha=-1.0, mfactor=2):
        """
        A function that will widen the residual connection, and is strongly tied to R2WiderR's implementation. It made
        more sense to keep the implementation of this as part of the class however.

        This function will alter self.residual_slice_map, to account for the widening of the volume that is propogated
        over the residual connection.

        The inputs give the number of channels from different modules who's outputs are concatenated together.

        We use the following terminology in this function:
        - input slice = (indices for) a slice of the volume that the residual connection is being added to (in y=x+F(x) this would be 'F(x)')
        - residual slice = (indices for) a slice of the volume forming the residual connection (in y=x+F(x), this would be 'x')
        - output slice = (indices for) a slice of the volume forming the output of this residual connection object.

        :param volume_slice_indices: TODO
        :param extra_channels: TODO
        :param multiplicative_widen: TODO
        :param alpha: A constant to multiply repeated output by in residual connections. For function preserving transforms,
                use -1.0. To use vanilla/unmodified residual connections, use 1.0.
        :param mfactor: When adding say 1.4 times the channels, we round up the number of new channels to be a multiple of
                'mfactor'. This parameter has no effect if multiplicative_widen == False.
        :return: TODO
        """
        # if we currently have no slice map, then, initialize it according to the first (assumed correct) volume slice indices
        if self.residual_slice_map is None:
            self._initialize_slice_map(volume_slice_indices)

        # Check for erroneous indices
        for slice_indx in volume_slice_indices:
            for (beg, end) in self.residual_slice_map:
                if beg < slice_indx and slice_indx < end:
                    raise Exception("Volume slice indices must be aligned with slice map indices in a residual connection.")

        # Handle each input slice independently
        current_residual_channel_offset = 0
        new_residual_slice_map = {}
        for i in range(1, len(volume_slice_indices)):
            vs_beg, vs_end = volume_slice_indices[i-1], volume_slice_indices[i]

            # For each residual slice in the current volume slice, copy the mapping to the new residual slice indices
            # We need to shift the indexing in 'res' in '__call__', as 'res' is currently being made larger
            for beg, end in self.residual_slice_map:
                if vs_beg <= beg and end <= vs_end:
                    new_beg, new_end = beg + current_residual_channel_offset, end + current_residual_channel_offset
                    new_residual_slice_map[(new_beg, new_end)] = self.residual_slice_map[(beg,end)]

            # Work out how many new channels will be added for this input slice
            input_slice_extra_channels = extra_channels
            if multiplicative_widen:
                input_slice_extra_channels = _round_up_multiply((extra_channels-1), vs_end-vs_beg, mfactor)
            half_input_slice_extra_channels = input_slice_extra_channels // 2

            # Compute new residual slice mappings for the additional volume being added by a widening transform
            new_res_beg = vs_end + current_residual_channel_offset
            new_res_end = vs_end + current_residual_channel_offset + half_input_slice_extra_channels
            new_out_beg = self.next_output_index
            new_out_end = self.next_output_index + half_input_slice_extra_channels
            new_residual_slice_map[(new_res_beg, new_res_end)] = (1.0, (new_out_beg, new_out_end))

            new_res_beg += half_input_slice_extra_channels
            new_res_end += half_input_slice_extra_channels
            new_residual_slice_map[(new_res_beg, new_res_end)] = (alpha, (new_out_beg, new_out_end))

            # Update parameters for the next input
            self.next_output_index += half_input_slice_extra_channels
            current_residual_channel_offset += input_slice_extra_channels

        # Finally, actually allocate out the new slice map
        self.residual_slice_map = new_residual_slice_map

        # A final sanatiy check that we did something sensible, is to check the slicings at the end
        residual_slice_list = []
        output_slice_list = []
        for res_slice in self.residual_slice_map:
            residual_slice_list.append(res_slice)
            output_slice_list.append(self.residual_slice_map[res_slice][1])
        residual_slice_list.sort()
        output_slice_list.sort()

        cur = 0
        for (beg, end) in residual_slice_list:
            if cur != beg:
                raise Exception("Slices at the end of the widen not as expected in residual connection.")
            cur = end

        last_beg, last_end = output_slice_list[0]
        for (beg, end) in output_slice_list:
            if last_end != beg and (last_beg != beg or last_end != end):
                raise Exception("Slices at the end of the widen not as expected in residual connection.")
            last_beg, last_end = beg, end





    def __call__(self, x, res):
        """
        The forward for residual connection. This should apply the slice mapping to a volume. Which, we recall, is
        defined with the intention that:
        if slice_map[(a,b)] = (k, (c,d)), and out = res_con(x, res), then we must have out[c:d] = x[c:d] + k * res[a:b].

        :param x: The volume that we should be adding to. (Note that this is technically the "residual volume" if we
            strictly follow the definitions in the ResNet paper).
        :param res: The volume for the residual connection (the identity part of the residual connection in the ResNet
            paper).
        :returns: The result of applying the residual connection (propogating 'res') to the volume x.
        """
        # If we haven't created a residual slice map yet, perform the bog standard residual connection
        if self.residual_slice_map is None:
            num_channels = res.size(1)
            x[:, :num_channels] += res
            return x

        # Check that the residual volume (res) being added hasn't been widened so much that it can't fit in x
        # Assumes x.size() = (batch_size, channels, height, width)
        # (Easier to do than you think)
        channel_capacity = x.size(1)
        for _, end in self.residual_slice_map:
            if end > channel_capacity:
                print(x.size())
                print(self.residual_slice_map)
                raise Exception("Residual connection being applied to a volume without a large enough channel " +
                                "capacity for the residual connection.")

        # Compute the result of the residual connection and return
        for ((res_beg, res_end), (k, (out_beg, out_end))) in self.residual_slice_map.items():
            x[:, out_beg:out_end] += k * res[:, res_beg:res_end]
        return x

