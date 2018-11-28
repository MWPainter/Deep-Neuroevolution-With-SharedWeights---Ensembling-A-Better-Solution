




class Residual_Connection(object):
    """
    TODO
        - especially differentiate between input slice, residual slice and output_slice

    If "out = residual_connection(x)", then for each self.slice_map[(a,b)] = (k, (c,d)) we have:
    out[c:d] = k * x[a:b]
    """
    def __init__(self, initial_residual_slice_indices):
        """
        TODO

        :param initial_residual_channels:
        """
        self.residual_slice_map = {}
        for i in range(1, len(initial_residual_slice_indices)):
            residual_slice = (initial_residual_slice_indices[i-1], initial_residual_slice_indices[i])
            self.residual_slice_map[residual_slice] = (1.0, residual_slice)
        self.next_output_index = initial_residual_slice_indices[-1]





    def widen_(self, volume_slice_indices, extra_channels, multiplicative_widen):
        """
        TODO: identify in r2widerr when we have the problem where the volume recieving the residual connection isn't widended enough to fit the residual connection

        TODO
        - especially differentiate between input slice, residual slice and output_slice

        :param volume_slice_indices:
        :param extra_channels:
        :param multiplicative_widen:
        :return:
        """
        # print(self.residual_slice_map)
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

            # For each residual slice in the current volume slice, copy the mapping to the new residual slice
            for beg, end in self.residual_slice_map:
                if vs_beg <= beg and end <= vs_end:
                    new_beg, new_end = beg + current_residual_channel_offset, end + current_residual_channel_offset
                    new_residual_slice_map[(new_beg, new_end)] = self.residual_slice_map[(beg,end)]

            # Work out how many new channels will be added for this input slice
            input_slice_extra_channels = extra_channels
            if multiplicative_widen:
                input_slice_extra_channels = (vs_end - vs_beg) * (extra_channels - 1)
            half_input_slice_extra_channels = input_slice_extra_channels // 2

            # Compute new residual slice mappings for the additional volume being added by a widening transform
            new_res_beg = vs_end + current_residual_channel_offset
            new_res_end = vs_end + current_residual_channel_offset + half_input_slice_extra_channels
            new_out_beg = self.next_output_index
            new_out_end = self.next_output_index + half_input_slice_extra_channels
            new_residual_slice_map[(new_res_beg, new_res_end)] = (1.0, (new_out_beg, new_out_end))

            new_res_beg += half_input_slice_extra_channels
            new_res_end += half_input_slice_extra_channels
            new_residual_slice_map[(new_res_beg, new_res_end)] = (-1.0, (new_out_beg, new_out_end))

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
        The forward for residual connection. This should apply the slice mapping to a volume

        TODO: properly describe + params description
        res = residual connection
        x = input to add to
        """
        for ((res_beg, res_end), (k, (out_beg, out_end))) in self.residual_slice_map.items():
            x[:, out_beg:out_end] += k * res[:, res_beg:res_end]
        return x

