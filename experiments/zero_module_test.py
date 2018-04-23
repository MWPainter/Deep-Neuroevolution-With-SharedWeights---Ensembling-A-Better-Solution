import tensorflow as tf 
import numpy as np 

# TODO: check that the initializations are as we expect (print out the actual values of the tensors)
# TODO: print out all of the variable names and check that all the things that we want to be trainable are, and that they're named as we want
# TODO: add optional max pooling (/min pooling if need to use that as a trick)
# TODO: finish the capacity tests


def compute_filter_shape(inputs, num_output_filters):
	"""
	Computes the filter shape for an input batch. Assume a 4x4 kernal for now
	"""
	batch_size = tf.shape(inputs)[0]
	num_input_filters = inputs.get_shape().as_list()[3]
	filter_shape = (4, 4, num_input_filters, num_output_filters)
	return filter_shape



def conv_he_initialization(filter_shape):
	"""
	Assume kernal shape is (width,height,in_channels,out_channels)
	Then there are in_channels number of inputs, for every one of the width * height * out_channels outputs
	Therefore, if we initialize with normal distributions with a stddev of sqrt(2.0/in_channels)

	:param filter_shape: tuple for the filter shape to be used in the conv net
	:return: random normals, scaled for he_initialization, in the shape of the filter
	"""
	in_channels = filter_shape[2]
	return np.random.randn(*filter_shape).astype(np.float32) * np.sqrt(2.0/in_channels)



def conv_zero_module(inputs, balance_inputs, num_output_filters, add_max_pool, stride, noise_stddev, scope):
	"""
	Initializes a convolutional layer such that if there are 2N filters output, then the output of filter 0 < i <= N 
	is equal to output of filter N+i.

	If we need to balance inputs, then it means our input is the output of some other zero_module. We need to make them 
	cancel out in our input. So we need to negate the inputs of the filters 0:N and N:2N in out initialization.

	For now, we use 4x4 convolutions, with a stride of 'strig', currently 2x2 max pooling, and only relu non-linearity

	Outline of the computation that we perform:	
	1. Input shape of [A,A,2D], and desired output shape is [A,A,2K]
	2.1. If balance_input, make a filter [4,4,D,K], and for indices [:,:,N+i,K+j] use the SAME value of [:,:,i,j]
	2.2. Otherwise make a filter [4,4,2D,K]
	3. Now, we have a filter of shape [3,3,2D,K]
	4. We actually want a filter of shape [3,3,2D,2K], and we want to initialize the weights negatively for the second half
	5. Set the weights for [:,:,:,K+j] equal to  [:,:,:,j]
	6. Finally add a batch norm (so that it's symmetric initially)
	7. And the non-linearity
	8. Output should be symmetric in the filters
	
	:param inputs: input to the zero module
	:param balance_inputs: if the input is the output from another zero module, and needs 'balancing'
	:param num_output_filters: the number of output filters to have in the output
	:param add_max_pool: should we add a max pool before anything else?
	:param stride: the stride to use in the convolution part of the module
	:param noise_stddev: the stddev of the noise 
	:param scope: tensorflow variable scoping
	:return: output from the zero module
	"""

	with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
		# If max pooling, we apply that first
		if add_max_pool:
			inputs = tf.layers.max_pooling2d(inputs, 2, 2)

		# Compute [A,A,2D,2K]
		filter_shape = compute_filter_shape(inputs, num_output_filters)

		# Make a buffer of zeros for the kernal init, and we will slowly fill it out appropriately
		filter_init = np.zeros(filter_shape, dtype=np.float32)

		# we're going initialize the first [A,A,2D,K], or the first [A,A,D,K] as describe above
		A = filter_shape[0]
		D = filter_shape[2] / 2
		K = filter_shape[3] / 2
		if balance_inputs:
			filter_init[:,:, :D, :K] = conv_he_initialization((A,A,D,K))
			filter_init[:,:, D:, :K] = -filter_init[:,:, :D, :K]
		else:
			filter_init[:,:,:, :K] = conv_he_initialization((A, A, 2*D, K))

		# Now to initialize the negated, second half of the filter
		filter_init[:,:,:, K:] = filter_init[:,:,:, :K]

		# Add noise to filter
		filter_init += np.random.randn(*filter_shape).astype(np.float32) * noise_stddev

		# Actually make the filter tf variable, and make the convolutional layer
		filter = tf.Variable(filter_init, name=scope+"/conv_filter")
		conv_outputs = tf.nn.conv2d(input = inputs,
									filter = filter,
									strides = [1,stride,stride,1],
									padding = "SAME")

		# Apply batch norm, it should be zero centered by default?
		# batch_norm_outputs = tf.contrib.layers.batch_norm(conv_outputs)
		batch_norm_outputs = conv_outputs

		# finally, apply relu
		return tf.nn.relu(batch_norm_outputs)



def fc_zero_module(input):	
	"""
	A fully connect layer, who's inputs need to be balanced, the output doesn't really matter, we're not currently 
	considering that addition/widenning of fc layers.

	It's just important that the first fc layer balances the inputs correctly.
	
	:param input: input to the zero module
	:return: output of the fc layer
	"""
	pass



def single_conv_zero_module_test():
	"""
	Single convolutional layer, manually check that the outputs are symmetric
	"""

	# Build a one layer network
	inputs_placeholder = tf.placeholder(shape=[None,4,4,2], dtype=tf.float32, name="input_placeholder")
	output_op = conv_zero_module(inputs_placeholder, False, 4, False, 2, 0.0, "myscope")

	# Randomly generate inputs, and check (manually) that the outputs are symmetric
	sess = tf.InteractiveSession()
	sess.run(tf.global_variables_initializer())

	# Manually test the symmetry
	for i in xrange(3):
		rand_inputs = np.random.uniform(low=-1.0, high=1.0, size=(1,4,4,2))
		feed_dict = {inputs_placeholder: rand_inputs}
		out = sess.run([output_op], feed_dict=feed_dict)

		print("")
		print("Random input:")
		print(rand_inputs)
		print("")
		print("Output:")
		print(out)
		print("")
		print("Subtracted output:")
		sub_out = out[0][:,:,:,:2] - out[0][:,:,:,2:]
		print(sub_out)

	for i in xrange(5):
		print("")


def single_second_layer_conv_zero_module_test(should_max_pool, noise_stddev):
	# Repeate the same, but replicate the symmetry in the input (which we've now confirmed from the above) and then 
	# confirm that the second layer will be zero preserving
	# Option to check if the max pooling is correct also, and doesn't effect the "zeroness"

	# Build (second layer) in a network
	inputs_placeholder = tf.placeholder(shape=[None,4,4,8], dtype=tf.float32, name="input_placeholder")
	output_op = conv_zero_module(inputs_placeholder, True, 4, should_max_pool, 1, noise_stddev, "myscopetwo")

	# Randomly generate inputs, and check (manually) that the outputs are symmetric
	sess = tf.InteractiveSession()
	sess.run(tf.global_variables_initializer())

	# Manually test the symmetry
	for i in xrange(5):
		rand_inputs = np.zeros((1,4,4,8))
		rand_inputs[:,:,:,:4] = np.random.uniform(low=-1.0, high=1.0, size=(1,4,4,4))
		rand_inputs[:,:,:,4:] = rand_inputs[:,:,:,:4]
		feed_dict = {inputs_placeholder: rand_inputs}
		out = sess.run([output_op], feed_dict=feed_dict)

		print("")
		print("Random inputs (symmetric):")
		print(rand_inputs)
		print("")
		print("Output (should be zero, because of the symetrically added input, unless noise added):")
		print(out)

	for i in xrange(5):
		print("")



def single_conv_single_fc_zero_module_test():
	"""
	2 layer network, initialized such that f(x) \approx 0 for all x
	"""
	pass


def multi_conv_zero_module_test():
	"""
	Multi layer network, initialized such that f(x) \approx 0 for all x
	"""
	pass

def capacity_test():
	"""
	Now we can we actually learn mnist at least with this architecture initialized to f(x) \approx 0 for all x?
	"""

def noise_breaking_test():
	"""
	Perform some testing on the amount of noise that's useful to break symmetry. (For now, we will just assume that the 
	noise should be proportional to initialization noise strength).
	"""





if __name__ == "__main__":
	single_conv_zero_module_test()
	single_second_layer_conv_zero_module_test(False, 0.0)
	single_second_layer_conv_zero_module_test(True, 0.0)
	single_second_layer_conv_zero_module_test(False, 0.1)
	single_second_layer_conv_zero_module_test(True, 0.001 * np.sqrt(2.0/8))
	# single_second_layer_conv_zero_module_test(True, 100.1)
	# single_conv_single_fc_zero_module_test()
	# multi_conv_zero_module_test()
	# capacity_test()
	# noise_breaking_test()