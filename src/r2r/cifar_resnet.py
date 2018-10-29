"""
TODO: copy mnist resnet's implementation, but just change the number of filters a little etc...
""" 




# class Cifar_Resnet_v1(nn.Module):
#     def __init__(self, identity_initialize=True, thin=False, deeper=True, noise_ratio=0.0, fn_preserving_transform=True,
#                 widen_init_type="stddev_match"):
#         # Superclass initializer
#         super(Cifar_Resnet_v1, self).__init__()
        
#         # Channel sizes to use across layers (widen will increase this if necessary)
#         self.channel_sizes = [3, 16, 32, 64]
#         self.linear_hidden_units = 128
#         self.noise_ratio = noise_ratio
#         self.identity_initialize = identity_initialize
#         self.fn_preserving_transform = fn_preserving_transform
#         self.widen_init_type=widen_init_type
        
#         # set deeper and thin, these will be update by the self.widen and self.deepen calls if necessary
#         self.deeper = False
#         self.thin = True
        
#         # Make the three conv layers, with three max pools        
#         self.resblock1 = R2R_residual_block_v2(input_channels=self.channel_sizes[0], 
#                                                output_channels=self.channel_sizes[1], 
#                                                identity_initialize=identity_initialize,
#                                                noise_ratio=noise_ratio) # [-1, 32, 32, 32]
#         self.pool1 = nn.MaxPool2d(kernel_size=2)                                        # [-1, 32, 16, 16]  
#         self.resblock2 = R2R_residual_block_v2(input_channels=self.channel_sizes[1], 
#                                                output_channels=self.channel_sizes[2], 
#                                                identity_initialize=identity_initialize,
#                                                noise_ratio=noise_ratio) # [-1, 32, 16, 16]
#         self.pool2 = nn.MaxPool2d(kernel_size=2)                                        # [-1, 64, 8, 8]  
# #         self.resblock3 = R2R_residual_block_v2(input_channels=self.channel_sizes[2], 
# #                                                output_channels=self.channel_sizes[3], 
# #                                                identity_initialize=identity_initialize,
# #                                                noise_ratio=noise_ratio) # [-1, 64, 8, 8]
#         self.pool3 = nn.MaxPool2d(kernel_size=2)                                        # [-1, 128, 4, 4]
        
#         # fully connected out
#         self.linear1 = nn.Linear(4*4*self.channel_sizes[3], self.linear_hidden_units)
#         self.linear2 = nn.Linear(self.linear_hidden_units, 10)
        
#         # Deepen and widen now if necessary
#         if deeper:
#             self.deepen()
#         if not thin:
#             self.widen()
        
    
#     def forward(self, x):
#         # convs
#         x = self.resblock1(x)
#         x = self.pool1(x)
#         x = self.resblock2(x)
#         x = self.pool2(x)
#         if self.deeper:
#             x = self.resblock3(x)
#         else:
#             # if ignoring 3rd block, then make it an identity (v. hacky here...)
#             buf = t.zeros((x.size()[0], self.channel_sizes[3], 8, 8), device=_get_device(x))
#             buf[:,:x.size()[1]] = x
#             x = buf
            
#         x = self.pool3(x)
        
#         # fc
#         x = x.view((-1, 4*4*self.channel_sizes[3]))
#         x = self.linear1(x)
#         x = F.relu(x)
#         x = self.linear2(x)
        
#         return x
    
    
#     def widen(self, init_type="stddev_match"):
#         # Num params before
#         old_num_params = self._count_params()
        
#         # Set that we're no longer thing
#         self.thin = False
        
#         # Pass on the widen call to all of the constituent residual blocks
#         self.resblock1.widen(self.channel_sizes[1], function_preserving=self.fn_preserving_transform, 
#                              init_type=self.widen_init_type)
#         self.resblock2.widen(self.channel_sizes[2], function_preserving=self.fn_preserving_transform, 
#                              init_type=self.widen_init_type)
#         if self.deeper: 
#             self.resblock3.widen(self.channel_sizes[3], function_preserving=self.fn_preserving_transform, 
#                                  init_type=self.widen_init_type)
        
#         # Compute the ratio of parameters
#         num_params = self._count_params()
#         weight_ratio = num_params / old_num_params
#         return weight_ratio
        
        
        
#     def deepen(self):
#         # Num params before
#         old_num_params = self._count_params()
        
#         # Set that we're no longer shallow
#         self.deeper = True
        
#         # Work out if we've widened yet or not
#         in_channels = self.channel_sizes[2]
#         out_channels = self.channel_sizes[3]
#         if not self.thin:
#             in_channels *= 2
#             out_channels *= 2
        
#         # Perform the deepening
#         self.resblock3 = R2R_residual_block_v2(input_channels=self.channel_sizes[2], 
#                                                output_channels=self.channel_sizes[3], 
#                                                identity_initialize=self.fn_preserving_transform,
#                                                noise_ratio=self.noise_ratio) # [-1, 64, 8, 8]
        
#         # Compute the ratio of parameters
#         num_params = self._count_params()
#         weight_ratio = num_params / old_num_params 
#         return weight_ratio
    
    
    
#     def _add_modules_to_optimizer(self, modules, optimizer):
#         """
#         Optimizer is a non optional parameter, but we pass none in from the initializer. It should never 
#         be None if we are calling widen/deepen from outside of the module. We allow it to be None for when 
#         its called (from the initializer) before an Optimizer is created.
#         """
#         if optimizer is not None:
#             for module in modules:
#                 for param_group in module.parameters():
#                     optimizer.add_param_group({'params': param_group})
    
    
#     def _count_params(self):
#         """
#         Comput the number of parameters
        
#         :return: the number of parameters
#         """
#         total_num_params = 0
#         for parameter in self.parameters():
#             num_params = np.prod(t.tensor(parameter.size()).numpy())
#             total_num_params += num_params
#         return total_num_params
    
    
#     def _magnitude(self):
#         """
#         Compute the number of parameters, and the average magnitude of the parameters
#         Uses a (probably over complicated) running average
        
#         :return: Numbers of parameters, and, mean *magnitude* of parameters
#         """
#         total_num_params = 0
#         params_mean_mag = 0.0
#         for parameter in self.parameters():
#             num_params = np.prod(t.tensor(parameter.size()).numpy())
#             ratio = num_params / (total_num_params + num_params)
#             params_mean_mag = ratio * np.mean(np.abs(parameter.data.cpu().numpy())) + (1.0 - ratio) * params_mean_mag
#             total_num_params += num_params
#         return (total_num_params, params_mean_mag)
