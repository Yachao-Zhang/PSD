import tensorflow as tf

# baseline /home/zcc/3d/HLContext/log/sem_seg/RandLANet_baseline/2020-10-10-area5/checkpoints/snapshots
ckpt = tf.train.get_checkpoint_state("/home/zcc/3d/HLContext/log/sem_seg/RandLANet_baseline/2020-10-10-area5/checkpoints/snapshots").model_checkpoint_path
saver = tf.train.import_meta_graph(ckpt+'.meta')
variables = tf.trainable_variables()
total_parameters = 0
for variable in variables:
    shape = variable.get_shape()
    variable_parameters = 1
    for dim in shape:
        # print(dim)
        variable_parameters *= dim.value
    # print(variable_parameters)
    total_parameters += variable_parameters
print(total_parameters)