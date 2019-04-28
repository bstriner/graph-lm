import tensorflow as tf

print("Compile flags: {}".format(" ".join(tf.sysconfig.get_compile_flags())))
print("Link flags: {}".format(" ".join(tf.sysconfig.get_link_flags())))
