import data_utils
import tensorflow as tf

tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_integer("Matrix_side_len", 80, "记忆矩阵的边长，总词(字)数为此数平方")
tf.app.flags.DEFINE_integer("character_size", 80 * 80, "总识别字数，应为矩阵边长平方")
tf.app.flags.DEFINE_integer("window_size", 20, "每多少字，窗口大小")
tf.app.flags.DEFINE_string("data_dir", "/tmp", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "/tmp", "Training directory.")
tf.app.flags.DEFINE_string("from_train_data", None, "Training data.")
tf.app.flags.DEFINE_string("to_train_data", None, "Training data.")
tf.app.flags.DEFINE_string("from_dev_data", None, "Training data.")
tf.app.flags.DEFINE_string("to_dev_data", None, "Training data.")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 200,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("apply", False,
                            "Set to True for interactive decoding.")

PARAMETERS = tf.app.flags.FLAGS


def read_data():
  return


def train():
  if PARAMETERS.from_train_data and PARAMETERS.to_train_data:
    from_train_data = PARAMETERS.from_train_data
    to_train_data = PARAMETERS.to_train_data
    from_dev_data = from_train_data
    to_dev_data = to_train_data
    if PARAMETERS.from_dev_data and PARAMETERS.to_dev_data:
      from_dev_data = PARAMETERS.from_dev_data
      to_dev_data = PARAMETERS.to_dev_data
    from_train, to_train, from_dev, to_dev, _ = data_utils.prepare_data(
      PARAMETERS.data_dir,
      from_train_data,
      to_train_data,
      from_dev_data,
      to_dev_data,
      PARAMETERS.character_size)

  else:
      print("Please Specify the file path in %s" % PARAMETERS.data_dir)
  
  
def apply():
  return


def main(_):
  if PARAMETERS.apply:
    apply()
  else:
    train()

if __name__ == "__main__":
  tf.app.run()