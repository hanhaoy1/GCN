import time
import tensorflow as tf

from utils import *
from gcn import GCN

seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'cora', 'Dataset string')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate')
flags.DEFINE_integer('epochs', 200, 'Epochs')
flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss')
flags.DEFINE_integer('early_stopping', 10, 'Tolerate for early stopping')

adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(FLAGS.dataset)

features = preprocess_features(features)
support = preprocess_adj(adj)

placeholders = {
    'support': tf.sparse_placeholder(tf.float32),
    'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder_with_default(tf.int32)
}

model = GCN(placeholders, input_dim=features[2][1], logging=True)

sess = tf.Session()


def evaluate(features, support, labels, mask, placeholders):
    t = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
    outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], time.time() - t


sess.run(tf.global_variables_initializer())
cost_val = []
for epoch in range(FLAGS.epochs):
    t = time.time()
    feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)
    cost, acc, duration = evaluate(features, support, y_val, val_mask, placeholders)
    cost_val.append(cost)
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
          "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
          "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))

print("Optimization Finished")

test_cost, test_acc, test_duration = evaluate(features, support, y_test, test_mask, placeholders)
print("Test set results:", "cost=", "{:.5f}".format(test_cost),
      "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))

