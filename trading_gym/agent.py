import tensorflow as tf
import collections


class DQNAgent(object):

    def __init__(self,
                 action_num,
                 state_shape,
                 sess=None):
        """
        Args:
            action_num: shape of action. ex: 3
            state_shape: shape of action. ex: (10, 10)
        """
        if sess is None:
            self._sess = tf.Session()
        else:
            self._sess = sess

        self.state_shape = state_shape
        self.action_num = action_num
        self.__build_netword()
        self.train_op = self.__build_train_op()

    def _get_network_type(self):
        return collections.namedtuple('DQN_network', ['q_values'])

    def _network_template(self, state):
        state = tf.cast(state, tf.float32)
        _, w, h = state.shape
        net = tf.reshape(state, [-1, w])
        net = tf.contrib.layers.linear(net, 32, activation_fn=tf.nn.relu)
        q_values = tf.contrib.layers.linear(net, 32, activation_fn=None)
        return self._get_network_type()(q_values)

    def __build_netword(self):
        self.state_input = tf.placeholder(tf.float32, self.state_shape,
                                          name='state_input')

        self.online_net = tf.make_template('Online', self._network_template)
        self.target_net = tf.make_template('Online', self._network_template)
        self.online_net_outputs = self.online_net(self.state_ph)
        self._q_argmax = tf.argmax(self.online_net_outputs.q_values, axis=1)[0]

        self._replay_net_outputs = self.online_convnet(self._replay.states)
        self._replay_next_target_net_outputs = self.target_net(
            self._replay.next_states)

    def __build_train_op(self):
        self.q_target = tf.placeholder(
            tf.float32, [None, self.n_actions], name='Q_target')
        self.loss = tf.reduce_mean(
            tf.squared_difference(self.q_target, self.q_eval))
        return tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

    def __build_replay_buffer(self):
        """
            todo
        """
        pass

    def train(self):
        self._sess.run(self._train_op)
