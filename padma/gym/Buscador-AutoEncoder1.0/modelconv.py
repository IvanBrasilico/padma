import tensorflow as tf

def modelconv1(size):
    inputs_ = tf.placeholder(tf.float32, (None, size[1], size[0], 1), name='inputs')
    targets_ = tf.placeholder(tf.float32, (None, size[1], size[0], 1), name='targets')

    ### Encoder
    conv1 = tf.layers.conv2d(inputs_, 32, (3,3), padding='same', activation=tf.nn.relu)
    maxpool1 = tf.layers.max_pooling2d(conv1, (2,2), (2,2), padding='same')
    conv2 = tf.layers.conv2d(maxpool1, 16, (3,3), padding='same', activation=tf.nn.relu)
    maxpool2 = tf.layers.max_pooling2d(conv2, (2,2), (2,2), padding='same')
    conv3 = tf.layers.conv2d(maxpool2, 16, (3,3), padding='same', activation=tf.nn.relu)
    maxpool3 = tf.layers.max_pooling2d(conv3, (2,2), (2,2), padding='same')
    conv31 = tf.layers.conv2d(maxpool3, 8, (3,3), padding='same', activation=tf.nn.relu)
    maxpool31 = tf.layers.max_pooling2d(conv31, (2,2), (2,2), padding='same')
    conv32 = tf.layers.conv2d(maxpool31, 8, (3,3), padding='same', activation=tf.nn.relu)
    encoded = tf.layers.max_pooling2d(conv32, (2,2), (2,2), padding='same')

    ### Decoder
    upsample00 = tf.image.resize_nearest_neighbor(encoded, (int(size[1]/16),int(size[0]/16)))
    conv34 = tf.layers.conv2d(upsample00, 8, (3,3), padding='same', activation=tf.nn.relu)
    upsample0 = tf.image.resize_nearest_neighbor(encoded, (int(size[1]/8),int(size[0]/8)))
    conv34 = tf.layers.conv2d(upsample0, 8, (3,3), padding='same', activation=tf.nn.relu)
    upsample1 = tf.image.resize_nearest_neighbor(encoded, (int(size[1]/4),int(size[0]/4)))
    conv4 = tf.layers.conv2d(upsample1, 8, (3,3), padding='same', activation=tf.nn.relu)
    upsample2 = tf.image.resize_nearest_neighbor(conv4, (int(size[1]/2),int(size[0]/2)))
    conv5 = tf.layers.conv2d(upsample2, 8, (3,3), padding='same', activation=tf.nn.relu)
    upsample3 = tf.image.resize_nearest_neighbor(conv5, (int(size[1]),int(size[0])))
    conv6 = tf.layers.conv2d(upsample3, 16, (3,3), padding='same', activation=tf.nn.relu)
    logits = tf.layers.conv2d(conv6, 1, (3,3), padding='same', activation=None)

    decoded = tf.nn.sigmoid(logits, name='decoded')

    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets_, logits=logits)
    cost = tf.reduce_mean(loss)
    opt = tf.train.AdamOptimizer(0.001).minimize(cost)
     
    #sess = tf.Session()
    #saver = tf.train.Saver()
    #saver.restore(sess, "modelconv.ckpt")
    return encoded, decoded, loss, cost, opt
