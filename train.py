import tensorflow as tf
from p2m.utils import *
from p2m.models import GCN
from p2m.fetcher import *
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


seed = 1024
np.random.seed(seed)
tf.set_random_seed(seed)

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_list', '/content/mesh_gen/Data/train_list.txt', 'Data list.') 
flags.DEFINE_float('learning_rate', 1e-5, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 5, 'Number of epochs to train.')
flags.DEFINE_integer('hidden', 256, 'Number of units in hidden layer.') 
flags.DEFINE_integer('feat_dim', 963, 'Number of units in feature layer.') 
flags.DEFINE_integer('coord_dim', 3, 'Number of units in output layer.') 
flags.DEFINE_float('weight_decay', 5e-6, 'Weight decay for L2 loss.')


num_blocks = 3
num_supports = 2
placeholders = {
    'features': tf.placeholder(tf.float32, shape=(None, 3)),
    'img_inp': tf.placeholder(tf.float32, shape=(224, 224, 3)),
    'labels': tf.placeholder(tf.float32, shape=(None, 6)),
    'support1': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'support2': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'support3': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'faces': [tf.placeholder(tf.int32, shape=(None, 4)) for _ in range(num_blocks)],  
    'edges': [tf.placeholder(tf.int32, shape=(None, 2)) for _ in range(num_blocks)], 
    'lape_idx': [tf.placeholder(tf.int32, shape=(None, 10)) for _ in range(num_blocks)], 
    'pool_idx': [tf.placeholder(tf.int32, shape=(None, 2)) for _ in range(num_blocks-1)] 
}
model = GCN(placeholders, logging=True)


data = DataFetcher(FLAGS.data_list)
data.setDaemon(True)
data.start()
config=tf.ConfigProto()

config.allow_soft_placement=True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())



train_loss = open('record_train_loss.txt', 'a')
train_loss.write('Start training, lr =  %f\n'%(FLAGS.learning_rate))
pkl = pickle.load(open('/content/mesh_gen/Data/ellipsoid/info_ellipsoid.dat', 'rb'))
feed_dict = construct_feed_dict(pkl, placeholders)

train_number = data.number
for epoch in range(FLAGS.epochs):
	all_loss = np.zeros(train_number,dtype='float32') 
	for iters in range(train_number):
		
		img_inp, y_train, data_id = data.fetch()
		feed_dict.update({placeholders['img_inp']: img_inp})
		feed_dict.update({placeholders['labels']: y_train})

		
		_, dists,out1,out2,out3 = sess.run([model.opt_op,model.loss,model.output1,model.output2,model.output3], feed_dict=feed_dict)
		all_loss[iters] = dists
		mean_loss = np.mean(all_loss[np.where(all_loss)])
		if (iters+1) % 128 == 0:
			print 'Epoch %d, Iteration %d'%(epoch + 1,iters + 1)
			print 'Mean loss = %f, iter loss = %f, %d'%(mean_loss,dists,data.queue.qsize())
	
	model.save(sess)
	train_loss.write('Epoch %d, loss %f\n'%(epoch+1, mean_loss))
	train_loss.flush()

data.shutdown()
print 'Training Finished!'
