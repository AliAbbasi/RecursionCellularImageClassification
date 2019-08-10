import data_loader
import model
import datetime
import tensorflow as tf
from functools import reduce
import numpy as np
import os, shutil, time

#----------------------------------------------------------------------------------------------------------------------

## Model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float  ('learning_rate', 0.0001,                                       "Initial learning rate.")
flags.DEFINE_float  ('dropout',       0.5,                                          "Dropout probability.")
flags.DEFINE_integer('max_steps',     10*1000*1000,                                 "Number of steps to run trainer.") 
flags.DEFINE_integer('batch_size',    512,                                          "Batch size. Must divide evenly into the dataset sizes.") 
flags.DEFINE_string ('train_path',    "I:\\Cellular\\saved_npy_data\\train\\",      "train data path")
flags.DEFINE_string ('valid_path',    "I:\\Cellular\\saved_npy_data\\validation\\", "validation data path")
flags.DEFINE_integer('input_size0',   128,                                          "input data shape")
flags.DEFINE_integer('input_size1',   128,                                          "input data shape")
flags.DEFINE_integer('input_size2',   6,                                            "input data shape")
flags.DEFINE_integer('output_size',   1108,                                         "Number of classes")
flags.DEFINE_boolean('restore',       True,                                         "restore saved weights")
flags.DEFINE_string ('weights',       "trained_weights_10000.meta",                 "restore saved weights")
flags.DEFINE_boolean('train',         True,                                         "train of test phase")
flags.DEFINE_string ('directory',     "saved_weights\\",                            "the directory for save weights" )
flags.DEFINE_string ('logs',          "logs\\",                                     "the directory for save weights" )
flags.DEFINE_boolean ('augmentation', True,                                         "augmentaion flag" )

#---------------------------------------------------------------------------------------------------------------------- 

def main(_):  
    train_data, train_label, valid_data, valid_label = data_loader.load_data_and_labels(FLAGS.train_path, FLAGS.valid_path, [FLAGS.input_size0, FLAGS.input_size1, FLAGS.input_size2])
    
    with tf.Graph().as_default():
        x         = tf.placeholder(tf.float32, [None, FLAGS.input_size0, FLAGS.input_size1, FLAGS.input_size2 ])
        y         = tf.placeholder(tf.float32, [None, FLAGS.output_size])
        keep_prob = tf.placeholder(tf.float32) 
        
        deep_model = model.Model([FLAGS.input_size0, FLAGS.input_size1, FLAGS.input_size2], x, y, FLAGS.learning_rate, keep_prob)
        
        init_var = tf.global_variables_initializer() 
        saver    = tf.train.Saver() 
        
        ## shows the model parameters number
        size = lambda v: reduce(lambda x, y: x*y, v.get_shape().as_list())
        n = sum(size(v) for v in tf.trainable_variables())
        print ("\r\n------------------------\r\n")
        print ("Model size: %dK" % (n/1000,))
        print ("\r\n------------------------\r\n")
        
        config = tf.ConfigProto() 
        config.gpu_options.allow_growth = True  
        with tf.Session(config=config) as sess:  
            sess.run(init_var) 
            
            summary_writer_train = tf.summary.FileWriter(FLAGS.logs + "train", graph=tf.get_default_graph())
            summary_writer_valid = tf.summary.FileWriter(FLAGS.logs + "valid", graph=tf.get_default_graph()) 
            
            # restore model trained weights
            if FLAGS.restore:
                if os.path.exists(FLAGS.directory + FLAGS.weights ): 
                    new_saver = tf.train.import_meta_graph(FLAGS.directory + FLAGS.weights)
                    new_saver.restore(sess, tf.train.latest_checkpoint(FLAGS.directory))  
                    print ("\r\n------------ Trained weights restored. ------------\r\n")
            
            # prevent to add extra node to graph during training        
            tf.get_default_graph().finalize()  
            
            # get the updateable operation from graph for batch norm layer
            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            
            # -------------- train phase --------------
            step           = 0 
            valid_loss     = 0 
            train_loss     = 0 
            valid_accuracy = 0 
            train_accuracy = 0 
            train_summary  = None
            valid_summary  = None
            
            ## main training loop
            while(step < FLAGS.max_steps):    
            
                x_batch, y_batch = data_loader.get_batch_data(train_data, train_label, FLAGS.batch_size, FLAGS.augmentation)   
                
                with tf.control_dependencies(extra_update_ops):  
                    # s = time.time()
                    sess.run([deep_model.optimizer], feed_dict={x: x_batch, y: y_batch, keep_prob: FLAGS.dropout})  
                    # print ("train opt: ", time.time() - s)
                    
                    # -------------- training acc loss  --------------
                    if step%2 == 0: 
                        ## train loss and accuracy
                        # s = time.time()
                        train_loss, train_accuracy, train_summary = sess.run([deep_model.loss, deep_model.accuracy, deep_model.sum], feed_dict={x: x_batch, y: y_batch, keep_prob: 1.0})  
                        # print ("train acc: ", time.time() - s)
                        
                        print ("%s, S:%5g, train_acc: %4.3g, train_loss: %2.3g, valid_acc: %4.3g, valid_loss: %2.3g"%
                          (str(datetime.datetime.now().time())[:-7], step, train_accuracy*1., train_loss*1., valid_accuracy*1., valid_loss*1.))
                        
                    # -------------- validation acc loss  --------------
                    if step%50 == 0: 
                        ## validation loss and accuracy
                        # s = time.time()
                        x_batch, y_batch = data_loader.get_batch_data(valid_data, valid_label, FLAGS.batch_size)  
                        valid_loss, valid_accuracy, valid_summary = sess.run([deep_model.loss, deep_model.accuracy, deep_model.sum], feed_dict={x: x_batch, y: y_batch, keep_prob: 1.0}) 
                        # print ("valid acc: ", time.time() - s)   
                    
                    # -------------- write summary on Tensorboard --------------                    
                    summary_writer_train.add_summary(train_summary, step)  
                    summary_writer_valid.add_summary(valid_summary, step)   
                    
                    # -------------- save weights --------------
                    if step%10000 == 0: 
                        print ("saving the weights...!")
                        saver.save(sess, FLAGS.directory + 'trained_weights_' + str(step))
                        
                    # --------------------------------------------- 
                    step += 1    
                    
            print(" --- \r\n --- \r\n  Trainig process is done after " + str(FLAGS.max_steps) + " iterations. \r\n --- \r\n ---")
            
#---------------------------------------------------------------------------------------------------------------------- 

if __name__ == '__main__': 
    tf.app.run()
    
    