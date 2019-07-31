import data_loader
import model
import datetime

#----------------------------------------------------------------------------------------------------------------------

# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float  ('learning_rate', 0.0001,                                       "Initial learning rate.")
flags.DEFINE_float  ('dropout',       0.5,                                          "Dropout probability.")
flags.DEFINE_integer('max_steps',     20000,                                        "Number of steps to run trainer.") 
flags.DEFINE_integer('batch_size',    32,                                           "Batch size. Must divide evenly into the dataset sizes.") 
flags.DEFINE_string ('train_path',    "I:\\Cellular\\saved_npy_data\\train\\",      "train data path")
flags.DEFINE_string ('valid_path',    "I:\\Cellular\\saved_npy_data\\validation\\", "validation data path")
flags.DEFINE_list   ('input_size',    64,64,6,                                      "input data shape")
flags.DEFINE_integer('output_size',   1108,                                         "Number of classes")
flags.DEFINE_boolean('restore',       False,                                        "restore saved weights")
flags.DEFINE_boolean('train',         True,                                        "train of test phase")

#----------------------------------------------------------------------------------------------------------------------

def count_params():
    "print number of trainable variables"
    size = lambda v: reduce(lambda x, y: x*y, v.get_shape().as_list())
    n = sum(size(v) for v in tf.trainable_variables())
    print "Model size: %dK" % (n/1000,)
    
#----------------------------------------------------------------------------------------------------------------------

def train(): 
    train_data, train_label, valid_data, valid_label = data_loader.load_data_and_labels(FLAGS.train_path, FLAGS.valid_path, FLAGS.input_size)
    
    x        = tf.placeholder(tf.float16, [None, FLAGS.input_size[0], FLAGS.input_size[1], FLAGS.input_size[2] ])
    y        = tf.placeholder(tf.float16, [None, FLAGS.output_size])
    keepProb = tf.placeholder(tf.float16) 
    
    dnn_obj  = model.DNN(FLAGS.input_size, x, y, keepProb)
    init_var = tf.global_variables_initializer() 
    saver    = tf.train.Saver() 
    
    count_params()
    
    with tf.Session(config=tf.ConfigProto(gpu_options.allow_growth = True)) as sess:  
        sess.run(init_var) 
        
        # restore model weights
        if FLAGS.restore:
            if os.path.exists(directory + '/my-model.meta'): 
                new_saver = tf.train.import_meta_graph(directory + '/my-model.meta')
                new_saver.restore(sess, tf.train.latest_checkpoint(directory))  
                print ("\r\n------------ Saved weights restored. ------------")
        
        # prevent to add extra node to graph during training        
        tf.get_default_graph().finalize()  
        
        # get the updateable operation from graph for batch norm layer
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        
        # -------------- test phase --------------
        if FLAGS.train == False:  
            #get_results(sess)  # TODO 
            print ("done")
            sys.exit(0)
            
        # -------------- train phase --------------
        step         = 1  
        train_cost   = []
        valid_cost   = []
        train_accu1  = []
        train_accu2  = []
        valid_accu1  = []
        valid_accu2  = [] 
        accu1tr, accu2tr = 0, 0
        
        while(step < FLAGS.max_steps):    
        
            x_batch, y_batch = fetch_x_y(train_data, batch_threshold)  
            
            with tf.control_dependencies(extra_update_ops):  
                cost, _ = sess.run([dnn_obj.loss_function, dnn_obj.optimizer], feed_dict={x: x_batch, y: y_batch, lr: FLAGS.learning_rate, keep_prob: FLAGS.dropout})    
                train_cost.append(cost) 
            
            # -------------- prints --------------
            if step%1 == 0: 
                print ("%s , S:%3g , lr:%g , accu1: %4.3g , Cost: %2.3g "% ( str(datetime.datetime.now().time())[:-7], step, learning_rate, accu1tr,  cost ))
            
            ### -------------- accuracy calculator --------------  
            ##if step % show_accuracy_step == 0 and show_accuracy:   
            ##    accu1tr, accu2tr = accuFun(sess, x_batch, y_batch, batch_size)  
            ##    train_accu1.append(accu1tr)
            ##    train_accu2.append(accu2tr) 
            ##    
            ##    # valid accuray
            ##    v_x_batch, v_y_batch = fetch_x_y(test_data, len(test_data)) 
            ##    accu1v, accu2v = accuFun(sess, v_x_batch, v_y_batch, batch_size)  
            ##    valid_accu1.append(accu1v)
            ##    valid_accu2.append(accu2v)
            ##    logging.info("accu1v: %4.3g , accu2v: %4.3g "% ( accu1v, accu2v ))
            ##    print       ("accu1v: %4.3g , accu2v: %4.3g "% ( accu1v, accu2v ))
            ##    
            ### -------------- save mode, write cost and accuracy --------------  
            ##if step % save_model_step == 0 and save_model: 
            ##    logging.info("Saving the model...") 
            ##    print       ("Saving the model...") 
            ##    saver.save(sess, directory + '/my-model')
            ##    logging.info("creating cost and accuray plot files...") 
            ##    print       ("creating cost and accuray plot files...")
            ##    utils.write_cost_accuray_plot(directory, train_cost, valid_cost, train_accu1, train_accu2, valid_accu1, valid_accu2) 
            ##    
            ### -------------- visualize scenes -------------- 
            ##if step % visualize_scene_step == 0 and visualize_scene:
            ##    show_result(sess)
                
            # --------------------------------------------- 
            step += 1    
            
        print(" --- \r\n --- \r\n  Trainig process is done after " + str(max_iter) + " iterations. \r\n --- \r\n ---")
#----------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__": 
    tf.app.run()
    