import tensorflow as tf

#----------------------------------------------------------------------------------------------------------------------

class Model: 
    #----------------------------------------------------------------------------------------------------------------------   
    
    def parameters(self):
        params_w = {
                    'w1'   : tf.Variable(tf.truncated_normal([3, 3, self.input_shape[2], 8], stddev=0.1)),
                    'w2'   : tf.Variable(tf.truncated_normal([3, 3, 8, 16],  stddev=0.1)),
                    'w3'   : tf.Variable(tf.truncated_normal([3, 3, 16, 16], stddev=0.1)),
                    'w4'   : tf.Variable(tf.truncated_normal([3, 3, 16, 32], stddev=0.1)),
                    'w5'   : tf.Variable(tf.truncated_normal([3, 3, 32, 64], stddev=0.1)),
                    # 'w6'   : tf.Variable(tf.truncated_normal([64*4*4, 128],  stddev=0.1)),  # for 128
                    'w6'   : tf.Variable(tf.truncated_normal([64*2*2, 128],  stddev=0.1)),   # for 64
                    'w7'   : tf.Variable(tf.truncated_normal([128, 1108],    stddev=0.1))                    
                   }
        params_b = {
                    'b1'   : tf.Variable(tf.truncated_normal([8],    stddev=0.1)),     
                    'b2'   : tf.Variable(tf.truncated_normal([16],   stddev=0.1)),     
                    'b3'   : tf.Variable(tf.truncated_normal([16],   stddev=0.1)),     
                    'b4'   : tf.Variable(tf.truncated_normal([32],   stddev=0.1)),     
                    'b5'   : tf.Variable(tf.truncated_normal([64],   stddev=0.1)),     
                    'b6'   : tf.Variable(tf.truncated_normal([128],  stddev=0.1)),     
                    'b7'   : tf.Variable(tf.truncated_normal([1108], stddev=0.1))
                   }
    
        return params_w, params_b
    #----------------------------------------------------------------------------------------------------------------------
    
    def score(self): 
        def conv2d(x, W, b, strides=1):
            x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
            x = tf.nn.bias_add(x, b)
            return x
            
        #-------------------------------------------------------------------------------------
            
        def maxpool2d(x, k=2):
            return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')
        
        #-------------------------------------------------------------------------------------
        
        # 1)  
        convLyr_1_conv = conv2d (self.x, self.params_w['w1'], self.params_b['b1'])
        convLyr_1_relu = tf.nn.relu(convLyr_1_conv) 
        convLyr_1_pool = maxpool2d(convLyr_1_relu)
        print (convLyr_1_pool.get_shape())
        
        # 2)
        convLyr_2_conv = conv2d(convLyr_1_pool, self.params_w['w2'], self.params_b['b2'])
        convLyr_2_relu = tf.nn.relu(convLyr_2_conv)
        convLyr_2_pool = maxpool2d(convLyr_2_relu)
        print (convLyr_2_pool.get_shape())

        # 3)
        convLyr_3_conv = conv2d(convLyr_2_pool, self.params_w['w3'], self.params_b['b3'])
        convLyr_3_relu = tf.nn.relu(convLyr_3_conv)
        convLyr_3_pool = maxpool2d(convLyr_3_relu)
        print (convLyr_3_pool.get_shape())
        
        # 4
        convLyr_4_conv = conv2d(convLyr_3_pool, self.params_w['w4'], self.params_b['b4'])
        convLyr_4_relu = tf.nn.relu(convLyr_4_conv)
        convLyr_4_pool =  maxpool2d(convLyr_4_relu)
        print (convLyr_4_pool.get_shape())
        
        # 5
        convLyr_5_conv = conv2d(convLyr_4_pool, self.params_w['w5'], self.params_b['b5'])
        convLyr_5_relu = tf.nn.relu(convLyr_5_conv)
        convLyr_5_pool =  maxpool2d(convLyr_5_relu)
        print (convLyr_5_pool.get_shape())
        
        # Fully Connected
        fcLyr_1 = tf.reshape(convLyr_5_pool, [-1, self.params_w['w6'].get_shape().as_list()[0]])
        fcLyr_1 = tf.add(tf.matmul(fcLyr_1, self.params_w['w6']), self.params_b['b6'])
        fcLyr_1 = tf.nn.relu(fcLyr_1)
        fcLyr_1 = tf.nn.dropout(fcLyr_1, self.keep_prob)
        print (fcLyr_1.get_shape())
        
        netOut = tf.add(tf.matmul(fcLyr_1, self.params_w['w7']), self.params_b['b7'])
        print (netOut.get_shape())
        
        return netOut

    #----------------------------------------------------------------------------------------------------------------------
    
    def loss_function(self):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.score, labels=self.y))    
    
    #----------------------------------------------------------------------------------------------------------------------
    
    def optimizer(self):
        return tf.train.AdamOptimizer(learning_rate = self.lr).minimize(self.loss)
        
    #----------------------------------------------------------------------------------------------------------------------
    
    def accuracy_function(self):
        correct_pred = tf.equal(tf.argmax(self.score,1), tf.argmax(self.y,1))
        return(tf.reduce_mean(tf.cast(correct_pred, tf.float32)))   
        
    #----------------------------------------------------------------------------------------------------------------------
    
    def __init__(self, input_shape, x, y, lr, keep_prob): 
        self.input_shape  = input_shape 
        self.x            = x
        self.y            = y
        self.lr           = lr 
        self.keep_prob    = keep_prob

        [self.params_w, self.params_b] = Model.parameters(self)   
        self.score                     = Model.score(self)   
        self.loss                      = Model.loss_function(self)   
        self.optimizer                 = Model.optimizer(self)   
        self.accuracy                  = Model.accuracy_function(self)   
        
    #----------------------------------------------------------------------------------------------------------------------
    