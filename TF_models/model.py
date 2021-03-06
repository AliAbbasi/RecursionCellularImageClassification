import tensorflow as tf

#----------------------------------------------------------------------------------------------------------------------

class Model: 
    #----------------------------------------------------------------------------------------------------------------------   
    
    def parameters(self):
    
        w1   = [5, 5, self.input_shape[2],              8      ]
        
        w2   = [3, 3, w1[3],                            16     ]
        w3   = [1, 1, w2[3],                            w2[3]  ]
        w4   = [3, 3, w2[3],                            w2[3]  ]
        
        w5   = [3, 3, w1[3]+w2[3],                      16     ]
        w6   = [1, 1, w5[3],                            w5[3]  ]
        w7   = [3, 3, w5[3],                            w5[3]  ]
        
        w8   = [3, 3, w1[3]+w2[3]+w5[3],                16     ]
        w9   = [1, 1, w8[3],                            w8[3]  ]
        w10  = [3, 3, w8[3],                            w8[3]  ]
        
        w11  = [3, 3, w1[3]+w2[3]+w5[3]+w8[3],          32     ]
        w12  = [1, 1, w11[3],                           w11[3] ]
        w13  = [3, 3, w11[3],                           w11[3] ]
        
        w14  = [3, 3, w1[3]+w2[3]+w5[3]+w8[3]+w11[3],   64    ]
        w15  = [1, 1, w14[3],                           w14[3] ]
        w16  = [3, 3, w14[3],                           w14[3] ]
        
        
        wFC  = [4*4*(w1[3]+w2[3]+w5[3]+w8[3]+w11[3]+w14[3]), 128]  # 128 
        wOut = [wFC[1], 1108]
        
        b1   = w1[3]
        b2   = w2[3]
        b3   = w3[3]
        b4   = w4[3]
        b5   = w5[3]
        b6   = w6[3]
        b7   = w7[3]
        b8   = w8[3]
        b9   = w9[3]
        b10  = w10[3]
        b11  = w11[3]
        b12  = w12[3]
        b13  = w13[3]
        b14  = w14[3]
        b15  = w15[3]
        b16  = w16[3]
        bFC  = wFC[1]
        bOut = wOut[1]
        
        params_w = {
                    'w1'   : tf.get_variable('w1'  , shape=w1,   initializer=tf.contrib.layers.xavier_initializer()),
                    'w2'   : tf.get_variable('w2'  , shape=w2,   initializer=tf.contrib.layers.xavier_initializer()),  
                    'w3'   : tf.get_variable('w3'  , shape=w3,   initializer=tf.contrib.layers.xavier_initializer()),
                    'w4'   : tf.get_variable('w4'  , shape=w4,   initializer=tf.contrib.layers.xavier_initializer()),  
                    'w5'   : tf.get_variable('w5'  , shape=w5,   initializer=tf.contrib.layers.xavier_initializer()),  
                    'w6'   : tf.get_variable('w6'  , shape=w6,   initializer=tf.contrib.layers.xavier_initializer()),  
                    'w7'   : tf.get_variable('w7'  , shape=w7,   initializer=tf.contrib.layers.xavier_initializer()),   
                    'w8'   : tf.get_variable('w8'  , shape=w8,   initializer=tf.contrib.layers.xavier_initializer()),  
                    'w9'   : tf.get_variable('w9'  , shape=w9,   initializer=tf.contrib.layers.xavier_initializer()),  
                    'w10'  : tf.get_variable('w10' , shape=w10,  initializer=tf.contrib.layers.xavier_initializer()),  
                    'w11'  : tf.get_variable('w11' , shape=w11,  initializer=tf.contrib.layers.xavier_initializer()),   
                    'w12'  : tf.get_variable('w12' , shape=w12,  initializer=tf.contrib.layers.xavier_initializer()),  
                    'w13'  : tf.get_variable('w13' , shape=w13,  initializer=tf.contrib.layers.xavier_initializer()),  
                    'w14'  : tf.get_variable('w14' , shape=w14,  initializer=tf.contrib.layers.xavier_initializer()),   
                    'w15'  : tf.get_variable('w15' , shape=w15,  initializer=tf.contrib.layers.xavier_initializer()),  
                    'w16'  : tf.get_variable('w16' , shape=w16,  initializer=tf.contrib.layers.xavier_initializer()),  
                    'wFC'  : tf.get_variable('wFC' , shape=wFC,  initializer=tf.contrib.layers.xavier_initializer()),  
                    'wOut' : tf.get_variable('wOut', shape=wOut, initializer=tf.contrib.layers.xavier_initializer())                
                   }
        params_b = {
                    'b1'   : tf.get_variable('b1'  , shape=b1,   initializer=tf.zeros_initializer()),  
                    'b2'   : tf.get_variable('b2'  , shape=b2,   initializer=tf.zeros_initializer()),   
                    'b3'   : tf.get_variable('b3'  , shape=b3,   initializer=tf.zeros_initializer()), 
                    'b4'   : tf.get_variable('b4'  , shape=b4,   initializer=tf.zeros_initializer()),  
                    'b5'   : tf.get_variable('b5'  , shape=b5,   initializer=tf.zeros_initializer()),  
                    'b6'   : tf.get_variable('b6'  , shape=b6,   initializer=tf.zeros_initializer()),  
                    'b7'   : tf.get_variable('b7'  , shape=b7,   initializer=tf.zeros_initializer()),  
                    'b8'   : tf.get_variable('b8'  , shape=b8,   initializer=tf.zeros_initializer()),  
                    'b9'   : tf.get_variable('b9'  , shape=b9,   initializer=tf.zeros_initializer()),  
                    'b10'  : tf.get_variable('b10' , shape=b10,  initializer=tf.zeros_initializer()),  
                    'b11'  : tf.get_variable('b11' , shape=b11,  initializer=tf.zeros_initializer()),  
                    'b12'  : tf.get_variable('b12' , shape=b12,  initializer=tf.zeros_initializer()),  
                    'b13'  : tf.get_variable('b13' , shape=b13,  initializer=tf.zeros_initializer()),  
                    'b14'  : tf.get_variable('b14' , shape=b14,  initializer=tf.zeros_initializer()),  
                    'b15'  : tf.get_variable('b15' , shape=b15,  initializer=tf.zeros_initializer()),  
                    'b16'  : tf.get_variable('b16' , shape=b16,  initializer=tf.zeros_initializer()),   
                    'bFC'  : tf.get_variable('bFC' , shape=bFC,  initializer=tf.zeros_initializer()),      
                    'bOut' : tf.get_variable('bOut', shape=bOut, initializer=tf.zeros_initializer())  
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
        
        conv_1    = conv2d( self.x, self.params_w['w1'], self.params_b['b1']) 
        
        ## Residual Block #1
        conv_r1_1 = tf.layers.batch_normalization(tf.nn.relu( conv_1 )) 
        conv_r1_2 = tf.layers.batch_normalization(tf.nn.relu( conv2d( conv_r1_1, self.params_w['w2'], self.params_b['b2'] ) ))   
        conv_r1_3 = tf.layers.batch_normalization(tf.nn.relu( conv2d( conv_r1_2, self.params_w['w3'], self.params_b['b3'] ) )) 
        conv_r1_4 =                                           conv2d( conv_r1_3, self.params_w['w4'], self.params_b['b4'] )  
        merge_1   = tf.concat([conv_1, conv_r1_4], 3) 
        merge_1   = maxpool2d(merge_1)
        print (merge_1.get_shape())
        
        ## Residual Block #2
        conv_r2_1 = tf.layers.batch_normalization(tf.nn.relu( merge_1 ))  
        conv_r2_2 = tf.layers.batch_normalization(tf.nn.relu( conv2d( conv_r2_1, self.params_w['w5'], self.params_b['b5'] ) ))   
        conv_r2_3 = tf.layers.batch_normalization(tf.nn.relu( conv2d( conv_r2_2, self.params_w['w6'], self.params_b['b6'] ) )) 
        conv_r2_4 =                                           conv2d( conv_r2_3, self.params_w['w7'], self.params_b['b7'] )  
        merge_2   = tf.concat([merge_1, conv_r2_4], 3) 
        merge_2   = maxpool2d(merge_2)   
        print (merge_2.get_shape())        
        
        ## Residual Block #3
        conv_r3_1 = tf.layers.batch_normalization(tf.nn.relu( merge_2 ))  
        conv_r3_2 = tf.layers.batch_normalization(tf.nn.relu( conv2d( conv_r3_1, self.params_w['w8'],  self.params_b['b8']  ) ))   
        conv_r3_3 = tf.layers.batch_normalization(tf.nn.relu( conv2d( conv_r3_2, self.params_w['w9'],  self.params_b['b9']  ) )) 
        conv_r3_4 =                                           conv2d( conv_r3_3, self.params_w['w10'], self.params_b['b10'] ) 
        # conv_r3_4 = tf.nn.dropout(conv_r3_4, self.keep_prob)
        merge_3   = tf.concat([merge_2, conv_r3_4], 3)  
        merge_3   = maxpool2d(merge_3)
        print (merge_3.get_shape())
        
        ## Residual Block #4
        conv_r4_1 = tf.layers.batch_normalization(tf.nn.relu( merge_3 ))  
        conv_r4_2 = tf.layers.batch_normalization(tf.nn.relu( conv2d( conv_r4_1, self.params_w['w11'], self.params_b['b11'] ) ))   
        conv_r4_3 = tf.layers.batch_normalization(tf.nn.relu( conv2d( conv_r4_2, self.params_w['w12'], self.params_b['b12'] ) )) 
        conv_r4_4 =                                           conv2d( conv_r4_3, self.params_w['w13'], self.params_b['b13'] ) 
        # conv_r4_4 = tf.nn.dropout(conv_r4_4, self.keep_prob)  
        merge_4   = tf.concat([merge_3, conv_r4_4], 3) 
        merge_4   = maxpool2d(merge_4)
        print (merge_4.get_shape())
        
        ## Residual Block #5
        conv_r5_1 = tf.layers.batch_normalization(tf.nn.relu( merge_4 ))  
        conv_r5_2 = tf.layers.batch_normalization(tf.nn.relu( conv2d( conv_r5_1, self.params_w['w14'], self.params_b['b14'] ) ))   
        conv_r5_3 = tf.layers.batch_normalization(tf.nn.relu( conv2d( conv_r5_2, self.params_w['w15'], self.params_b['b15'] ) )) 
        conv_r5_4 =                                           conv2d( conv_r5_3, self.params_w['w16'], self.params_b['b16'] )  
        # conv_r5_4 = tf.nn.dropout(conv_r5_4, self.keep_prob)        
        merge_5   = tf.concat([merge_4, conv_r5_4], 3)     
        merge_5   = maxpool2d(merge_5)
        print (merge_5.get_shape())
        
        ## Fully Connected
        fcLyr_1 = tf.reshape(merge_5, [-1, self.params_w['wFC'].get_shape().as_list()[0]])
        fcLyr_1 = tf.add(tf.matmul(fcLyr_1, self.params_w['wFC']), self.params_b['bFC'])
        fcLyr_1 = tf.nn.relu(fcLyr_1)
        fcLyr_1 = tf.nn.dropout(fcLyr_1, self.keep_prob)
        print (fcLyr_1.get_shape())
        
        netOut = tf.add(tf.matmul(fcLyr_1, self.params_w['wOut']), self.params_b['bOut'])
        print (netOut.get_shape())
        
        return netOut 

    #----------------------------------------------------------------------------------------------------------------------
    
    def loss_function(self):
    
        def focal_loss(labels, logits, gamma=2.0, alpha=4.0): 
            epsilon = 1.e-9 
            num_cls = logits.shape[1] 
            model_out = tf.add(logits, epsilon) 
            ce = tf.multiply(labels, -tf.log(model_out))
            weight = tf.multiply(labels, tf.pow(tf.subtract(1., model_out), gamma))
            fl = tf.multiply(alpha, tf.multiply(weight, ce))
            reduced_fl = tf.reduce_max(fl, axis=1) 
            return reduced_fl 
        
        local_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.score, labels=self.y))
        # focal_loss = tf.reduce_mean(focal_loss(self.y, tf.nn.softmax(self.score)))
        
        weights_penalty = 0.0
        for w in self.params_w:
            weights_penalty += tf.nn.l2_loss(self.params_w[w]) * 0.001
            
        return (local_loss + weights_penalty)
        
    
    #----------------------------------------------------------------------------------------------------------------------
    
    def optimizer(self):
        return tf.train.AdamOptimizer(learning_rate = self.lr).minimize(self.loss)
        # return tf.train.GradientDescentOptimizer(learning_rate = self.lr).minimize(self.loss)
        
    #----------------------------------------------------------------------------------------------------------------------
    
    def accuracy_function(self):
        correct_pred = tf.equal(tf.argmax(self.score,1), tf.argmax(self.y,1))
        return(tf.reduce_mean(tf.cast(correct_pred, tf.float32)))   
        
    #----------------------------------------------------------------------------------------------------------------------
    
    def get_summaries(self):
        tf.summary.scalar("loss", self.loss ) 
        tf.summary.scalar("accuracy", self.accuracy) 
        merged_summary_op = tf.summary.merge_all()
        
        return merged_summary_op 
        
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
        self.sum                       = Model.get_summaries(self) 
        
    #----------------------------------------------------------------------------------------------------------------------
    