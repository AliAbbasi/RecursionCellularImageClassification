import tensorflow as tf
import numpy as np

#----------------------------------------------------------------------------------------------------------------------

class Model: 

    cardinality          = 8 # how many split  
    blocks               = 3 # res_block (split + transition)

    #----------------------------------------------------------------------------------------------------------------------   
    
    def parameters(self):
    
        wFC  = [4*4*64, 512]  # 128 
        wOut = [wFC[1], 1108]
        
        bFC  = wFC[1]
        bOut = wOut[1]
        
        params_w = {
                    'wFC'  : tf.get_variable('wFC' , shape=wFC,  initializer=tf.contrib.layers.xavier_initializer()),  
                    'wOut' : tf.get_variable('wOut', shape=wOut, initializer=tf.contrib.layers.xavier_initializer())                
                   }
        params_b = { 
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
        
        def conv_layer(input, filter, kernel, stride, padding='SAME', layer_name="conv"):
            with tf.name_scope(layer_name):
                network = tf.layers.conv2d(inputs=input, use_bias=True, filters=filter, kernel_size=kernel, strides=stride, padding=padding)
                return network
                
        #-------------------------------------------------------------------------------------
            
        def maxpool2d(x, k=2):
            return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')
        
        #------------------------------------------------------------------------------------- 
        
        def split_layer(input_x, stride, layer_name):
            with tf.name_scope(layer_name) :
                layers_split = list()
                for i in range(4) : # cardinality
                    splits = transform_layer(input_x, stride=stride, scope=layer_name + '_splitN_' + str(i))
                    layers_split.append(splits)

                return tf.concat(layers_split, axis=3)
        
        #------------------------------------------------------------------------------------- 
        
        def transform_layer(x, stride, scope):
            with tf.name_scope(scope) :
                x = conv_layer(x, filter=16, kernel=[1,1], stride=stride, layer_name=scope+'_conv1') 
                x = tf.layers.batch_normalization(x)
                x = tf.nn.relu(x) 
                
                x = conv_layer(x, filter=16, kernel=[3,3], stride=1, layer_name=scope+'_conv2')
                x = tf.layers.batch_normalization(x)
                x = tf.nn.relu(x)
                return x
                
        #------------------------------------------------------------------------------------- 
        
        def transition_layer(x, out_dim, scope):
            with tf.name_scope(scope):
                x = conv_layer(x, filter=out_dim, kernel=[1,1], stride=1, layer_name=scope+'_conv1')
                x = tf.layers.batch_normalization(x)  
                return x
                
        #------------------------------------------------------------------------------------- 
        
        def residual_layer(input_x, out_dim, layer_num, res_block=3): 
            for i in range(res_block): 
                input_dim = int(np.shape(input_x)[-1])
                
                flag = False
                stride = 1
                x = split_layer(input_x, stride=stride, layer_name='split_layer_'+layer_num+'_'+str(i))
                x = transition_layer(x, out_dim=out_dim, scope='trans_layer_'+layer_num+'_'+str(i))
                
                input_x = tf.nn.relu(x + input_x)

            return input_x 
            
        #-------------------------------------------------------------------------------------   
        conv1_1 = conv_layer(self.x,  filter=32, kernel=[3,3], stride=1, layer_name='conv1_1')
        conv1_2 = conv_layer(conv1_1, filter=64, kernel=[1,1], stride=1, layer_name='conv1_2')
        conv1_3 = conv_layer(conv1_2, filter=32, kernel=[3,3], stride=1, layer_name='conv1_3')
        merge1  = tf.concat([conv1_1, conv1_3], 3)  
        res1 = residual_layer(merge1, out_dim=64, layer_num='1')
        res1 = maxpool2d(res1)
        
        conv2_1 = conv_layer(res1,    filter=32, kernel=[3,3], stride=1, layer_name='conv2_1')
        conv2_2 = conv_layer(conv2_1, filter=64, kernel=[1,1], stride=1, layer_name='conv2_2')
        conv2_3 = conv_layer(conv2_2, filter=32, kernel=[3,3], stride=1, layer_name='conv2_3')
        merge2  = tf.concat([conv2_1, conv2_3], 3) 
        res2 = residual_layer(merge2, out_dim=64, layer_num='2')
        res2 = maxpool2d(res2)
        
        conv3_1 = conv_layer(res2,    filter=32, kernel=[3,3], stride=1, layer_name='conv3_1')
        conv3_2 = conv_layer(conv3_1, filter=64, kernel=[1,1], stride=1, layer_name='conv3_2')
        conv3_3 = conv_layer(conv3_2, filter=32, kernel=[3,3], stride=1, layer_name='conv3_3')
        merge3  = tf.concat([conv3_1, conv3_3], 3) 
        res3 = residual_layer(merge3, out_dim=64, layer_num='3')
        res3 = maxpool2d(res3) 
        
        conv4_1 = conv_layer(res3,    filter=32, kernel=[3,3], stride=1, layer_name='conv4_1')
        conv4_2 = conv_layer(conv4_1, filter=64, kernel=[1,1], stride=1, layer_name='conv4_2')
        conv4_3 = conv_layer(conv4_2, filter=32, kernel=[3,3], stride=1, layer_name='conv4_3')
        merge4  = tf.concat([conv4_1, conv4_3], 3) 
        res4 = residual_layer(merge4, out_dim=64, layer_num='4')
        res4 = maxpool2d(res4) 
        
        conv5_1 = conv_layer(res4,    filter=32, kernel=[3,3], stride=1, layer_name='conv5_1')
        conv5_2 = conv_layer(conv5_1, filter=64, kernel=[1,1], stride=1, layer_name='conv5_2')
        conv5_3 = conv_layer(conv5_2, filter=32, kernel=[3,3], stride=1, layer_name='conv5_3')
        merge5  = tf.concat([conv5_1, conv5_3], 3) 
        res5 = residual_layer(merge5, out_dim=64, layer_num='5')
        res5 = maxpool2d(res5) 
        
        ## Fully Connected
        fcLyr_1 = tf.reshape(res5, [-1, self.params_w['wFC'].get_shape().as_list()[0]])
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
        ## focal_loss = tf.reduce_mean(focal_loss(self.y, tf.nn.softmax(self.score)))
        
        weights_penalty = 0.0
        for w in self.params_w:
            weights_penalty += tf.nn.l2_loss(self.params_w[w]) * 0.001
            
        return (local_loss + weights_penalty ) ##+ focal_loss)
        
    
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
    