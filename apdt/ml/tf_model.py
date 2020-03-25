import tensorflow as tf

def wavenet_weight(self, name, args, trainable=True):
    with tf.variable_scope(name) as scope:
        res_channel=args['res_channel']
        skip_channel=args['skip_channel']
        bits = 5
        w = {}
        w['kernel'] = tf.get_variable('kernel',[3, res_channel, 2*res_channel],initializer=tf.contrib.layers.xavier_initializer(), trainable=trainable)
        w['1_by_1_res'] = tf.get_variable('1_by_1_res',[1, res_channel, res_channel],initializer=tf.contrib.layers.xavier_initializer(), trainable=trainable)
        w['1_by_1_skip'] = tf.get_variable('1_by_1_skip',[1, res_channel, skip_channel],initializer=tf.contrib.layers.xavier_initializer(), trainable=trainable)
        w['1_by_1_x'] = tf.get_variable('1_by_1_x', [1, args['input_dim'], res_channel],initializer=tf.contrib.layers.xavier_initializer(), trainable=trainable)
        w['1_by_1_skip1'] = tf.get_variable(
                    '1_by_1_skip1', [1, skip_channel*args['DilatedConvLayers']+args['input_dim'], args['n_hidden']],
                    initializer=tf.contrib.layers.xavier_initializer(), trainable=trainable)
        w['1_by_1_skip2'] = tf.get_variable(
                    '1_by_1_skip2', [1, args['n_hidden'], 2**bits],
                    initializer=tf.contrib.layers.xavier_initializer(), trainable=trainable)
        return w

def WaveNet(self, input, weights, name, args):
    '''
    Map a time series into another using WaveNet.

    Input: a tensor shaped [batch_size, time_length, feature_dim]
    Output: a tensor shaped [batch_size, time_length, 1]
    '''
    def CasualResUnit(x,res_channel=args['res_channel'],skip_channel=args['skip_channel'],name='CasualResUnit'):
        n = x.shape[-1].value
        w = weights['kernel']
        _x = tf.pad(x,[[0,0],[2,0],[0,0]])
        y = tf.nn.conv1d(_x, w, 1, 'VALID')
        h,g = tf.split(y, 2, axis=-1)
        h = tf.tanh(h)
        g = tf.sigmoid(g)
        h = h*g
        w = weights['1_by_1_res']
        o = tf.nn.conv1d(h, w, 1, 'SAME')
        o = o+x
        w = weights['1_by_1_skip']
        skip = tf.nn.conv1d(h, w, 1, 'SAME')
        return o,skip
    
    def DilatedConv(x,dilated=args['dilated'],res_channel=args['res_channel'],skip_channel=args['skip_channel'],name='DilatedConv'):
        n = x.shape[-1].value
        l = x.shape[1].value
        if l%dilated != 0:
            print('Dilated Error, when dilated at '+str(dilated))
            exit()
        num = l//dilated
        with tf.variable_scope(name) as scope:
            x = tf.reshape(x,[-1,num,dilated,n])
            _out = []
            _skip = []
            for i in range(dilated):
                out, skip = CasualResUnit(x[:,:,i],res_channel=res_channel,skip_channel=skip_channel,name='CasualResUnit#'+str(i))
                _out.append(out)
                _skip.append(skip)
                if i==0:
                    scope.reuse_variables()
            o = tf.stack(_out,axis=2)
            o = tf.reshape(o,[-1,l,n])
            skip = tf.stack(_skip,axis=2)
            skip = tf.reshape(skip,[-1,l,skip_channel])
            return o,skip
    
    def u_law_encoder(x,bits=5):
        bits = 2**bits
        x = tf.clip_by_value(x,0.0,0.99999)
        x = tf.floor(tf.log(1+bits*x)*bits/np.log(1+bits))
        x = tf.one_hot(tf.cast(tf.squeeze(x,-1),tf.int32),depth=bits)
        return x
    
    def u_law_decoder(x,bits=5):
        x = tf.cast(x,tf.float32)
        x = (tf.exp((x+0.5)/(2**bits)*np.log(1.0+(2**bits)))-1.0)/(2**bits)
        return tf.expand_dims(x,-1)
    
    def SingleChannelNetwork(x,channel,bits,encoder,decoder,name='Channel'):
        with tf.variable_scope(name) as scope:
            y = x[:,1:,channel:channel+1]
            x = x[:,:-1,:]
            _skip = [x]
            w = weights['1_by_1_x']
            x = tf.nn.conv1d(x, w, 1, 'SAME')
            for i in range(args['DilatedConvLayers']):
                if i==0:
                    o,skip = DilatedConv(x,dilated=3**i,name='DilatedConv'+str(i))
                    _skip.append(skip)
                else:
                    o,skip = DilatedConv(o,dilated=3**i,name='DilatedConv'+str(i))
                    _skip.append(skip)
            skip = tf.concat(_skip,axis=-1)
            skip = tf.nn.relu(skip)
            w = weights['1_by_1_skip1']
            skip = tf.nn.conv1d(skip, w, 1, 'SAME')
            skip = tf.nn.relu(skip)
            w = weights['1_by_1_skip2']
            pred = tf.nn.conv1d(skip, w, 1, 'SAME')
            pred_prob = tf.nn.softmax(pred)
            pred = tf.nn.conv1d(pred_prob, decoder(tf.constant(np.arange(2**bits).reshape(1,2**bits))), 1, 'SAME')
            loss = tf.reduce_mean(tf.abs(pred-y))
            return pred,loss
    
    pred,loss = SingleChannelNetwork(input,0,5,u_law_encoder,u_law_decoder,name=name)
    return pred,loss
pass