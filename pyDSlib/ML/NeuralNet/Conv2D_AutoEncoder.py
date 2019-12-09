from . import Conv2D as _Conv2D

import tensorflow as _tf
import tensorflow.keras as _keras

_pooling_layer = _tf.keras.layers.MaxPool2D
_loss= 'mae'
_optimizer = 'adam'
_kernel_size = (3,3)

def _build_graph(img_shape = (256, 256, 3),
                  batch_size = None ,
                  n_outputs_per_img = 2,
                  layers_per_group =  1,
                  initial_filter_size = 16,
                  max_filter_size = 512,
                  filter_scaling_factor = 2,
                  dense_scaling_factor = 20,
                  kernel_size = _kernel_size,
                  activation = 'relu',
                  final_activation = 'relu',
                  pooling_layer = _pooling_layer,
                  pool_size = (2,2),
                  batch_norm_rate = None,
                  dropout_layer_rate = None,
                  dropout_rate = 0.5,
                  verbose = 0):
    
    #build encoder using Conv2D module
    encoder = _Conv2D._build_graph(img_shape = img_shape,
                          batch_size = batch_size ,
                          n_outputs_per_img = n_outputs_per_img,
                          layers_per_group =  layers_per_group,
                          initial_filter_size = initial_filter_size,
                          max_filter_size = max_filter_size,
                          filter_scaling_factor = filter_scaling_factor,
                          dense_scaling_factor = dense_scaling_factor,
                          kernel_size = kernel_size,
                          activation = activation ,
                          final_activation = final_activation,
                          pooling_layer = pooling_layer,
                          pool_size =  pool_size,
                          batch_norm_rate = batch_norm_rate,
                          dropout_layer_rate = dropout_layer_rate ,
                          dropout_rate = dropout_rate,
                          verbose = verbose)
    
    decoder = {}
    decoder['decoder_inputs'] = encoder['outputs']

    G=-1
    L=-1

    prev_enc_G = 0
    prev_enc_L = 0

    encoder_keys = list(encoder.keys())
    for key in reversed(encoder_keys[1:-1]):

        enc_G = key.split('_')[0].replace('G','')
        enc_L = key.split('_')[1].replace('L','')
        type_ = key.split('_')[2]

        if prev_enc_G!=enc_G:
            G+=1
            L = 0
        elif prev_enc_L!=enc_L:
            L+=1

        prev_enc_G = enc_G
        prev_enc_L = enc_L

        name = 'G{}_L{}_{}'.format(G, L, type_)
        
        if type_ == 'Dense' or type_=='Flatten':
            decoder[name] = _keras.layers.Dense(units = encoder[key].shape[1], 
                                                activation=activation,
                                                name = name)(decoder[list(decoder.keys())[-1]])
        
        elif 'Flatten' in list(decoder.keys())[-1] : #if the last layer was the flat layer
            name = 'G{}_L{}_{}'.format(G, L, 'Reshape')
            decoder[name] = _keras.layers.Reshape(encoder[key].shape[1:])(decoder[list(decoder.keys())[-1]])
            
        elif type_ == 'Conv2D' or type_=='Pool':
            strides = encoder[key].shape[1]//decoder[list(decoder.keys())[-1]].shape[1] #scale up the convolution
            
            if type_ == 'Conv2D':
                decoder[name] = _keras.layers.Conv2DTranspose(filters=encoder[key].shape[-1],
                                                            kernel_size=kernel_size,
                                                            padding='same',
                                                            activation = activation,
                                                              strides = strides,
                                                            name = name)(decoder[list(decoder.keys())[-1]])
            elif type_ == 'Pool':
                decoder[name] = _keras.layers.Conv2DTranspose(filters=encoder[key].shape[-1],
                                                        kernel_size=kernel_size,
                                                        padding='same',
                                                        activation = None, #no activation for reverse pooling
                                                        strides = strides,
                                                        name = name)(decoder[list(decoder.keys())[-1]])
                
    strides = encoder[key].shape[1]//decoder[list(decoder.keys())[-1]].shape[1] #scale up the convolution
    name = 'decoder_outputs'
    decoder[name] = _keras.layers.Conv2DTranspose(filters=encoder['inputs'].shape[-1],
                                                    kernel_size=kernel_size,
                                                    padding='same',
                                                    activation = activation,
                                                    strides = strides,
                                                    name = name)(decoder[list(decoder.keys())[-1]])
            
    return encoder, decoder

def model(img_shape = (256, 256, 3),
              batch_size = None ,
              n_outputs_per_img = 2,
              layers_per_group =  1,
              initial_filter_size = 16,
              max_filter_size = 512,
              filter_scaling_factor = 2,
              dense_scaling_factor = 20,
              kernel_size = _kernel_size,
              activation = 'relu',
              final_activation = 'relu',
              pooling_layer = _pooling_layer,
              pool_size = (2,2),
              batch_norm_rate = None,
              dropout_layer_rate = None,
              dropout_rate = 0.5,
              optimizer = _optimizer,
              loss = _loss,
              verbose = 0):
    
    encoder, decoder = _build_graph(img_shape = img_shape,
                                      batch_size = batch_size ,
                                      n_outputs_per_img = n_outputs_per_img,
                                      layers_per_group =  layers_per_group,
                                      initial_filter_size = initial_filter_size,
                                      max_filter_size = max_filter_size,
                                      filter_scaling_factor = filter_scaling_factor,
                                      dense_scaling_factor = dense_scaling_factor,
                                      kernel_size = _kernel_size,
                                      activation = activation,
                                      final_activation = final_activation,
                                      pooling_layer = _pooling_layer,
                                      pool_size = pool_size,
                                      batch_norm_rate = batch_norm_rate,
                                      dropout_layer_rate = dropout_layer_rate ,
                                      dropout_rate = dropout_rate,
                                      verbose = verbose)

    model = _keras.Model(inputs = encoder['inputs'],
                         outputs = decoder['decoder_outputs'])

    model.compile(optimizer = _optimizer, loss = loss)

    model.encoder = encoder
    model.decoder = decoder
    
    return model
    