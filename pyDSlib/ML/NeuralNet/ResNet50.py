   
"""ResNet50 model adapted from https://github.com/keras-team/keras-applications/blob/master/keras_applications/resnet50.py.
# Reference:
- [Deep Residual Learning for Image Recognition](
    https://arxiv.org/abs/1512.03385) (CVPR 2016 Best Paper Award)
"""
        
import os as _os

import tensorflow as _tf
_keras = _tf.keras
_layers = _tf.keras.layers
_regularizers = _tf.keras.regularizers

def _identity_block(graph, 
                    kernel_size, 
                    filters1,
                    filters2,
                    filters3,
                    stage, 
                    block,
                    kernel_initializer = 'he_normal',
                    BatchNormMomentum = 0.99,
                    activation = 'relu'):
    
    """The identity block is the block that has no conv layer at shortcut.
    Arguments:
    ----------
        graph: the graph dictionary to whie the block objects will be added
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters1,2,3: integers. the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        kernel_initializer: 'he_normal' was originally used in the resnet paper, but 'glorot_uniform' could also be used
        BatchNormMomentum:  Batch Norm. Momentum for the moving average.
        activation: activation function to be used
    Returns:
    --------
        graph: updated graph dictionary
    """
    bn_axis = 3 #chanels last
    
    shortcut_input_name = list(graph.keys())[-1]
    
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    name = conv_name_base + '2a'
    graph[name] = _layers.Conv2D(filters1, (1, 1),
                      kernel_initializer=kernel_initializer,
                      name=name)(graph[list(graph.keys())[-1]])
    
    name = bn_name_base + '2a'
    graph[name] = _layers.BatchNormalization(axis=bn_axis, 
                                             momentum = BatchNormMomentum,
                                             name=name)(graph[list(graph.keys())[-1]])
    name = bn_name_base + '2a'+activation
    graph[name] = _layers.Activation(activation, name = name)(graph[list(graph.keys())[-1]])
    
    name = name+'_'+activation
    graph[name] = _layers.Conv2D(filters2, kernel_size,
                      padding='same',
                      kernel_initializer=kernel_initializer,
                      name = name)(graph[list(graph.keys())[-1]])
    
    name = bn_name_base + '2b'
    graph[name] = _layers.BatchNormalization(axis=bn_axis, 
                                             momentum = BatchNormMomentum,
                                             name = name)(graph[list(graph.keys())[-1]])
    name = name+'_'+activation
    graph[name] = _layers.Activation(activation, 
                                     name = name)(graph[list(graph.keys())[-1]])
    
    name=conv_name_base + '2c'
    graph[name] = _layers.Conv2D(filters3, (1, 1),
                      kernel_initializer=kernel_initializer,
                      name = name)(graph[list(graph.keys())[-1]])
    name=bn_name_base + '2c'
    graph[name] = _layers.BatchNormalization(axis=bn_axis,
                                             momentum = BatchNormMomentum,
                                             name = name)(graph[list(graph.keys())[-1]])
    
    name = 'shortcut_add'+ str(stage) + block + '_branch'
    graph[name] = _layers.add([graph[list(graph.keys())[-1]], 
                               graph[shortcut_input_name]], 
                              name = name)
    name = name+'_'+activation
    graph[name] = _layers.Activation(activation, name = name)(graph[list(graph.keys())[-1]])
    
    return graph


def _conv_block(graph,
               kernel_size,
               filters1,
                filters2,
                filters3,
               stage,
               block,
               strides=(2, 2),
               kernel_initializer = 'he_normal',
               BatchNormMomentum = 0.99,
               activation = 'relu'):
    """A block that has a conv layer at shortcut.
    Arguments:
    ---------
        img_input: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the first conv layer in the block.
        kernel_initializer: 'he_normal' was originally used in the resnet paper, but 'glorot_uniform' could also be used
    Returns:
    --------
        graph: updated graph dictionary
    Note that from stage 3,
    the first conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
    bn_axis = 3 #chanels last
    
    shortcut_input_name = list(graph.keys())[-1]
    
    #build main branch
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    name = conv_name_base + '2a'
    graph[name] = _layers.Conv2D(filters1, (1, 1), 
                                  strides=strides,
                                  kernel_initializer=kernel_initializer,
                                  name = name)(graph[list(graph.keys())[-1]])
    name = bn_name_base + '2a'
    graph[name] = _layers.BatchNormalization(axis=bn_axis, 
                                   momentum = BatchNormMomentum,
                                   name = name)(graph[list(graph.keys())[-1]])
    name = name+'_'+activation
    graph[name] = _layers.Activation(activation, 
                                     name = name)(graph[list(graph.keys())[-1]])
    
    name=conv_name_base + '2b'
    graph[name] = _layers.Conv2D(filters2, kernel_size, padding='same',
                      kernel_initializer=kernel_initializer,
                      name = name)(graph[list(graph.keys())[-1]])
    name=bn_name_base + '2b'
    graph[name] = _layers.BatchNormalization(axis=bn_axis, 
                                   momentum = BatchNormMomentum,
                                    name = name)(graph[list(graph.keys())[-1]])
    name = name+'_'+activation
    graph[name] = _layers.Activation(activation, name = name)(graph[list(graph.keys())[-1]])

    name=conv_name_base + '2c'
    graph[name] = _layers.Conv2D(filters3, (1, 1),
                      kernel_initializer=kernel_initializer,
                     name = name)(graph[list(graph.keys())[-1]])
    name=bn_name_base + '2c'
    graph[name] = _layers.BatchNormalization(axis=bn_axis, 
                                   momentum = BatchNormMomentum,
                                   name = name)(graph[list(graph.keys())[-1]])
    main_branch_output_name = name
    
    #build shortcut branch
    name = conv_name_base + '1'
    graph[name] = _layers.Conv2D(filters3, (1, 1), strides=strides,
                            kernel_initializer=kernel_initializer,
                              name = name)(graph[shortcut_input_name])
    name = bn_name_base + '1'
    graph[name] = _layers.BatchNormalization( axis=bn_axis, 
                                          momentum = BatchNormMomentum,
                                          name = name)(graph[list(graph.keys())[-1]])
    shortcut_output_name = name
    
    name = 'shortcut_add'+ str(stage) + block + '_branch'
    graph[name] = _layers.add([graph[main_branch_output_name], 
                               graph[shortcut_output_name]],
                             name = name)
    name = name+'_'+activation
    graph[name] = _layers.Activation(activation, name = name)(graph[list(graph.keys())[-1]])
    
    return graph

def _graph(img_shape = (None, None, 3),
           weights=None,#'imagenet',
           initial_filters = 64,
           initial_kernel_size = (7,7),
           initial_strides = (2,2),
           kernel_size = 3,
           kernel_initializer = 'he_normal',
           BatchNormMomentum = 0.99,
           activation = 'relu',
           include_top=True,
           GlobalPooling=None,
           n_classes=1000,
          ):
    
    """Instantiates the ResNet50 architecture.
    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.
    
    Arguments:
    ---------
        img_shape: tuple. (x, y, color) shape of an image
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        img_input: optional Keras tensor (i.e. output of `_layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 32.
            E.g. `(200, 200, 3)` would be one valid value.
        GlobalPooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional block.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional block, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    
    valid_weights = [None, 'imagenet']
    assert(weights in valid_weights), 'Invalid weights argument ('+str(weights)+') specified. Valid weigths include: '+str(valid_weights)
    
    _keras.backend.clear_session()
    
    graph = {}
    
    if weights == 'imagenet': #Use tensorflows ResNet50V2 model
        pretrained_model = _tf.keras.applications.ResNet50(input_shape=img_shape,
                                                               include_top=include_top,
                                                               weights=weights)
        
        name = 'img_input'
        graph[name] = pretrained_model.input
        
        name = pretrained_model.layers[-1].name
        graph[name] = pretrained_model.output
            
        stage = 5 #ResNet50V2 has 5 resnet blocks (stages)
        
            
    else: #Build the model from scratch
    
        bn_axis = 3

        name = 'img_input'
        graph[name] = _layers.Input(shape=img_shape)

        name='conv1_pad'
        graph[name] = _layers.ZeroPadding2D(padding=(3, 3),
                                            name = name)(graph[list(graph.keys())[-1]])
        name='conv1'
        graph[name] = _layers.Conv2D(filters = initial_filters, 
                                      kernel_size = initial_kernel_size,
                                      strides=initial_strides,
                                      padding='valid',
                                      kernel_initializer=kernel_initializer,
                                      name = name)(graph[list(graph.keys())[-1]])
        name='bn_conv1'
        graph[name] = _layers.BatchNormalization(axis=bn_axis, 
                                                 momentum = BatchNormMomentum,
                                                 name = name)(graph[list(graph.keys())[-1]])
        name = name+'_'+activation
        graph[name] = _layers.Activation(activation, 
                                        name = name)(graph[list(graph.keys())[-1]])

        name='pool1_pad'
        graph[name] = _layers.ZeroPadding2D(padding=(1, 1),
                                           name = name)(graph[list(graph.keys())[-1]])
        name = 'pool1_MaxPooling2D'
        graph[name] = _layers.MaxPooling2D((3, 3), 
                                          strides=(2, 2),
                                          name = name)(graph[list(graph.keys())[-1]])
        filters = initial_filters
        for stage in [2,3,4, 5]:

            #build conv block for stage
            graph = _conv_block(graph, 
                                kernel_size = kernel_size,
                                filters1 = filters,
                                filters2 = filters,
                                filters3 = filters*4,
                                stage = stage,
                                block = 'a',
                                strides=(1, 1),
                                kernel_initializer = kernel_initializer,
                                BatchNormMomentum = BatchNormMomentum,
                                activation = activation)

            #fetch list of identity block IDs for given stage
            if stage==2:
                blocks = ['b','c']
            elif stage == 3:
                blocks = ['b','c','d']
            elif stage == 4:
                blocks = ['b','c','d','e','f']
            elif stage == 5:
                blocks = ['b','c']

            for block in blocks:
                graph = _identity_block(graph, 
                                        kernel_size = kernel_size,
                                        filters1 = filters,
                                        filters2 = filters,
                                        filters3 = filters*4,
                                        stage=stage, 
                                        block=block,
                                        kernel_initializer = kernel_initializer,
                                        BatchNormMomentum = BatchNormMomentum,
                                        activation = activation)
            #update filters for next iteration
            filters = filters*2

    stage +=1
    if include_top and weights == None:
        name= 'GlobalAveragePooling2D_'+str(stage)
        graph[name] = _layers.GlobalAveragePooling2D(name=name)(graph[list(graph.keys())[-1]])
        name = 'Dense_Softmax_'+str(stage)
        graph[name] = _layers.Dense(n_classes, 
                                   activation='softmax', 
                                   name= name)(graph[list(graph.keys())[-1]])
    elif include_top==False:
        if GlobalPooling == 'avg':
            name= 'GlobalAveragePooling2D_'+str(stage)
            graph[name] = _layers.GlobalAveragePooling2D(name = name)(graph[list(graph.keys())[-1]])
        elif GlobalPooling == 'max':
            name = 'GlobalMaxPooling2D_'+str(stage)
            graph[name] = _layers.GlobalMaxPooling2D(name = name)(graph[list(graph.keys())[-1]])
            
    return graph
    

def model(img_shape = (None, None, 3) ,
          include_top=True,
             weights=None,
             GlobalPooling =None,
             n_classes=1000,
             kernel_initializer = 'he_normal',
             BatchNormMomentum = 0.99,
             activation = 'relu',
             loss = None,
             optimizer = _keras.optimizers.Adam(),
             metrics = None
         ):
    
    """Instantiates the ResNet50 architecture.
    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.
    
    Arguments:
    ---------
        img_shape: shape of the image to be passed. If unknown at instantiation, pass (None, None, 3)
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        GlobalPooling : Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional block.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional block, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        n_classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
        loss: keras loss function
    Returns
    -------
        A compiled Keras model instance.
    
    """
    
    graph = _graph(img_shape = img_shape,
                   weights=weights,
                   initial_filters = 64,
                   initial_kernel_size = (7,7),
                   initial_strides = (2,2),
                   kernel_size = 3,
                   kernel_initializer = kernel_initializer,
                   BatchNormMomentum = BatchNormMomentum,
                   activation = activation,
                   include_top=include_top,
                   GlobalPooling=GlobalPooling,
                   n_classes=n_classes,
                  )
    
    # Create model.
    model = _keras.models.Model(graph[list(graph.keys())[0]], 
                         graph[list(graph.keys())[-1]], 
                         name='resnet50')

    try:
        model.compile(loss=loss,
                      optimizer=optimizer,
                      metrics=metrics)
    except exception as e:
        display(model.summary())
        raise e

    return model
    
    