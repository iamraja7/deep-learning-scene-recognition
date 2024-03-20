# -------
#     num_classes = 6
#     img_height, img_width = (150, 150)
#     X_input = Input(shape=(img_height, img_width, 3))
#     X = ZeroPadding2D((2, 2))(X_input)
#     X = Conv2D(16, (2, 2), strides=(1, 1), padding='same', name='conv0')(X)
#     X = BatchNormalization(axis=3, name='bn0')(X)
#     X = Activation('relu')(X)
#     X = MaxPooling2D((2, 2), padding='same', name='max_pool0')(X)
    
#     X = Conv2D(32, (2, 2), strides=(1, 1), padding='same', name='conv1')(X)
#     X = BatchNormalization(axis=3, name='bn1')(X)
#     X = Activation('relu')(X)
#     X = MaxPooling2D((2, 2), padding='valid', name='max_pool1')(X)
    
#     X = Conv2D(64, (2, 2), strides=(1, 1), padding='same', name='conv2')(X)
#     X = BatchNormalization(axis=3, name='bn2')(X)
#     X = Activation('relu')(X)
#     X = MaxPooling2D((2, 2), padding='valid', name='max_pool2')(X)
    
#     X = Conv2D(128, (2, 2), strides=(1, 1), padding='same', name='conv3')(X)
#     X = BatchNormalization(axis=3, name='bn3')(X)
#     X = Activation('relu')(X)
#     X = MaxPooling2D((2, 2), padding='valid', name='max_pool3')(X)
    
#     X = Flatten()(X)
#     X = Dense(256)(X)
#     X = BatchNormalization()(X)
#     X = Activation('relu')(X)
    
#     X = Dense(256)(X)
#     X = BatchNormalization()(X)
#     X = Activation('relu')(X)
    
#     X_output = Dense(num_classes, activation='softmax')(X)
    
#     model = Model(inputs=X_input, outputs=X_output)
    
#     return model