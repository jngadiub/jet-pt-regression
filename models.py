from keras.models import Model, Sequential, Input
from keras.layers import Dense, BatchNormalization
from keras.layers.merge import concatenate

def dense_model(Inputs):

    #Inputs = Input(shape=X_train_val.shape[1:])
    x = Dense(32, kernel_initializer='normal', activation='relu')(Inputs)
    x = Dense(16, kernel_initializer='normal', activation='relu')(x)
    x = Dense(16, kernel_initializer='normal', activation='relu')(x)
    predictions = Dense(1, kernel_initializer='normal', activation='relu')(x)
    model = Model(inputs=Inputs, outputs=predictions)
    return model

def dense_model_batchnorm(Inputs):

    #Inputs = Input(shape=X_train_val.shape[1:])
    x = BatchNormalization()(Inputs)
    x = Dense(32, kernel_initializer='normal', activation='relu')(x)
    x = Dense(16, kernel_initializer='normal', activation='relu')(x)
    x = Dense(16, kernel_initializer='normal', activation='relu')(x)
    predictions = Dense(1, kernel_initializer='normal', activation='relu')(x)
    model = Model(inputs=Inputs, outputs=predictions)
    return model

def dense_model_batchnorm_2(Inputs):

    #Inputs = Input(shape=X_train_val.shape[1:])
    x = BatchNormalization()(Inputs)
    x = Dense(32, kernel_initializer='normal', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(16, kernel_initializer='normal', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(16, kernel_initializer='normal', activation='relu')(x)
    x = BatchNormalization()(x)
    predictions = Dense(1, kernel_initializer='normal', activation='relu')(x)
    #x = BatchNormalization()(x)
    model = Model(inputs=Inputs, outputs=predictions)
    return model
    
def dense_model_multi_inputs(Inputs1,Inputs2):

    x = Dense(32, kernel_initializer='normal', activation='relu')(Inputs1)
    x = Dense(16, kernel_initializer='normal', activation='relu')(x)
    x = Dense(16, kernel_initializer='normal', activation='relu')(x)
    x = BatchNormalization()(x)
    
    x2 = Dense(32, kernel_initializer='normal', activation='relu')(Inputs2)
    x2 = Dense(16, kernel_initializer='normal', activation='relu')(x)
    x2 = Dense(16, kernel_initializer='normal', activation='relu')(x)
    x2 = BatchNormalization()(x2)
    
    x = concatenate([x, x2])

    x = Dense(64, kernel_initializer='normal', activation='relu')(x)
    x = Dense(64, kernel_initializer='normal', activation='relu')(x)
        
    predictions = Dense(1, kernel_initializer='normal', activation='relu')(x)
    
    model = Model(inputs=[Inputs1,Inputs2], outputs=predictions)
    return model   
