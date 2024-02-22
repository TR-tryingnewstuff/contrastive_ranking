#%%
import pandas as pd 
import tensorflow as tf 
from keras.layers import Dense, Input, Dropout
import numpy as np

import keras
import keras.losses

def   similarity(u, v, temperature):
    return tf.tensordot(u, v, axes=1)
    


def contrastive_loss_reg(u, N, temperature):

    loss = 0
    for i inrange(1, 2*N + 1):
        numerator = tf.exp(sim(u[i-1], u [i], temperature)
        denominator = tf.reduce_sum([tf.exp(sim[i-1],u[k-1], tau)) for k in range(1, 2*N+1) if k != i], axis=0)

    loss += -tf.math.log(numerator / denominator)

    return loss / (2 * N)
    


def mqe_pow(pow=4.):
    def mqe(y_true, y_pred):
    
        return tf.abs(y_true - y_pred)**pow
    
    return mqe

def sign_ae(x, y):

    sign_x = tf.sign(x)
    sign_y = tf.sign(y)
    delta = x - y
    return sign_x * sign_y * tf.abs(delta)
    
    
def linex_loss(delta, a=-1, b=100):

    if a!= 0 and b > 0:
        loss = b * (tf.exp(a * delta) - a * delta - 1)
        return loss
        
        
def linex_loss_val(y_true, y_pred):

    delta = sign_ae(y_true, y_pred)
    res = linex_loss(delta)
    return res



def hlc_nn_model(X_train_shape, drop_rate=0.2, ):
    
    inputs = Input(X_train_shape)

    drop = Dropout(drop_rate)(inputs)
    encode_branch = Dense(128, 'sigmoid', bias_regularizer=tf.keras.regularizers.L1L2(0.01, 0.04))(inputs)
    batch_norm = tf.keras.layers.BatchNormalization(scale=False)(encode_branch)
    drop = Dropout(drop_rate)(batch_norm)
    encode_branch = Dense(8, 'sigmoid')(encode_branch)

    close_out = Dense(1, 'linear', name='close')(encode_branch)
    low_out = Dense(1, 'linear', name='low')(encode_branch)
    high_out = Dense(1, 'linear', name='high')(encode_branch)

    model = tf.keras.Model(inputs=[inputs], outputs=[close_out, low_out, high_out])
    
    
    losses = {
        'close': mqe_pow(3),
        'low': mqe_pow(3),
        'high': linex_loss_val
    }
    
    model.compile(optimizer=tf.keras.optimizers.Nadam(0.001), loss=losses)
    
    return model


def close_nn_model(X_train_shape, drop_rate=0.2, loss_power=3):
    
    inputs = Input(X_train_shape)

    drop = Dropout(drop_rate)(inputs)
    encode_branch = Dense(128, 'sigmoid', bias_regularizer=tf.keras.regularizers.L1L2(0.01, 0.04))(drop)
    batch_norm = tf.keras.layers.BatchNormalization(scale=False)(encode_branch)
    drop = Dropout(drop_rate)(batch_norm)
    encode_branch = Dense(8, 'sigmoid')(drop)

    close_out = Dense(1, 'linear', name='close')(encode_branch)

    model = tf.keras.Model(inputs=[inputs], outputs=[close_out])
    
    
    losses = {
        'close': mqe_pow(loss_power),
    }
    
    model.compile(optimizer=tf.keras.optimizers.Nadam(0.001), loss=losses)
    
    return model
