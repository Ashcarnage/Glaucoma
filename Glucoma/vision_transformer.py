import tensorflow as tf
from tensorflow import keras
from keras.layers import *
from keras.models import Model 


class ClassToken(Layer):
    def __init__(self):
        super().__init__()

    def build(self,input_shape):
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(initial_value= w_init(shape = (1,1,input_shape[-1])),dtype=tf.float32,trainable=True)
    
    def call(self,inputs):
        batch_size = tf.shape(inputs)[0]
        hidden_dim = self.w.shape[-1]
        cls  = tf.broadcast_to(self.w,[batch_size,1,hidden_dim])
        cls = tf.cast(cls,dtype=inputs.dtype)
        return cls

def mlp(x,config):
    x = Dense(config["mlp_dim"],activation="gelu")(x)
    x = Dropout(config["dropout_rate"])(x)
    x = Dense(config["hidden_dim"])(x)
    x = Dropout(config["dropout_rate"])(x)
    return x

def transformer_ecoder(x,config):
    skip_1 = x
    x = LayerNormalization()(x)
    x = MultiHeadAttention(num_heads=config["num_heads"],key_dim=config["hidden_dim"])(x,x)
    x = Add()([x,skip_1])
    skip_2 = x
    x  = LayerNormalization()(x)
    x = mlp(x,config)
    x = Add()([x,skip_2])
    return x

def VisTrans(config):
    inputs_shape = (config["num_patches"],config["patch_size"]*config["patch_size"]*config["num_channels"])
    inputs = Input(inputs_shape)
    
    """PATCH AND POSITIONAL EMBEDDINGS """
    patch_embed = Dense(config["hidden_dim"])(inputs)
    # Positions
    positions = tf.range(start = 0,limit = config["num_patches"],delta = 1)
    pos_embed = Embedding(input_dim=config["num_patches"],output_dim = config["hidden_dim"])(positions) # (256,768)
    embed = patch_embed + pos_embed #(None, 256, 768)

    """INGECTING CLASSTOKEN"""
    token = ClassToken()(embed) # class token 
    x = Concatenate(axis = 1)([token,embed]) #(None, 257, 768)
    
    """TRANSFORMER ENCODER"""
    for _ in range(config["num_layers"]): # (None, 257, 768)
        x = transformer_ecoder(x ,config)
    
    """CLASSIFICATION HEAD"""
    x = BatchNormalization()(x)
    x = x[:,0,:]
    x = Dropout(0.1)(x)
    x = Dense(2,activation = "softmax")(x)

    model  = Model(inputs, x)
    return model 


if __name__ == "__main__":
    config = {}
    config["num_layers"] = 5
    config["hidden_dim"] = 768
    config["mlp_dim"] = 3072
    config["num_heads"] = 12
    config["dropout_rate"] = 0.1
    config["num_patches"] = 64
    config["patch_size"] = 25
    config["num_channels"] = 3
    model  = VisTrans(config)
    model.summary()



