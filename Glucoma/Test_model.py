import numpy as np 
import cv2
from glob import glob 
from sklearn.utils import shuffle
from sklearn. model_selection import train_test_split 
from patchify import patchify
import tensorflow as tf
import os
from vision_transformer import VisTrans
from preprocessing import load_data,f_dataset
hyper_pam = {}

hyper_pam["image_size"] = 300
hyper_pam["num_channels"] = 3
hyper_pam["patch_size"] = 25
hyper_pam ["num_patches"] = (hyper_pam["image_size"]**2) // (hyper_pam["patch_size"]**2)
hyper_pam["flat_patches_shape"] = (hyper_pam["num_patches"], hyper_pam["patch_size"]*hyper_pam["patch_size"]*hyper_pam["num_channels"])

hyper_pam["batch_size"] = 5
hyper_pam["lr"] = 1e-4
hyper_pam["num_epochs"] = 60
hyper_pam['num_classes'] = 2
hyper_pam["class_names"] = ["GLAUCOMA","NORMAL"]

config = {}
config["num_layers"] = 5
config["hidden_dim"] = 768
config["mlp_dim"] = 1875
config["num_heads"] = 15
config["dropout_rate"] = 0.1
config["num_patches"] = 64
config["patch_size"] = 25
config["num_channels"] = 3

if __name__ == "__main__":
    ''' Seeding '''
    np.random.seed(42)
    tf.random.set_seed(42)
    datapath = "/Users/ayushbhakat/Documents/Neurome/data"
    model_path = os.path.join("files","model.h5")

    ''' dataset '''
    train_x,valid_x,test_x = load_data(datapath)
    test_ds = f_dataset(train_x,batch = hyper_pam["batch_size"])

    ''' MODEL '''
    model = tf.keras.models.load_model('transformer_model')
    model.compile(loss = "categorical_crossentropy",optimizer = tf.keras.optimizers.Adam(hyper_pam["lr"],clipvalue=1.0),metrics = ["acc"])
    model.evaluate(test_ds)


