import numpy as np 
import cv2
from glob import glob 
from sklearn.utils import shuffle
from sklearn. model_selection import train_test_split 
from patchify import patchify
import tensorflow as tf
import os
import pickle 
from vision_transformer import VisTrans
from preprocessing import load_data,f_dataset
from sklearn.metrics import confusion_matrix
hyper_pam = {}

hyper_pam["image_size"] = 200
hyper_pam["num_channels"] = 3
hyper_pam["patch_size"] = 25
hyper_pam ["num_patches"] = (hyper_pam["image_size"]**2) // (hyper_pam["patch_size"]**2)
hyper_pam["flat_patches_shape"] = (hyper_pam["num_patches"], hyper_pam["patch_size"]*hyper_pam["patch_size"]*hyper_pam["num_channels"])

hyper_pam["batch_size"] = 4
hyper_pam["lr"] = 3e-6
hyper_pam["num_epochs"] = 100 
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
    datapath = "/Users/ayushbhakat/Documents/Neurome/Data"
    model_path = os.path.join("files","model.h5")

    ''' dataset '''
    train_x,valid_x,test_x = load_data(datapath)
    test_ds = f_dataset(test_x[:-2],batch = hyper_pam["batch_size"])
    valid_ds = f_dataset(valid_x,batch = hyper_pam["batch_size"])

    # ''' MODEL '''
    # for test in test_ds:
    #     print(test[1].shape)
    model = tf.keras.models.load_model('transformer_model')
    # model.compile(loss = "categorical_crossentropy",optimizer = tf.keras.optimizers.Adam(hyper_pam["lr"],clipvalue=1.5),metrics = ["acc"])
    predictions = model.predict(test_ds)
    true_predictions = (predictions>0.5).astype(int)
    labels = tf.convert_to_tensor([features[1]for features in test_ds])
    cm = confusion_matrix(labels, true_predictions)
    print("HAVE AT IT BITCHHHHH!!!!!")

    # try:
    #     model.fit(test_ds,epochs = hyper_pam["num_epochs"],validation_data=valid_ds)
    # except KeyboardInterrupt:
    #     model.save("transformer_model")
    #     print("model saved !")
    # else:
    #     model.save("transformer_model")
    #     print("model saved ...")


