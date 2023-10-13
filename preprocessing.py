import numpy as np 
import cv2
from glob import glob 
from sklearn.utils import shuffle
from sklearn. model_selection import train_test_split 
from patchify import patchify
import tensorflow as tf
import os
from tensorflow import keras
import pickle 
from keras.optimizers import Adamax
from vision_transformer import VisTrans
from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping
hyper_pam = {}

hyper_pam["image_size"] = 200
hyper_pam["num_channels"] = 3
hyper_pam["patch_size"] = 25
hyper_pam ["num_patches"] = (hyper_pam["image_size"]**2) // (hyper_pam["patch_size"]**2)
hyper_pam["flat_patches_shape"] = (hyper_pam["num_patches"], hyper_pam["patch_size"]*hyper_pam["patch_size"]*hyper_pam["num_channels"])

hyper_pam["batch_size"] = 4
hyper_pam["lr"] = 2e-4
hyper_pam["num_epochs"] = 20
hyper_pam['num_classes'] = 2
hyper_pam["class_names"] = ["GLAUCOMA","NORMAL"]

# config = {}
# config["num_layers"] = 5
# config["hidden_dim"] = 768
# config["mlp_dim"] = 1875
# config["num_heads"] = 14
# config["dropout_rate"] = 0.1
# config["num_patches"] = 64
# config["patch_size"] = 25
# config["num_channels"] = 3

config = {}
config["num_layers"] = 9
config["hidden_dim"] = 650#770
config["mlp_dim"] = 1050#2800
config["num_heads"] = 8#14
config["dropout_rate"] = 0.1
config["num_patches"] = 64
config["patch_size"] = 25
config["num_channels"] = 3

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path) 

def load_data(path,split = 0.1):
    images = shuffle(glob(os.path.join(path,"*","*.png")))

    split_size = float(len(images)*split)
    train_x,valid_x = train_test_split(images , test_size=split , random_state=42)
    train_x,test_x = train_test_split(train_x , test_size=split , random_state=42)
    return train_x,valid_x,test_x

def process_image_label(path):
    path = path.decode()
    ''' reading images '''
    image = cv2.imread(path,cv2.IMREAD_COLOR)
    image  = cv2.resize(image,(hyper_pam["image_size"],hyper_pam["image_size"]))
    # path1 = path.split('/')[-1]
    # print("heyyyyyyooo",path1)
    # cv2.imwrite(f"/Users/ayushbhakat/Documents/Neurome/Glucoma/patched/{path1}",image)
    image  = image/255.0
    # cv2.imwrite(f"/Users/ayushbhakat/Documents/Neurome/Glucoma/patched/{path1}",image)
    
    ''' Preprocessing to patches '''

    patch_shape = (hyper_pam["patch_size"],hyper_pam["patch_size"],hyper_pam["num_channels"])
    patches = patchify(image, patch_shape, hyper_pam["patch_size"])
    # print("69")
    patches = np.reshape(patches,hyper_pam["flat_patches_shape"])
    # cv2.imwrite(f"/Users/ayushbhakat/Documents/Neurome/Glucoma/patched/{path1}",patches)
    patches = patches.astype(np.float32)

    
    ''' Labels '''
    class_name = path.split('/')[-2]
    cls_idx = hyper_pam["class_names"].index(class_name)
    cls_idx = np.array(cls_idx,dtype = np.int32)

    return patches, cls_idx

def parse(path):
    patches,labels = tf.numpy_function(process_image_label,[path],[tf.float32,tf.int32])
    labels = tf.one_hot(labels,hyper_pam["num_classes"])
    patches.set_shape(hyper_pam["flat_patches_shape"])
    labels.set_shape(hyper_pam["num_classes"])
    return patches, labels

def f_dataset(images, batch = 5):
    ds = tf.data.Dataset.from_tensor_slices((images))
    ds = ds.map(parse).batch(batch).prefetch(10)
    return ds

if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)
    # Creating a directory for the data files
    create_dir("files")
    
    ''' Paths '''
    dataset_path = "/Users/ayushbhakat/Documents/Neurome/Data"
    model_path = os.path.join("files","model.h5")
    csv_path = os.path.join("files","log.csv" )
    train_x,valid_x,test_x = load_data(dataset_path)
    # print(f"Train: {np.array(train_x)} - Valid: {len (valid_x)} - Test: {len (test_x)}")

    ''' dataset '''
    train_ds = f_dataset(train_x,batch = hyper_pam["batch_size"])
    l = tf.convert_to_tensor([features[1]for features in train_ds])
    with open ("labels.dat","wb") as f: pickle.dump(l,f)
    valid_ds = f_dataset(valid_x,batch = hyper_pam["batch_size"])

    ''' Model '''
    metrics = [
        tf.metrics.BinaryAccuracy(name='accuracy'),
        tf.metrics.Precision(name='precision'),
        tf.metrics.Recall(name='recall'),
        #tf.metrics.AUC(name='auc')
            ]
    


    model = VisTrans(config)
    model.compile(loss = "categorical_crossentropy",optimizer = Adamax(hyper_pam["lr"],clipvalue=1.0),metrics = metrics) #tf.keras.optimizers.Adam(hyper_pam["lr"],clipvalue=1.0)
    # callbacks = [   
    #                 ModelCheckpoint(model_path, monitor='val_loss', verbose=1, save_best_only=True),
    #                 #ReduceLROnPlateau (monitor='val loss' , factor=0.1, patience=10, min_lr=1e-1),
    #                 CSVLogger (csv_path) ,
    #                 #EarlyStopping (monitor='val_ loss', patience=50, restore_best_weights=False)
    #             ]
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath='transformer_model',  # Path to save the model
        save_best_only=True,             # Save only the best model
        monitor='val_loss',              # Metric to monitor for saving
        mode='min',                      # Minimize the monitored metric
        verbose=1                         # Display progress during training
    )

    
    model.fit(train_ds,epochs = hyper_pam["num_epochs"],validation_data=valid_ds)
    model.save("transformer_model")




