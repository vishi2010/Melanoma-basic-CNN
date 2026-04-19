import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import os


IMAGE_FOLDER = "ENTER_YOUR_DATASET_LOCATION"
CSV_FILE = "ENTER_YOUR_DATASET_KEY_LOCATION"

# the dimensions or something 

IMAGE_HEIHT = 225 
IMAGE_WIDTH = 225
EPOCHS = 50 # how many repitions of the data set it does. for example every epoch it will change somthing slightly so it doesn't get used to the images. for example it will flip it. or it will slightl angle the photo, etc. 
BATCH_SIZE = 32 # this is basically how many photos it processes before changing like the weights or the. 32 is the most common since it uses minimal memory but isn't too slow

VALIDATION_SPLIT = 0.15 # the validation split is how much of the data the computer learns off of. it basically learns off 15 percent. 
TEST_SPLIT = 0.15  # ok so this is a bit confusing since how is 0.15 for the test and then 0.15 percent eqaull all of the photos. It doesn't, test split sets aside 15% of the data for testing for the final test to see how much you have learned.  
LEARNING_RATE = 0.0001 # this learning rate is really really slow, for example lets say that the weight of the neuron was 0.5 and the prediction was 2, and the actaul number was 5 the error is 5 - 1 = 4 and then you do error^2 to find the gradient which is 16. 16 is what you multiple the learning rate by to get the new weight. this is too stoop if from guessing too high and then too low and into a loop basically. 

#loading the data 

print("loadind data")

df = pd.read_csv(CSV_FILE) # pretty self explanatory, you make a df or a dataframe that loads everythign into a nice collumn using pandas which we imported as pd. we read the csv file which we set as CSV_FILE above
print (f"\nTotal image: {len(df)}")
print (f"collumns: {df.columns.tolist()}") #doesn't make sense at all but somehow still works just don't mess with it
print (df.head())

#count target distribution
print(df['target'].value_counts) #it counts how manny times each unique value appears in the 'target' collumn. for our melanoma dataset, it shows 32,611 benign cases  (0) and 515 melanoma cases (1) 
melanoma_count = df['target'].sum() # why does this line have no  == 0 but the one underneath it does? this is because we can simply add up all the ones in the dataset to get this number since melanoma is labeled 1 and benign is labeled 0 we can just do 1 + 0 + 0 + 0 + 1 + 0 and it would only count the melalonmas since the benign doesn't have a value.
benign_count =  (df['target'] == 0).sum()  # this just gets the value of the benign. We need the part that says == 0 since if we just did .sum() like the one above we would just get 0. since the value of benign is 0 it would simply do 0 + 0 + 0 to get the result instead of doing how many 0s there are. ***important**** we DO NOT do len(df['target] == 0 ) the reason we don't do this is because lenght counts how many items are in the list not how many true items are in the list. sum() asks "whats the total when I add these up" basically len() counts the size of the container, sum() counts how many trues are in it. 
print(f"\n Benign(0): {benign_count}") # self explanotory just prints in the terminal 
print(f"\n malignant(1): {melanoma_count}")  # self explanotory just prints in the terminal 
print(f"class imbalance: {melanoma_count / len(df) *100:.2f}% melanoma") #lil bit of math going on here just decipher it step by step and you should get it. we pull the melanoma_count from above and then divides it by the lenght of the data, here we may use len() since it is getting the total number. we time it by 100 for a percent and then a bit of weird python where we basically say with the colon (:) this is how we want to format it where we put the .2f which means put 2 decimal point here after so that 1.023498729347 becomes 1.02

#organization 
df['filepath'] = df["image_name"].apply(lambda x: os.path.join(IMAGE_FOLDER, x + '.jpg')) # we use lambda x: which might be a bit confusing here so lemme break it down a bit. its saying hey from now on df['filepath] is just gonna be df["IMAGE_FOLDER"] with all our photos but just apply (.apply) this function which is lambda x: which is essentially just a mini function where it says just add a .jpg at the end of all the photo names so its easier to use later on. 
print("VERIFYING IF FILE EXISTS") #we are going to check wether the file we are sorting through exists or not
df['exists'] = df["filepath"].apply(os.path.exists) # what python is sees is basically a for loop. the .apply basically makes a mini for loop execpt its just less code. bellow you can see what python sees and the codes "equivalant" the df[exists] is basically creating a new collumn that either says true or false depending on wether the file was found or not 
# results = []
# for filepath in df['filepath']:
#   results = os.path.exists(filepath)
#   results.append(results)
# df['exists'] = results
print(f"files found: {df['exists'].sum()} / {len(df)}") # this line is pretty simple if you just pay attention. it uses the .sum() of the collumn 'exists' this is since in python True = 1, False = 0 which is why we can use sum since if we count the number it also gives us the number of trues since if you add 0 it doesn't affect the sum. 

if df['exists'].sum() == 0:     #self explanatory 
    print("no files found its the wrong directory i think")  #self explanatory 
    exit()      #self explanatory 
    
df = df[df['exists']].copy() # the inner df['exists'] accesses the collumn that checks if the image exists. we do another df to create another dataframe with only the true cases. if you don't understand this ask claude it useses a good analogy/explanation
df = df.drop(collumns=['exists']) #we essntially overwrite the variable df from the last line. df now points to a dataframe that doesn't have the images that don't have a filepath/don't exist. and it removes the collumn exists to remove clutter

# 3. stratified data, train/val/test split/

print("SPLITTING DATA (STRATIFIED)")

train_val_df, test_df = train_test_split(     # a lil bit of jargon here so ill go in depth on this line. this is called something called the tran_val_df, test_df = train_Test_split this is called tuple unpacking in python. we got the function from sklearn.model_selection. if you go into imports you would see it as from sklearn.model_selection import train_test_split. this function splits one dataframe into 2
    df,                              #you're basically saying to the dataframe "hey, here's the data i want you to split"
    test_size=TEST_SPLIT,           #we set the variable at the top for this at 0.15 we basically 
    stratify=df['target'],          #first of all we use stratify because if you do a random split, irregularities could occur. for example lets say on the first split you got 95% benign and 5% malignant thats way too many benign cases and for the test case that would bean that there would be 70% benign and 30% melanoma thats way too many malignant cases. we are telling the code with this line that "look at the target of 1s 0s and make sure both the test and train splits have the same percentage of each type"
    random_state=42 #this is to make sure that nn doesn't memorize the order of the cases and its random but reproducible.
)
train_df, val_df = train_test_split (
    train_val_df,           # its the same thing that is happening above except that instead of splitting the whole dataset your only splitting the training dataset which is the dataset that your model learns from
    test_size=VALIDATION_SPLIT / (1 - TEST_SPLIT),  # if you look closely you will realise that this is pretty simple to unerstand. we do this code to make sure we grab 150 photos and no less. so basically we are doing 0.15/(1-0.15) = 0.176 which in percent form is 17.6%. that means that if you take 17.6% of 850 images you will have 150. 
    stratify=train_val_df['target'],        # we do the same thing with stratify to ensure that we are getting an even ratio of the benign vs malignant cases
    random_state=42 # for randomization 
    
)

print(f"train set: {len(train_df)} images ({train_df['target'].sum()} melanoma)") 
print(f"train set: {len(val_df)} images ({val_df['target'].sum()} melanoma)")
print(f"train set: {len(test_df)} images ({test_df['target'].sum()} melanoma)")

#4. computing class weight 
print("COMPUTING CLASS WEIGHTS")

class_weights = compute_class_weight(                    # we get this function from the sklearn.utils.class_weights import in line 5.          
    class_weight='balanced',                            # this is telling the model to balance out the weights so that penalty of missing a melanoma is bad enough for it to care about learning it. the math it does is: total # of cases / (# of classes / # of cases in that class) that means that for benign in that scenario is: 850 / (2 * 765) = 0.556. this means that each benign is worth that many "points"
    classes=np.unique(train_df['target']),              #check how many classes there are by using the .unique method "your stupid self is prolly wondering can't i just put 2 since its either benign or malignant" NO NO NO NO NO YOU MAY NOT PUT JUST 2 that's cause it expects an array like [0,1] not just 2 thats why the np.unique(train_df['target]) gives us an array. if you wanted tho you could hardcode the array of [0,1] or np.array([0,1])
    y=train_df['target']                                 # we basically pass the whole array of 1s and 0s in this one function. we use y since its machine learning convention to use y for labels/targets so y is just a common variable name for what your trying to predict
)

class_weights_dict = {0: class_weights[0], 1: class_weights[1]}         # we simply put the list of weights into a dictionary here before it was just class_weights = [0.556, 5.0] after it bacame class_weights_dict = {0: 0.556, 1: 5} we did this because later on the model's .fit() function wants a dictionary not a list

# 5.  DATA AUGMENTATION
print("SETTING UP DATA AUGMENTATION")

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(    # ImageDataGenerator is a tool that loads and modifies images for training think of it as a image procccessing machine that loads images from files, transforms them (rotate, flip, zoom, etc) feeds them to your model
    rescale=1./255,    # Neural networks work better with smaller numbers so we divide every pixel's value by 255 so thats between 0 to 1. 
    rotation_range=20,  # randomaly rotates the image up to 20 degrees. this basically teaches them to look for moles regardless of orientaion 
    width_shift_range=0.2, # the moles are usaully not in the center of photos which is what we are recreating with this. where we basically move the image left or right by 20% of the width. 
    height_shift_range=0.2, # the same as the width shift range except instead of left or right it is up or down by 20% of its height.
    horizontal_flip=True, # this will randomly flip the photo horizontally for a mirror image where if a mole is on the left side it appearas on the right after the flip. 
    zoom_range=0.2, # this zooms in and out of the photo by 20% the range of the zoom would be anywhere from 0.8x to 1.2x  to simulate the height of which the camera is held when the photo is taken 
    shear_range=0.2, # the 0.2 is in radians which is about 11.5 degrees. sheering means at an angle. it is like the photo was put into italics 
    fill_mode='nearest' # it fills in empty space with the nearest ex isting pixel's color
)

val_test_datagen = tf.keras.preprocessing.image.ImageDataGenerator( #this is a simple tool from tensorflow that loads images can modify them. 
    rescale=1./255 #rescales the images by dividing the pixel values by 255
)


#6. creating data generators 

print("creating data generators..")

train_generator = train_datagen.flow_from_dataframe(        #this line is like a conveyer belt it transports the training images 
    dataframe=train_df,        # this is where the "conveyer belt" gets the images from the 
    x_col='filepath',          # this tells the generator the input (images) are located at the paths in the 'filepath'
    y_col="target",            # tells the generator: the outputs are in the 'target' column
    target_size=(IMAGE_HEIHT, IMAGE_WIDTH),     #this resizes every photo to 224*224 since some photos are different sizes than other such as 640*640 or 1920*1080
    batch_size= BATCH_SIZE,    # this is how many photos are placed on the conveyer belt at a time. 
    class_mode='binary',        # this is how many categories there are 
    shuffle=True,       #this shuffles the images order so it doesn't memorize the orders of the benigns and malignant after each epoch
    seed=42
)


val_generator = train_datagen.flow_from_dataframe(        #this line is like a conveyer belt it transports the validation split images 
    dataframe=val_df,           # this is where the "conveyer belt" gets the images from the 
    x_col='filepath',           # this tells the generator the input (images) are located at the paths in the 'filepath'
    y_col="target",             # tells the generator: the outputs are in the 'target' column
    target_size=(IMAGE_HEIHT, IMAGE_WIDTH),     #this resizes every photo to 224*224 since some photos are different sizes than other such as 640*640 or 1920*1080
    batch_size= BATCH_SIZE,         # this is how many photos are placed on the conveyer belt at a time. 
    class_mode='binary',    # this is how many categories there are 
    shuffle=False       # we want this false since we want reproducuble images. like for example lets say on the first epoch it did 70% and then second epoch it did 82 % third it got to 75%. now we have to understand wether the second iteration it just had easier questions or if actaully did learn better. 
)


test_generator = train_datagen.flow_from_dataframe(        #this line is like a conveyer belt it transports the test split images to the nn 
    dataframe=test_df,              # this is where the "conveyer belt" gets the images from the 
    x_col='filepath',             # this tells the generator the input (images) are located at the paths in the 'filepath'
    y_col="target",                  # tells the generator: the outputs are in the 'target' column
    target_size=(IMAGE_HEIHT, IMAGE_WIDTH),     #this resizes every photo to 224*224 since some photos are different sizes than other such as 640*640 or 1920*1080
    batch_size= BATCH_SIZE,                 # this is how many photos are placed on the conveyer belt at a time. 
    class_mode='binary',    # this is how many categories there are 
    shuffle=False       # we want this false since we want reproducuble images. like for example lets say on the first epoch it did 70% and then second epoch it did 82 % third it got to 75%. now we have to understand wether the second iteration it just had easier questions or if actaully did learn better. 
)

# how are val_df and test_df different if they are both evulauting the nn. the val_df is used after every epoch. but for test_df it is similar to the final exam after the whole 50 epoch is done. 

print("Data generators ready!")

# 7. build CNN model (the meat of the NN)

print("CNN model building...")

model = models.Sequential([
    #conv block 1
    layers.Conv2D(32, (3,3), activation='relu'), # the conv is a convulution the 2D means it works on 2d images(height * width) the first paramter is 32. what does that signify? it basically means it is looking for 32 different patteren for example one looking for the horizontal line, the other looks for curves, etc. the 3,3 means that the images slides across a 3*3 plane, don't confuse it with compressing the images it looks for patterns in these 3x3 areas it is just for simplicity for the computer this 3x3 area does not compress the iamge whatsoever. the last part of this line is the activation which says relu. this part is a bit more complicated, ReLU started for rectified linear unit.  it removes negative values and squishes values into a range. we didn't not specify any range so it doesn't do that. it is fast to compute and helps learn complex curves and patterns. 
    layers.BatchNormalization(), #after the conv2D layer you have 32 feature maps which might wildly different values. it would change the values that would center 0 with similar spread like [-1.2, -0.8, -04, -0.2, 0.3, 0.9. 1.1]. how does it do this? it calculates the mean for example lets the feature map is [100, 200, 300, 400, 500] you would fine the meadian which would be 300. measure how spread apart the valuels are which is symbolised as std = 141. step 3, normalize(subtract mean, divide by std) 100 -> (100-3000) / 141 = -1.4. this would be eqaul to the new value which would replace the value of 100. 
    layers.MaxPooling2D((2,2)), # maxpooling takes a 2x2 sqaure and floats it across the image which would be 224x224. it would go in groups on 4 pixels and will take the highest number in that 2x2 area. Why do we choose the highest value in the sqaure. this is because we want the strongest feature, since low value = weak or no feature. think of the low values like adlibs to a song and the high value is like the actual vocals. 
    layers.Dropout(0.25), # this part is a bit weird. dropout randomly turns off 25% of neurons during training to prevent memerization. an example is if the following list went through [10, 20, 30, 40, 50, 60, 70,] it would go through this line and become [10, 0,  30, 40, 0, 60, 0, 80] to make it so that every neuron is being used in the process. if we don't use dropout it just says for img1 use neuron 123 for img2 use neuron 234
    
    
    #conv block 2
    layers.Conv2D(64, (3,3), activation='relu'), # the conv is a convulution the 2D means it works on 2d images(height * width) the first paramter is 32. what does that signify? it basically means it is looking for 32 different patteren for example one looking for the horizontal line, the other looks for curves, etc. the 3,3 means that the images slides across a 3*3 plane, don't confuse it with compressing the images it looks for patterns in these 3x3 areas it is just for simplicity for the computer this 3x3 area does not compress the iamge whatsoever. the last part of this line is the activation which says relu. this part is a bit more complicated, ReLU started for rectified linear unit.  it removes negative values and squishes values into a range. we didn't not specify any range so it doesn't do that. it is fast to compute and helps learn complex curves and patterns. 
    layers.BatchNormalization(), #after the conv2D layer you have 32 feature maps which might wildly different values. it would change the values that would center 0 with similar spread like [-1.2, -0.8, -04, -0.2, 0.3, 0.9. 1.1]. how does it do this? it calculates the mean for example lets the feature map is [100, 200, 300, 400, 500] you would fine the meadian which would be 300. measure how spread apart the valuels are which is symbolised as std = 141. step 3, normalize(subtract mean, divide by std) 100 -> (100-3000) / 141 = -1.4. this would be eqaul to the new value which would replace the value of 100. 
    layers.MaxPooling2D((2,2)), # maxpooling takes a 2x2 sqaure and floats it across the image which would be 224x224. it would go in groups on 4 pixels and will take the highest number in that 2x2 area. Why do we choose the highest value in the sqaure. this is because we want the strongest feature, since low value = weak or no feature. think of the low values like adlibs to a song and the high value is like the actual vocals. 
    layers.Dropout(0.25), # this part is a bit weird. dropout randomly turns off 25% of neurons during training to prevent memerization. an example is if the following list went through [10, 20, 30, 40, 50, 60, 70,] it would go through this line and become [10, 0,  30, 40, 0, 60, 0, 80] to make it so that every neuron is being used in the process. if we don't use dropout it just says for img1 use neuron 123 for img2 use neuron 234
    
    #conv block 3
    layers.Conv2D(128, (3,3), activation='relu'), # the conv is a convulution the 2D means it works on 2d images(height * width) the first paramter is 32. what does that signify? it basically means it is looking for 32 different patteren for example one looking for the horizontal line, the other looks for curves, etc. the 3,3 means that the images slides across a 3*3 plane, don't confuse it with compressing the images it looks for patterns in these 3x3 areas it is just for simplicity for the computer this 3x3 area does not compress the iamge whatsoever. the last part of this line is the activation which says relu. this part is a bit more complicated, ReLU started for rectified linear unit.  it removes negative values and squishes values into a range. we didn't not specify any range so it doesn't do that. it is fast to compute and helps learn complex curves and patterns. 
    layers.BatchNormalization(), #after the conv2D layer you have 32 feature maps which might wildly different values. it would change the values that would center 0 with similar spread like [-1.2, -0.8, -04, -0.2, 0.3, 0.9. 1.1]. how does it do this? it calculates the mean for example lets the feature map is [100, 200, 300, 400, 500] you would fine the meadian which would be 300. measure how spread apart the valuels are which is symbolised as std = 141. step 3, normalize(subtract mean, divide by std) 100 -> (100-3000) / 141 = -1.4. this would be eqaul to the new value which would replace the value of 100. 
    layers.MaxPooling2D((2,2)), # maxpooling takes a 2x2 sqaure and floats it across the image which would be 224x224. it would go in groups on 4 pixels and will take the highest number in that 2x2 area. Why do we choose the highest value in the sqaure. this is because we want the strongest feature, since low value = weak or no feature. think of the low values like adlibs to a song and the high value is like the actual vocals. 
    layers.Dropout(0.25), # this part is a bit weird. dropout randomly turns off 25% of neurons during training to prevent memerization. an example is if the following list went through [10, 20, 30, 40, 50, 60, 70,] it would go through this line and become [10, 0,  30, 40, 0, 60, 0, 80] to make it so that every neuron is being used in the process. if we don't use dropout it just says for img1 use neuron 123 for img2 use neuron 234
    
    #conv block 4
    layers.Conv2D(256, (3,3), activation='relu'), # the conv is a convulution the 2D means it works on 2d images(height * width) the first paramter is 32. what does that signify? it basically means it is looking for 32 different patteren for example one looking for the horizontal line, the other looks for curves, etc. the 3,3 means that the images slides across a 3*3 plane, don't confuse it with compressing the images it looks for patterns in these 3x3 areas it is just for simplicity for the computer this 3x3 area does not compress the iamge whatsoever. the last part of this line is the activation which says relu. this part is a bit more complicated, ReLU started for rectified linear unit.  it removes negative values and squishes values into a range. we didn't not specify any range so it doesn't do that. it is fast to compute and helps learn complex curves and patterns. 
    layers.BatchNormalization(), #after the conv2D layer you have 32 feature maps which might wildly different values. it would change the values that would center 0 with similar spread like [-1.2, -0.8, -04, -0.2, 0.3, 0.9. 1.1]. how does it do this? it calculates the mean for example lets the feature map is [100, 200, 300, 400, 500] you would fine the meadian which would be 300. measure how spread apart the valuels are which is symbolised as std = 141. step 3, normalize(subtract mean, divide by std) 100 -> (100-3000) / 141 = -1.4. this would be eqaul to the new value which would replace the value of 100. 
    layers.MaxPooling2D((2,2)), # maxpooling takes a 2x2 sqaure and floats it across the image which would be 224x224. it would go in groups on 4 pixels and will take the highest number in that 2x2 area. Why do we choose the highest value in the sqaure. this is because we want the strongest feature, since low value = weak or no feature. think of the low values like adlibs to a song and the high value is like the actual vocals. 
    layers.Dropout(0.25), # this part is a bit weird. dropout randomly turns off 25% of neurons during training to prevent memerization. an example is if the following list went through [10, 20, 30, 40, 50, 60, 70,] it would go through this line and become [10, 0,  30, 40, 0, 60, 0, 80] to make it so that every neuron is being used in the process. if we don't use dropout it just says for img1 use neuron 123 for img2 use neuron 234
    
    
    #flatten and dense layers 
    layers.Flatten(), # conv2d needs like a 3D grid of numbers into a 1D list. imagine a bunch of 13x13 numbers in a 256 thick stack that would eqaul into 43,264 numbers, for the operation "dense" to work on the next line all of these numbers need to be in a list. flatten essentially does this by putting all 40 thousand numbers into a list. iomagine flatten is the bridge between the convultion and the densing. 
    layers.Dense(512, activation='relu'), # the 512 means 512 neurons every input cojnnects to every output. lets take a single neuron it would do the following calculations: (input[0] * weight[0]) + (input[1] * weight[1]) + ... + (input[43263] * weight[43263]) + bias. then we apply ReLu if negative = 0 if positive -> keep it. result is one output value for neuron 1. in summary the dense layer combines everything that the convulation layer found fore xample if dense layer combines them and says: if i see irregular edges + dark color + specific texture" that might be melanoma.
    layers.BatchNormalization(), # batch normalization sees the different degrees of numbers for each list then finds the median of the set then subtracts the number from the median then divide by the standard deviation of the numbers. 
    layers.Dropout(0.5),   # this drops 50% of the neurons randomly so they don't memorize the order and the features of each case in the dataset
    layers.Dense(128, activation='relu'), #we are doing the same thing as two lines above but instead of 512 neurons we only use 128. 
    layers.dropout(0.5), #it drops 50% of the neurosn again so it doesn' memorize. 
    
    #output layer
    layers.Dense(1, activation='sigmoid') # this squishes down any number into 0-1. if the number is greater than 0.5 it is melanoma if it is less than it is 0.5
    
])

# 8. compile model 

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),#the optimizer just adjutss knowledge after each mistake this updates the weights to make the model better. the adam stands for adaptive movement estimation adam is the industry standard
    loss="binary_crossentropy", # how to measure mistakes. binary cross entropy or BCE is used when there are only two possible outcomes such as spam or not spam, true or false. you model outputs a probability: loss = -[actaul * log(prediction)]
    metrics=[           # this checks what grades to track. 
        'accuracy',  # accuracy just says "ok 100 cases 85 right 15 wrong that means 85%" that is not great though since model can just guess benign due to class inbalance that is why the rest of the metrics are useful/needed       
        keras.metrics.Precision(name='Precision'),   # this calculates the precision of each melanoma lets say for example each time model guesses melanoma its usaully wrong we want to change that.
        keras.metrics.Recall(name='recall'), #out of all actaull melanoma cases how many did the model catch? this is the most important metric for this since missing a melanoma case could kill someone
        keras.metrics.AUC(name='auc') #auc stands for area under curve this measures how well the models separetess benign from melanoma 0.8-1.0 is great
    ]
)

model.summary()
# 9. CALLBACKS

callbacks = [
    keras.callbacks.ModelCheckpoint(
        'best_melanoma_model.keras',
        monitor='val_auc', # these are basically checkpoints on what to see during training like if the progession is not bettering during training that means that the model is not learning. 
        mode='max',
        save_best_only=True, # this part of the code checks for which time frame the model was during the best at. if at epoch 10 the model was at 0.85 percent accuracy and then got worse at the end of it like epoch 50 when it dipped down to 0.67 percen you would still keep the best model. 
        verbose=1 # shows training progress after each epoch.
        
    ),
    keras.callbacks.ReduceLROnPlateau(          #automatically reduces the learning rate when the models rate of learning plateus (stops improving). seems counterintuitive since learning slowly is better than not learning at all but in the later stages it might cause problems such as bouncing around too much. this basically cuts down the learning rate like right now its at 0.0001 but after this it will be 0.00005. 
        monitor='val_auc',          #this the data the thing monitors to see the progress 
        mode='max',   # this is telling the function that we want the number to get higher we are not trying to minimize the loss instead we are trying to improve the learning 
        factor=0.5,     # this how much we reduce it by so new_lr = 0.5 * 0.0001 and after reduction it becomes 0.00005
        patience=5,     # this says wait 5 epochs before implementing the factor to reduce the learning rate
        min_lr=1e-7,       # this is the lowest learning rate that it may go to. 1e - 7 is scientific notation for 0.0000001
        verbose=1           # training visaul on how long it might take
     
        
    ),
    
    keras.callbacks.EarlyStopping(          # this stops the training all together if the model stops improving. the problem which we combat with this is overfitting. the model is trying to memorize the data instead of learning patterns
    monitor='val_auc',          # this the thing we monitor and see whe
    mode='max',                 # this is max becasue we want the val_auc to increase. if we wanted it to decrease we would use min
    patience=10,            # this is how many epochs we wait before using this. 10 is the sweetspot since anything lower might miss improvement over time and something too big like 30 might waste time on a dying model
    restore_best_weights=True,      #if the model does better with certain weights even if it was a couple epochs back it will still use that one.
    verbose=1               # progress check
        
    )
]


#10. training the model

print("STARTING TRAINING")

history = model.fit(            #this is where the actaull training happens.
    train_generator,            #this is telling the function where to get the data from
    steps_per_epoch=len(train_df) // BATCH_SIZE,        # this is telling it how many batches = one epoch. so we reverse it to by doing how many numbers in total are there? and then dividing that by the batch size. 
    epochs=EPOCHS,          #  how many epochs or times of the same dataset we need to use to train the model.
    validation_data=val_generator,          #this is telling it to where the validation data is.
    validation_steps=len(val_df) // BATCH_SIZE,         # this is doing again how many batches for validation 
    class_weights= class_weights,           # this is telling it that the class weights is found in the variable class weights. this is why we had to previously change the class weights from a list to a dictionary since in this line it requires it to be in a dictionary format
    callbacks= callbacks,   # same thing as the rest
    verbose= 1,     # progress check
    
)

# 11. evaluate on test set 

print("Evaluating on test set")

model.load_weights("best_melanoma_model.keras")         # this loads the best model which might be from epoch 20 or epoch 50. This just depends on which epoch had the highest val_auc

test_generator.reset()      #this resets the generator to start from the beginning. this is important because without rest the generator might be at iamge #75 and predictions woud start from image #75with rest the genrator goes back to iamge #0
y_pred_probs = model.predict(test_generator, verbose=1)       #this just gets the models probability predictiosn for all test images.   
y_pred = (y_pred_probs > 0.5).astype(int).flatten()     # this line looks all special and stuff but it just converts probability to binary prediction. lets break it down further. the y_pred_probs > 0.5 compares each porbability to 0.5 threshold so it does [0.23, 0.6, 0.1] into [False, True, False]. the astype(int) converts true/false to 1/0 so from the previous list [false,true, false] it becomes [0, 1, 0] this just makes sure its a 1D array and nothing weird is going on. lets do a complete example to drill this home. y_pred_probs = [0.43, 0.54, 0.59, 0.47, 0.88] after step 2 which is the >0.5 it becomes into [false, true, true, false, true] then we use the astype(int) to make into  [ 0, 1, 1, 0, 1]
y_true = test_df['target'].values           # this gets the true labels 

print("Classification report")
print(classification_report(y_true, y_pred, target_names=['Benign', 'Melanoma'])) # this gives us the detailed breakdown of the performance so the F1 score , accuracy, macro avg, weighted avg, precision, recalll, and support. the precision means out of predictiosn for this class how many were correct. recall means out of the actuall cases of this class how many did we catch?. f1 score is the harmonic mean of precision and recall ( this is the most improtant measure) the f1 score is calculated by doing 2 * (precision * recal) / precision + recall). support just shows us how many actaul cases of this class there are. 

print("confusion matrix")

cm = confusion_matrix(y_true, y_pred) # 2*2 grid showing all predicted outcomes.
print(cm)
print(f"\n true negatives (correct benign): {cm[0,0]}")     # self explantory 
print(f"\n false positves (benign called melanoma): {cm[0,1]}")  # self explantory 
print(f"\n false negatives (Melanoma called benign): {cm[1,0]}")         # self explantory 
print(f"\n true positvies (correct melanoma): {cm[1,1]}")        # self explantory 

print(f"\n ROC-AUC score: {roc_auc_score(y_true, y_pred_probs):.4f}")       # this gives us that one number showing overall clasification ability. this is almost the same thing as the f1 score but better.

#12. visualize training history


fig, axes = plt.subplots(2, 2, figsize=(15, 10))

axes[0, 0].plot(history.history['accuracy'], label='Train')
axes[0, 0].plot(history.history['val_accuracy'], label='Validation')
axes[0, 0].set_title('Model Accuracy')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].legend()
axes[0, 0].grid(True)

axes[0, 1].plot(history.history['loss'], label='Train')
axes[0, 1].plot(history.history['val_loss'], label='Validation')
axes[0, 1].set_title('Model Loss')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].legend()
axes[0, 1].grid(True)

axes[1, 0].plot(history.history['precision'], label='Train')
axes[1, 0].plot(history.history['val_precision'], label='Validation')
axes[1, 0].set_title('Model Precision')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Precision')
axes[1, 0].legend()
axes[1, 0].grid(True)

axes[1, 1].plot(history.history['recall'], label='Train')
axes[1, 1].plot(history.history['val_recall'], label='Validation')
axes[1, 1].set_title('Model Recall')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Recall')
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig('melanoma_training_history.png', dpi=300, bbox_inches='tight')
print("\nTraining history saved to 'melanoma_training_history.png'")

print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)

