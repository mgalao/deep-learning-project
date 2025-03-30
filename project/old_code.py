# Set up the dataset directory

# Bruna
#path = r"/Users/brunasimoes/Desktop/nova_ims/2_semester/Trimestral/deep_learning/Projeto/rare_species"

#Margas
# path = r"/Users/margaridabravocardoso/Desktop/dsaa/fun_year/2nd_semester/deep_learning/rare_species"









# Set up the ImageDataGenerators and define the augmentation parameters:
train_datagen = ImageDataGenerator(
    #rotation_range=20,      
    #zoom_range=0.2,
    #rotation_range=0, 
    #width_shift_range=0.1,  
    #height_shift_range=0.1,  
    #zoom_range=0,  # No zoom
    #horizontal_flip=False,  
    #vertical_flip=False,  
    #brightness_range=[0.8, 1.2],  
    #fill_mode='nearest', 
)
valid_datagen = ImageDataGenerator(
    #rotation_range=20,      
    #zoom_range=0.2,
    """""
    rotation_range=0, 
    width_shift_range=0.1,  
    height_shift_range=0.1,  
    zoom_range=0,  # No zoom
    horizontal_flip=False,  
    vertical_flip=False, 
    brightness_range=[0.8, 1.2],  
    fill_mode='nearest', 
    """
)
test_datagen = ImageDataGenerator(
    """""
    #rotation_range=20,       
    #zoom_range=0.2,  
    rotation_range=0,  
    width_shift_range=0.1,  
    height_shift_range=0.1,  
    zoom_range=0,  
    horizontal_flip=False,  
    vertical_flip=False, 
    brightness_range=[0.8, 1.2],  
    fill_mode='nearest', 
    """
)

# Flow from directory for training data
train_generator = train_datagen.flow_from_dataframe(
    train_df,
    directory=train_path,
    x_col='full_file_path',
    y_col='family',
    color_mode='rgb', 
    target_size=(224, 224), 
    batch_size=20, 
    class_mode='categorical',
    shuffle = True,
    class_weights = 'balanced',
    rescale=1./255  # Normalization factor
)

# Flow from directory for validation data
valid_generator = valid_datagen.flow_from_dataframe(
    valid_df,
    directory=valid_path,
    x_col='full_file_path',
    color_mode='rgb',
    y_col='family',
    target_size=(224, 224),
    batch_size=20,
    class_mode='categorical',
    shuffle = False,
    class_weights = 'balanced',
    rescale=1./255  # Normalization factor
)

# Flow from directory for testing data
test_generator = test_datagen.flow_from_dataframe(
    test_df,
    directory=test_path,
    x_col='full_file_path',
    color_mode='rgb',
    y_col='family',
    target_size=(224, 224),
    batch_size=20,
    class_mode='categorical',
    shuffle = False, 
    class_weights = 'balanced',
    rescale=1./255  # Normalization factor
)