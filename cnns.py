import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.optimizers import RMSprop
from codecarbon import EmissionsTracker
base_dir = '/Users/furkangulkan/PycharmProjects/TasarımCalismasi/mnist'
train_dir = os.path.join(base_dir, 'trainingSet/trainingSet')
validation_dir = os.path.join(base_dir, 'testSet/testSet')
# Image Augmentation
train_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator()
# Training and Validation Sets
train_generator = train_datagen.flow_from_directory(train_dir, batch_size = 64, class_mode="categorical", target_size = (224, 224) , color_mode="rgb")
validation_generator = test_datagen.flow_from_directory( validation_dir,  batch_size = 64, class_mode="categorical", target_size = (224, 224) , color_mode="rgb")
#train_generator = train_datagen.flow_from_directory(train_dir, batch_size = 64, class_mode="categorical", target_size = (224, 224) , color_mode="rgb")
#validation_generator = test_datagen.flow_from_directory( validation_dir,  batch_size = 64, class_mode="categorical", target_size = (224, 224) , color_mode="rgb")

# Dataset has been set

# Xception
print("Xception State")

base_model = tf.keras.applications.xception.Xception(
    input_shape = (224, 224, 3),
    include_top=False,
    weights = None
    #weights= 'imagenet'
)
for layer in base_model.layers:
    layer.trainable = False
# Compile and Fit
    # Flatten the output layer to 1 dimension
x = layers.Flatten()(base_model.output)
x = layers.Dense(1024, activation="relu")(x)
    # Add a dropout rate of 0.5
x = layers.Dropout(0.5)(x)
    # Add a final sigmoid layer with 1 node for classification output
x = layers.Dense(10, activation='sigmoid')(x)
model = tf.keras.models.Model(base_model.input, x)


model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['acc'])
# Saving the models weight to reset before each step for 1-10-50 epoch
Wsave = model.get_weights()
# Fitting the model
tracker = EmissionsTracker()
tracker.start()
print("Xception is being run for 1 epoch")
xception_history1 = model.fit(train_generator, validation_data=validation_generator, steps_per_epoch = 100, epochs = 1)
emissions: float = tracker.stop()
print("%.4f" % (emissions*1000) + " gram")
model.save("Xception_1epoch")
model.set_weights(Wsave)# Resetting the model's weight
print("Xception is being run for 10 epoch")
tracker = EmissionsTracker()
tracker.start()
xception_history10 = model.fit(train_generator, validation_data=validation_generator, steps_per_epoch = 100, epochs = 10)
emissions: float = tracker.stop()
print("%.4f" % (emissions*1000) + " gram")
model.save("Xception_10epoch")
model.set_weights(Wsave)
print("Xception is being run for 25 epoch")
tracker = EmissionsTracker()
tracker.start()
xception_history50 = model.fit(train_generator, validation_data=validation_generator, steps_per_epoch = 100, epochs = 25)
emissions: float = tracker.stop()
print("%.4f" % (emissions*1000) + " gram")
model.save("Xception_50epoch")
model.set_weights(Wsave)


### 1.Very Deep Convolutional Networks for Large-Scale Image Recognition(VGG-16)
print("VGG16 State")
# Loading the Base Model
from tensorflow.keras.applications.vgg16 import VGG16

base_model = VGG16( input_shape = (224, 224, 3), # Shape of our images
                    include_top = False, # Leave out the last fully connected layer
                    weights = None)#"imagenet"

for layer in base_model.layers:
    layer.trainable = False
# Compile and Fit
    # Flatten the output layer to 1 dimension
x = layers.Flatten()(base_model.output)
x = layers.Dense(1024, activation="relu")(x)
    # Add a dropout rate of 0.5
x = layers.Dropout(0.5)(x)
    # Add a final sigmoid layer with 1 node for classification output
x = layers.Dense(10, activation='sigmoid')(x)

model = tf.keras.models.Model(base_model.input, x)
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.0001 ), loss='categorical_crossentropy', metrics=['acc'])
# Saving the models weight to reset before each step for 1-10-50 epoch
Wsave = model.get_weights()
# Fitting the model
print("VGG16 is being run for 1 epoch")
tracker = EmissionsTracker()
tracker.start()
vgg_history1 = model.fit(train_generator, validation_data=validation_generator, steps_per_epoch = 100, epochs = 1)
emissions: float = tracker.stop()
print("%.4f" % (emissions*1000) + " gram")
model.save("vgg16_1epoch")
model.set_weights(Wsave)# Resetting the model's weight
print("VGG16 is being run for 10 epoch")
tracker = EmissionsTracker()
tracker.start()
vgg_history10 = model.fit(train_generator, validation_data=validation_generator, steps_per_epoch = 100, epochs = 10)
emissions: float = tracker.stop()
print("%.4f" % (emissions*1000) + " gram")
model.save("vgg16_10epoch")
model.set_weights(Wsave)
print("VGG16 is being run for 25 epoch")
tracker = EmissionsTracker()
tracker.start()
vgg_history50 = model.fit(train_generator, validation_data=validation_generator, steps_per_epoch = 100, epochs = 25)
emissions: float = tracker.stop()
print("%.4f" % (emissions*1000) + " gram")
model.save("vgg16_50epoch")
model.set_weights(Wsave)

### 2.Inception
print("Inception State")
# Loading the Base Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
base_model = InceptionV3(input_shape = (224, 224, 3), include_top = False, weights = None)

# Compile and Fit
for layer in base_model.layers:
    layer.trainable = False

x = layers.Flatten()(base_model.output)
x = layers.Dense(1024, activation="relu")(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(10, activation='sigmoid')(x)

model = tf.keras.models.Model(base_model.input, x)
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.0001 ), loss='categorical_crossentropy', metrics=['acc'])
# Saving the models weight to reset before each step for 1-10-50 epoch
Wsave = model.get_weights()
# Fitting the model
print("Inceptionv3 is being run for 1 epoch")
tracker = EmissionsTracker()
tracker.start()
inc_history1 = model.fit(train_generator, validation_data=validation_generator, steps_per_epoch = 100, epochs = 1)
emissions: float = tracker.stop()
print("%.4f" % (emissions*1000) + " gram")
model.save("Inceptionv3_1epoch")
model.set_weights(Wsave)# Resetting the model's weight
print("Inceptionv3 is being run for 10 epoch")
tracker = EmissionsTracker()
tracker.start()
inc_history10 = model.fit(train_generator, validation_data=validation_generator, steps_per_epoch = 100, epochs = 10)
emissions: float = tracker.stop()
print("%.4f" % (emissions*1000) + " gram")
model.save("Inceptionv3_10epoch")
model.set_weights(Wsave)
print("Inceptionv3 is being run for 50 epoch")
tracker = EmissionsTracker()
tracker.start()
inc_history50 = model.fit(train_generator, validation_data=validation_generator, steps_per_epoch = 100, epochs = 25)
emissions: float = tracker.stop()
print("%.4f" % (emissions*1000) + " gram")
model.save("Inceptionv3_50epoch")
model.set_weights(Wsave)

### 3. ResNet50v2
print("ResNet50v2 State")
# Import the base model
from tensorflow.keras.applications import ResNet50V2
base_model = ResNet50V2(input_shape=(224, 224,3), include_top=False, weights=None)

for layer in base_model.layers:
    layer.trainable = False

# Build and Compile the Model
x = layers.Flatten()(base_model.output)
x = layers.Dense(1024, activation="relu")(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(10, activation='sigmoid')(x)

model = tf.keras.models.Model(base_model.input, x)
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['acc'])
# Saving the models weight to reset before each step for 1-10-50 epoch
Wsave = model.get_weights()
# Fitting the model
print("Resnet50v2 is being run for 1 epoch")
tracker = EmissionsTracker()
tracker.start()
resnet_history1 = model.fit(train_generator, validation_data = validation_generator, steps_per_epoch = 100, epochs = 1)
emissions: float = tracker.stop()
print("%.4f" % (emissions*1000) + " gram")
model.save("Resnet50v2_1epoch")
model.set_weights(Wsave)# Resetting the model's weight
print("Resnet50v2 is being run for 10 epoch")
tracker = EmissionsTracker()
tracker.start()
resnet_history10 = model.fit(train_generator, validation_data = validation_generator, steps_per_epoch = 100, epochs = 10)
emissions: float = tracker.stop()
print("%.4f" % (emissions*1000) + " gram")
model.save("Resnet50v2_10epoch")
model.set_weights(Wsave)
print("Resnet50v2 is being run for 25 epoch")
tracker = EmissionsTracker()
tracker.start()
resnet_history50 = model.fit(train_generator, validation_data = validation_generator, steps_per_epoch = 100, epochs = 25)
emissions: float = tracker.stop()
print("%.4f" % (emissions*1000) + " gram")
model.save("Resnet50v2_50epoch")
model.set_weights(Wsave)