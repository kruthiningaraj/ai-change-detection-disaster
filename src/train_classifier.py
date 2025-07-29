import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def train_classifier(train_dir, val_dir, img_size=(224,224), batch_size=32, epochs=10):
    datagen = ImageDataGenerator(rescale=1./255)
    train_gen = datagen.flow_from_directory(train_dir, target_size=img_size, batch_size=batch_size, class_mode='binary')
    val_gen = datagen.flow_from_directory(val_dir, target_size=img_size, batch_size=batch_size, class_mode='binary')

    base_model = tf.keras.applications.InceptionV3(weights='imagenet', include_top=False, input_shape=(224,224,3))
    base_model.trainable = False

    x = tf.keras.layers.Flatten()(base_model.output)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(train_gen, validation_data=val_gen, epochs=epochs)
    model.save('models/disaster_classifier.h5')
