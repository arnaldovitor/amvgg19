import tensorflow as tf
import numpy as np
import os

vgg19 = tf.keras.applications.VGG19(weights='imagenet', include_top=True)
model = tf.keras.models.Model(inputs=vgg19.input, outputs=vgg19.get_layer('fc2').output)


def frames_to_vectors(input_path, output_name):
    all_features = []
    listing = os.listdir(input_path)
    for frame in listing:
        prep_frame = tf.keras.preprocessing.image.load_img(input_path+frame, target_size=(224, 224))
        x = tf.keras.preprocessing.image.img_to_array(prep_frame)
        x = np.expand_dims(x, axis=0)
        x = tf.keras.applications.vgg19.preprocess_input(x)
        features = model.predict(x).reshape(4096)
        print(features)
        all_features.append(features)
    np.savetxt(output_name + ".csv", all_features, delimiter=",")


if __name__ == '__main__':
    frames_to_vectors(r"/home/user/input_path/", "file_name", 0)