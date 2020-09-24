import tensorflow as tf
import numpy as np
import os

vgg19 = tf.keras.applications.VGG19(weights='imagenet', include_top=True)
model = tf.keras.models.Model(inputs=vgg19.input, outputs=vgg19.get_layer('fc2').output)


def frames_to_vectors(input_path, output_path, output_name):
    all_features = []
    listing = os.listdir(input_path)
    for frame in listing:
        prep_frame = tf.keras.preprocessing.image.load_img(input_path+frame, target_size=(224, 224))
        x = tf.keras.preprocessing.image.img_to_array(prep_frame)
        x = np.expand_dims(x, axis=0)
        x = tf.keras.applications.vgg19.preprocess_input(x)
        features = model.predict(x).reshape(4096)
        all_features.append(features)
    np.savetxt(output_path+output_name + ".csv", all_features, delimiter=",")


def run_feature_extractor(input_path, output_path, dict_directory):
    if dict_directory:
        frames_to_vectors(input_path, output_path, "dict")
    else:
        listing = os.listdir(input_path)
        progress_count = 0
        for directory in listing:
            progress_count += 1
            print("# progress: " + str(progress_count) + '/' + str(len(listing)))
            frames_to_vectors(input_path+directory+"/", output_path, directory)


if __name__ == '__main__':
    run_feature_extractor(r"/home/arnaldo/Documentos/aie-dataset-separada/dict-frames/", r"/home/arnaldo/Documentos/aie-dataset-separada/csv/", True)
    run_feature_extractor(r"/home/arnaldo/Documentos/aie-dataset-separada/validation-frames/assault-frames/", r"/home/arnaldo/Documentos/aie-dataset-separada/csv/assault/", False)
    run_feature_extractor(r"/home/arnaldo/Documentos/aie-dataset-separada/validation-frames/non-assault-frames/", r"/home/arnaldo/Documentos/aie-dataset-separada/csv/non-assault/", False)