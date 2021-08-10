import tensorflow as tf
import numpy as np
import os

vgg19 = tf.keras.applications.VGG19(weights='imagenet', include_top=True)
model = tf.keras.models.Model(inputs=vgg19.input, outputs=vgg19.get_layer('fc2').output)

def frames_to_vectors(input_path, output_path, output_name, interval):
    all_features = []
    listing = os.listdir(input_path)

    for i in range(0, len(listing), interval):
        print("## frame: "+ str(i) + '/' +str(len(listing)-1))
        prep_frame = tf.keras.preprocessing.image.load_img(input_path+listing[i], target_size=(224, 224))
        x = tf.keras.preprocessing.image.img_to_array(prep_frame)
        x = np.expand_dims(x, axis=0)
        x = tf.keras.applications.vgg19.preprocess_input(x)
        features = model.predict(x).reshape(4096)
        all_features.append(features)

    np.savetxt(output_path+output_name + ".csv", all_features, delimiter=",")


def run_feature_extractor(input_path, output_path, dict_directory, interval):
    if dict_directory:
        frames_to_vectors(input_path, output_path, "dict", interval)
    else:
        listing = os.listdir(input_path)
        listing.sort()
        progress_count = 0
        for directory in listing:
            progress_count += 1
            print("# progress: " + str(progress_count) + '/' + str(len(listing)))
            frames_to_vectors(input_path+directory+"/", output_path, directory, interval)


if __name__ == '__main__':
    #run_feature_extractor(r"/home/arnaldo/Documents/violent-flows-dataset-separada/dict-frames/", r"/home/arnaldo/Documents/violent-flows-dataset-separada/csv/", True, 1)
    run_feature_extractor(r"/home/arnaldo/Documents/violent-flows-dataset-separada/train-frames/violence-frames/", r"/home/arnaldo/Documents/violent-flows-dataset-separada/csv/train/violence/", False, 1)
    run_feature_extractor(r"/home/arnaldo/Documents/violent-flows-dataset-separada/train-frames/non-violence-frames/", r"/home/arnaldo/Documents/violent-flows-dataset-separada/csv/train/non-violence/", False, 1)
    run_feature_extractor(r"/home/arnaldo/Documents/violent-flows-dataset-separada/test-frames/violence-frames/", r"/home/arnaldo/Documents/violent-flows-dataset-separada/csv/test/violence/", False, 1)
    run_feature_extractor(r"/home/arnaldo/Documents/violent-flows-dataset-separada/test-frames/non-violence-frames/", r"/home/arnaldo/Documents/violent-flows-dataset-separada/csv/test/non-violence/", False, 1)