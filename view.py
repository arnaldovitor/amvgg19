import os
import cv2
import process_videos
import feature_extractor
import vbow
import classifiers
import youtube
import util
import tensorflow as tf
import pickle

# youtube.get_yt_video('https://www.youtube.com/watch?v=Hr_fYoUkVuo&ab_channel=22brigada', 'video_path')
# process_videos.videos_to_frames(r"video_path/", r"frame_path/", 1, False, True)

video_name = os.listdir(r"video_path/")[0][:-4]
# feature_extractor.run_feature_extractor(r"frame_path/"+video_name+"/", r"csv_path/", False, 1)
#
# kmeans = pickle.load(open("clt.pkl", "rb"))
# vbow.gen_vbow(r"csv_path/", r"csv_path/test", kmeans, 0)

X, y = util.separe_data(r"train.csv", 256)
X_val, y_val = util.separe_data(r"val.csv", 256)
X_test, y_test = util.separe_data(r"csv_path/test.csv", 256)

_, _, y_pred = classifiers.apply_svm(X, y, X_test, y_test, 'rbf')
# y_pred = classifiers.apply_tree(X, y, X_test, y_test)
#y_pred = classifiers.apply_fcn(X, y, X_val, y_val)

# model = tf.keras.models.load_model(r"model_path/")
# y_pred = model.predict(X_test)
limiar = 0.5

print(y_pred)
print(y_pred > limiar)

cap = cv2.VideoCapture(r'video_path/'+video_name+'.mp4')

if (cap.isOpened()== False):
  print("Error opening video stream or file")

i_frame = 0
i_pred = 0

while(cap.isOpened()):
    i_frame += 1
    if i_frame % 90 == 0:
        i_pred += 1

    ret, frame = cap.read()

    if ret == True:
        if y_pred[i_pred] > limiar:
            frame[:, :, 0] = 0
            frame[:, :, 1] = 0
        else:
            frame[:, :, 0] = 0
            frame[:, :, 2] = 0

        cv2.imshow('Frame', frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()



