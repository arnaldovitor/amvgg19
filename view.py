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

def main():
    limiar = 0.5

    print('# (1) Teste com vídeo do YouTube')
    print('# (2) Teste com vídeos da base')

    op = int(input())

    if op == 1:
        print('# Insira a URL do vídeo:')
        url = input()
        video_name = youtube.get_yt_video(url, 'video_path')
        # video_name = os.listdir(r"video_path/")[0]

        if not os.path.exists("frame_path/"+video_name[:-4]+"/"):
            process_videos.single_video_to_frames(r"video_path/"+video_name, r"frame_path/", 5)
        if not os.path.exists("csv_path/" + video_name[:-4] + "/"):
            os.mkdir("csv_path/" + video_name[:-4] + "/")
            feature_extractor.run_feature_extractor(r"frame_path/"+video_name[:-4]+"/", r"csv_path/" + video_name[:-4] + "/", False, 1)
        yt = True

    elif op == 2:
        print('# Qual vídeo quer testar? [1-28]')
        video_name = input()+".avi"
        # video_name = os.listdir(r"video_path/")[0]

        if not os.path.exists("frame_path/" + video_name[:-4] + "/"):
            process_videos.single_video_to_frames(r"video_path/" + video_name, r"frame_path/", 5)
        if not os.path.exists("csv_path/" + video_name[:-4] + "/"):
            os.mkdir("csv_path/" + video_name[:-4] + "/")
            feature_extractor.run_feature_extractor(r"frame_path/" + video_name[:-4] + "/", r"csv_path/" + video_name[:-4] + "/", False, 1)
        yt = False

    kmeans = pickle.load(open("clt.pkl", "rb"))
    vbow.gen_vbow(r"csv_path/" + video_name[:-4] + "/", r"csv_path/" + video_name[:-4] + "/test", kmeans, 0)

    X_test, y_test = util.separe_data(r"csv_path/" + video_name[:-4] + "/test.csv", 128)
    X_test = classifiers.normalize_x_test(X_test)

    model = tf.keras.models.load_model(r"model_path/")
    y_pred = model.predict(X_test)


    print(y_pred)
    print(y_pred > limiar)

    view(video_name[:-4], y_pred, limiar, yt)


def view(video_name, y_pred, limiar, yt_video):
    if yt_video:
        cap = cv2.VideoCapture(r'video_path/'+video_name+'.mp4')
    else:
        cap = cv2.VideoCapture(r'video_path/' + video_name + '.avi')


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


if __name__ == '__main__':
    main()
