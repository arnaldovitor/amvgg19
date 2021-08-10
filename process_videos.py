import numpy as np
import cv2 as cv
import os


def capture_frame(video_path, i):
    cap = cv.VideoCapture(video_path)
    cap.set(1, i)
    ret, frame = cap.read()
    return ret, frame


def count_frames(video_path):
    cap = cv.VideoCapture(video_path)
    frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    return frames


def rgb_to_flow(frame, next_frame):
    mask = np.zeros_like(frame)
    mask[..., 1] = 255

    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    next_frame = cv.cvtColor(next_frame, cv.COLOR_BGR2GRAY)

    flow = cv.calcOpticalFlowFarneback(frame, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])
    mask[..., 0] = angle * 180 / np.pi / 2
    mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)
    flow_frame = cv.cvtColor(mask, cv.COLOR_HSV2BGR)

    return flow_frame


def videos_to_frames(input_path, output_path, interval, dict_directory, crop):
    listing = os.listdir(input_path)
    progress_count = 0

    for video_name in listing:
        progress_count += 1
        print("# progress: "+str(progress_count)+'/'+str(len(listing)))
        video_path = input_path+video_name
        frames = count_frames(video_path)
        crop_i = 1
        for i in range(1, frames-1, interval):
            try:
                ret, frame = capture_frame(video_path, i)
                ret, next_frame = capture_frame(video_path, i+1)
                flow_frame = rgb_to_flow(frame, next_frame)

                if dict_directory:
                    cv.imwrite(output_path+video_name[:-4]+"_"+str(i)+"_rgb.png", frame)
                    cv.imwrite(output_path+video_name[:-4]+"_"+str(i)+"_flow.png", flow_frame)
                else:
                    if not os.path.exists(output_path+video_name[:-4]):
                        os.mkdir(output_path + video_name[:-4])
                    if crop:
                        if i % 90 == 0:
                            crop_i = crop_i * 10
                        if not os.path.exists(output_path+video_name[:-4]+"/"+str(crop_i)):
                            os.mkdir(output_path + video_name[:-4]+"/"+str(crop_i))
                        cv.imwrite(output_path+video_name[:-4]+"/"+str(crop_i)+"/"+video_name[:-4]+"_"+str(i)+"_rgb.png", frame)
                        cv.imwrite(output_path+video_name[:-4]+"/"+str(crop_i)+"/"+video_name[:-4]+"_"+str(i)+"_flow.png", flow_frame)
                    else:
                        cv.imwrite(output_path+video_name[:-4]+"/"+video_name[:-4]+"_"+str(i)+"_rgb.png", frame)
                        cv.imwrite(output_path+video_name[:-4]+"/"+video_name[:-4]+"_"+str(i)+"_flow.png", flow_frame)
            except:
                print('Error in video: {} and frame: {}'.format(video_name, i))

if __name__ == '__main__':
    videos_to_frames(r"/home/arnaldo/Documents/violent-flows-dataset-separada/dict/", r"/home/arnaldo/Documents/violent-flows-dataset-separada/dict-frames/", 1,True, False)
    videos_to_frames(r"/home/arnaldo/Documents/violent-flows-dataset-separada/train/violence/", r"/home/arnaldo/Documents/violent-flows-dataset-separada/train-frames/violence-frames/", 1, False, False)
    videos_to_frames(r"/home/arnaldo/Documents/violent-flows-dataset-separada/train/non-violence/", r"/home/arnaldo/Documents/violent-flows-dataset-separada/train-frames/non-violence-frames/", 1, False, False)
    videos_to_frames(r"/home/arnaldo/Documents/violent-flows-dataset-separada/test/violence/", r"/home/arnaldo/Documents/violent-flows-dataset-separada/test-frames/violence-frames/", 1, False, False)
    videos_to_frames(r"/home/arnaldo/Documents/violent-flows-dataset-separada/test/non-violence/", r"/home/arnaldo/Documents/violent-flows-dataset-separada/test-frames/non-violence-frames/", 1, False, False)