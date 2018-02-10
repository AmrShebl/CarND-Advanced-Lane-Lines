import cv2
import glob
import matplotlib.pyplot as plt
from pipeline import Lane
from moviepy.editor import VideoFileClip, ImageSequenceClip

def main():
    dir = './camera_cal/calibration*.jpg'
    my_lane = Lane()
    my_lane.calibrate_camera(dir,(9,6))
    # dir = './test_images/test*.jpg'
    # dir = './test_images/straight_lines*.jpg'
    # imageFs=glob.glob(dir)
    # for imageF in imageFs:
    #     img = cv2.imread(imageF)
    #     my_lane.process_image(img, True)


    input_clip_file='project_video.mp4'
    input_clip = VideoFileClip(input_clip_file)
    output_clip_file='output_project_video.mp4'
    output_frame_list=[]
    for frame in input_clip.iter_frames():
        output_frame=my_lane.process_image(frame)
        output_frame_list.append(output_frame)

    output_clip=ImageSequenceClip(output_frame_list, input_clip.fps)
    output_clip.write_videofile(output_clip_file)


if __name__ == "__main__":
    main()