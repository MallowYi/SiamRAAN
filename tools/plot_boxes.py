from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import natsort
import sys
sys.path.append('../')

import argparse
import cv2
import torch
from glob import glob

torch.set_num_threads(1)


parser = argparse.ArgumentParser(description='SiamCAR demo')
parser.add_argument('--config', type=str, default='../experiments/siamcar_r50/config.yaml', help='config file')
# parser.add_argument('--video_name', default='../test_dataset/ants1', type=str, help='videos or image files')
parser.add_argument('--video_name', default='/home/dl/project/xzy/SiamCAR-master/testing_dataset/GOT10k/', type=str, help='videos or image files')
args = parser.parse_args()
gt = [129,246,404,303]

def get_frames(video_name):
    if not video_name:
        cap = cv2.VideoCapture(0)

        # warmup
        for i in range(5):
            cap.read()
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    elif video_name.endswith('avi') or \
        video_name.endswith('mp4'):
        cap = cv2.VideoCapture(args.video_name)
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    else:
        # images = sorted(glob(os.path.join(video_name, 'img', '*.jp*')))
        images = sorted(glob(os.path.join(video_name, '*.jp*')))

        for img in images:
            frame = cv2.imread(img)
            yield frame


def main():
    first_frame = True
    if args.video_name:
        video_name = args.video_name.split('/')[-1].split('.')[0]
    else:
        video_name = 'webcam'
    cv2.namedWindow('webcam', cv2.WND_PROP_FULLSCREEN)
    files = natsort.natsorted(os.listdir('/home/dl/project/xzy/SiamCAR-master/testing_dataset/GOT-10k/'))
    for i in range(180):
        # with open(args.video_name + '/groundtruth.txt') as f:
        # images = natsort.natsorted(os.listdir('/home/dl/code/SiamCAR-master/testing_dataset/GOT10k/'+files[i+2]))
        fra = 1
        if i+1 < 10:
            t = str('00')+str(i+1)
        elif i+1 < 100:
            t = str('0')+str(i+1)
        else:
            t = str(i+1)
        ours = open('/home/dl/project/xzy/SiamCAR-master/tools/results/GOT-10k/siamraan/'+files[i+1]+'/GOT-10k_Test_000'+t+'_001.txt')
        afsn = open('/home/dl/project/xzy/SiamCAR-master/tools/results/GOT-10k/AFSN/'+files[i+1]+'/GOT-10k_Test_000'+t+'_001.txt')
        siamcar = open('/home/dl/project/xzy/SiamCAR-master/tools/results/GOT-10k/siamcar/'+files[i+1]+'/GOT-10k_Test_000'+t+'_001.txt')
        siamrpn = open('/home/dl/project/xzy/SiamCAR-master/tools/results/GOT-10k/siamrpn++/'+files[i+1]+'/GOT-10k_Test_000'+t+'_001.txt')
        for frame in get_frames('/home/dl/code/SiamCAR-master/testing_dataset/GOT10k/'+files[i+1]):
                # gt = list(map(int, f.readline().split(',')))
            ours_boxes = list(map(float, ours.readline().split(',')))
            afsn_boxes = list(map(float, afsn.readline().split(',')))
            siamcar_boxes = list(map(float, siamcar.readline().split(',')))
            siamrpn_boxes = list(map(float, siamrpn.readline().split(',')))
            path = '/home/dl/project/xzy/SiamCAR-master/testing_dataset/GOT-10k/test/'+files[i+2]
            if not os.path.exists(path):
               os.makedirs(path)
            if first_frame:

                cv2.rectangle(frame, (int(afsn_boxes[0]), int(afsn_boxes[1])),
                              (int(afsn_boxes[0] + gt[2]), int(afsn_boxes[1] + afsn_boxes[3])),
                              (0, 255, 0), 3)
                cv2.rectangle(frame, (int(ours_boxes[0]), int(ours_boxes[1])),
                              (int(ours_boxes[0] + ours_boxes[2]), int(ours_boxes[1] + ours_boxes[3])),
                              (0, 0, 255), 3)
                cv2.rectangle(frame, (int(siamcar_boxes[0]), int(siamcar_boxes[1])),
                              (int(siamcar_boxes[0] + siamcar_boxes[2]), int(siamcar_boxes[1] + siamcar_boxes[3])),
                              (255, 0, 0), 3)
                cv2.rectangle(frame, (int(siamrpn_boxes[0]), int(siamrpn_boxes[1])),
                              (int(siamrpn_boxes[0] + siamrpn_boxes[2]), int(siamrpn_boxes[1] + siamrpn_boxes[3])),
                              (34, 227, 214), 3)
                cv2.putText(frame, '#'+str(fra), (50,100), cv2.FONT_HERSHEY_COMPLEX, 2.0, (255,255,0), 2)
                # print(fra)
                cv2.imwrite(path + '/' + str(fra).zfill(4) + '.jpg', frame)
                first_frame = False
            else:
                fra += 1
                print(fra)
                cv2.rectangle(frame, (int(afsn_boxes[0]), int(afsn_boxes[1])),
                              (int(afsn_boxes[0]+afsn_boxes[2]), int(afsn_boxes[1]+afsn_boxes[3])),
                              (0, 255, 0), 3)
                cv2.rectangle(frame, (int(ours_boxes[0]), int(ours_boxes[1])),
                              (int(ours_boxes[0] + ours_boxes[2]), int(ours_boxes[1] + ours_boxes[3])),
                              (0, 0, 255), 3)
                cv2.rectangle(frame, (int(siamcar_boxes[0]), int(siamcar_boxes[1])),
                              (int(siamcar_boxes[0] + siamcar_boxes[2]), int(siamcar_boxes[1] + siamcar_boxes[3])),
                              (255, 0, 0), 3)
                cv2.rectangle(frame, (int(siamrpn_boxes[0]), int(siamrpn_boxes[1])),
                              (int(siamrpn_boxes[0] + siamrpn_boxes[2]), int(siamrpn_boxes[1] + siamrpn_boxes[3])),
                              (34, 227, 214), 3)
                cv2.putText(frame, '#'+str(fra), (50,100), cv2.FONT_HERSHEY_COMPLEX, 2.0, (255,255,0), 2)
                cv2.imwrite(path + '/' + str(fra).zfill(4) + '.jpg',frame)
                    # cv2.rectangle(frame, (lines[0], lines[1]),
                    #               (lines[0] + lines[2], lines[1] + lines[3]),
                    #               (0, 0, 255), 1)
                cv2.imshow(video_name, frame)
                cv2.waitKey(40)


if __name__ == '__main__':
    main()
