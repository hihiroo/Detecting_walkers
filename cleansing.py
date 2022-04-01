# -- coding: utf-8 --
import cv2
import os
import argparse

gt_dir = "/home/adriv/Carla/CARLA_0.9.8/PythonAPI/custom/gt/"

'''
SpaceBar: select a bounding box
BackSpace: delete the selected bounding box
Left: previous frame
Right: next frame
Ctrl+Z: get back
Enter: save new ground truth file (new_gt.txt)
A: move the selected bounding box to the left
D: move the selected bounding box to the right
W: move the selected bounding box up
S: move the selected bounding box down
'''
''' Option
--start: select starting frame number
'''

def draw_bb(img, gt, frame):
    cv2.putText(img, str(frame), (10, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2)
    for label in gt:
        x, y = int(label[2]), int(label[3])
        w, h = int(label[4]), int(label[5])
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        description='bounding box cleanser')
    argparser.add_argument(
        '--start',
        default=0, type=int)
    args = argparser.parse_args()
    
    new_gt = []
    start_frame_idx = 0
    with open(gt_dir+"gt.txt", 'r') as label:
        frame = -1
        while True:
            line = label.readline()
            if line is None or line == '':
                break
            
            obj = line.split(',')
            if int(obj[0]) != frame:
                frame = int(obj[0])
                new_gt.append([])
                if frame == args.start:
                    start_frame_idx = len(new_gt) - 1
            new_gt[len(new_gt)-1].append(obj)
    
    finish = 0
    cnt = start_frame_idx
    cache_new_gt = [[] for _ in range(len(new_gt))]
    while 0 <= cnt < len(new_gt):
        gt = new_gt[cnt]
        cache_gt = cache_new_gt[cnt]
        
        if len(gt): frame = int(gt[0][0])
        else: frame = int(cache_gt[0][0])
        
        org = cv2.imread(gt_dir+"{0:06d}.jpg".format(frame), cv2.IMREAD_COLOR)
        src = org.copy()
        draw_bb(src, gt, frame)
        
        idx = -1
        while True:
            cv2.imshow("result", src)
            key = cv2.waitKeyEx()
            if key == 0x270000: # -> 방향키
                cnt += 1
                break
            
            elif key == 27: # esc
                finish = 1
                break
            
            elif key == 0x250000: # <- 방향키
                cnt -= 1
                break
            
            elif key == 32 and len(gt) > 0: # -> spacebar
                if idx != -1:
                    x, y = int(gt[idx][2]), int(gt[idx][3])
                    w, h = int(gt[idx][4]), int(gt[idx][5])
                    cv2.rectangle(src, (x,y), (x+w, y+h), (0,255,0), 2)
                idx = (idx + 1) % len(gt)
                x, y = int(gt[idx][2]), int(gt[idx][3])
                w, h = int(gt[idx][4]), int(gt[idx][5])
                cv2.rectangle(src, (x,y), (x+w, y+h), (0,0,255), 2)
                
            elif key == 8 and idx != -1 and len(gt) > 0: # backspace
                cache_gt.append(gt[idx])
                gt.remove(gt[idx])
                src = org.copy()
                draw_bb(src, gt, frame)
            
            elif key == ord('d') and idx != -1:
                gt[idx][2] = str(int(gt[idx][2]) + 2)
                src = org.copy()
                draw_bb(src, gt, frame)
                
            elif key == ord('a') and idx != -1:
                gt[idx][2] = str(int(gt[idx][2]) - 2)
                src = org.copy()
                draw_bb(src, gt, frame)
                
            elif key == ord('w') and idx != -1:
                gt[idx][3] = str(int(gt[idx][3]) - 2)
                src = org.copy()
                draw_bb(src, gt, frame)
            
            elif key == ord('s') and idx != -1:
                gt[idx][3] = str(int(gt[idx][3]) + 2)
                src = org.copy()
                draw_bb(src, gt, frame)
                    
            elif key == 26 and len(cache_gt) > 0: # ctrl+z
                label = cache_gt.pop()
                gt.append(label)
                x, y = int(label[2]), int(label[3])
                w, h = int(label[4]), int(label[5])
                cv2.rectangle(src, (x,y), (x+w, y+h), (0,255,0), 2)
                
            elif key == 13: # enter
                with open(gt_dir+"new_gt.txt", 'w') as f:
                    for new_label in new_gt:
                        for obj in new_label:
                            for i, component in enumerate(obj):
                                f.write(component)
                                if i+1 < len(obj):
                                    f.write(',')
                print("Saved")
        if finish: break
        
    if not finish:
        with open(gt_dir+"new_gt.txt", 'w') as f:
                    for new_label in new_gt:
                        for obj in new_label:
                            for i, component in enumerate(obj):
                                f.write(component)
                                if i+1 < len(obj):
                                    f.write(',')
    cv2.destroyAllWindows()