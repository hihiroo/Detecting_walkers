from nis import cat
import os
import argparse
import readline


# Program Argument
parser = argparse.ArgumentParser(description='arguparser')
parser.add_argument('--path', required=True,
                    help='the path where has target files.')
parser.add_argument('--target', required=True,
                    help='the environment file for target files.')
parser.add_argument('--type', required=True,
                    help='if you want to remove gt version, typing gt, else if origin recoding version,typing recoding.')


# Convert raw file name to formatting version name
# For example, '100' to '00000100.png' and '00000100.txt'
def to_format(raw, type):
    result = []

    if type == 'gt':
        result.append(raw.zfill(6) + '.jpg') # gt file format : 000nnn.jpg, ...
    elif type == 'recoding':
        result.append(raw.zfill(8) + '.png') # recoding file format : 0000nnnn.png, 0000nnnn.txt, ...
        result.append(raw.zfill(8) + '.txt')
    else:
        print("[Error] Formatting error." + raw + " " + type)
    
    return result


# Reading Target file list from argument '--target'
# Return list of filename
def read_file(target_file, file_type):
    filelist = []
    framelist = []
    # result = []
    
    try :
        f = open(target_file, 'r')
        while True:
            line = f.readline().split()
            if not line: break
            if len(line) == 1: # abc
                framelist.append(line[0])
                filelist += to_format(line[0], file_type) # remove abc.png
            elif len(line) == 2: # abc def
                for raw in range(int(line[0]), int(line[1])+1): # remove abc.png~def.png
                    framelist.append(str(raw))
                    filelist += to_format(str(raw), file_type)
        f.close()

    except:
        print("[Error] Reading target_file error.")
        filelist = []
        framelist = []

    return filelist, framelist


# 변경 사항을 적용한다
def apply_changes(file_list, frame_list, path, type):
    # 파일 삭제
    for filename in file_list:
            file_path = args.path + filename
            if os.path.isfile(file_path):
                os.remove(file_path)
            # print(file_path)
    
    # gt파일 수정
    if type == 'gt':
        new_f = open(path+'gt_after.txt', 'w')
        prv_f = open(path+'gt.txt', 'r')

        while True:
            line = prv_f.readline()
            if not line: break
            frame_no = line.split(',')[0]
            if (frame_no in frame_list):
                continue
            else:
                new_f.write(line)

        new_f.close()
        prv_f.close()

        print("gt file changes at" + path+'gt_after.txt')



if __name__ == '__main__':
    args = parser.parse_args()

    if (os.path.isdir(args.path) and os.path.isfile(args.target)):
        filelist, framelist = read_file(args.target, args.type)
        apply_changes(file_list=filelist, frame_list=framelist, path=args.path, type=args.type)

        
        print('\nRemove success.')

    else:
        print("[Error] Argument file or folder path is not exist. Exit program.")
    
