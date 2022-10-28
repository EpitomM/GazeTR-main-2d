import numpy as np
import scipy.io as sio
import cv2 
import os
import sys
import gtools
from easydict import EasyDict as edict
sys.path.append("../core/")
import data_processing_core as dpc
# import im_plot as ipt

root = "D:\datasets\MPIIFaceGaze"
sample_root = "D:\datasets\MPIIGaze\Evaluation Subset\sample list for eye image"
out_root = "D:\datasets\MPIIFaceGaze_GazePoint"

def ImageProcessing_MPII():
    persons = os.listdir(sample_root)
    persons.sort()
    for person in persons:
        sample_list = os.path.join(sample_root, person)

        person = person.split(".")[0]
        im_root = os.path.join(root, person)
        anno_path = os.path.join(root, person, f"{person}.txt")

        im_outpath = os.path.join(out_root, "Image", person)
        label_outpath = os.path.join(out_root, "Label", f"{person}.label")

        if not os.path.exists(im_outpath):
            os.makedirs(im_outpath)
        if not os.path.exists(os.path.join(out_root, "Label")):
            os.makedirs(os.path.join(out_root, "Label"))

        screen = os.path.join(root, person, "Calibration", "screenSize.mat")

        print(f"Start Processing {person}")
        ImageProcessing_Person(im_root, anno_path, screen, sample_list, im_outpath, label_outpath, person)


def ImageProcessing_Person(im_root, anno_path, screen_path, sample_list, im_outpath, label_outpath, person):

    # Read gaze annotation
    annotation = os.path.join(anno_path)
    with open(annotation) as infile:
        anno_info = infile.readlines()
    anno_dict = {line.split(" ")[0]: line.strip().split(" ")[1:-1] for line in anno_info}

    screen = edict()
    infile = sio.loadmat(screen_path)
    screen.width_p = infile['width_pixel'].flatten()[0]
    screen.height_p = infile['height_pixel'].flatten()[0]
    screen.width_m = infile['width_mm'].flatten()[0]
    screen.height_m = infile['height_mm'].flatten()[0]
    screen.width_r = screen.width_m/screen.width_p
    screen.height_r = screen.height_m/screen.height_p


    # Create the handle of label 

    outfile = open(label_outpath, 'w')
    outfile.write("Face Left Right Grid Origin whicheye 2DPoint HeadRot HeadTrans ratio FaceCorner LeftEyeCorner RightEyeCorner\n")

    if not os.path.exists(os.path.join(im_outpath, "face")):
        os.makedirs(os.path.join(im_outpath, "face"))
    if not os.path.exists(os.path.join(im_outpath, "left")):
        os.makedirs(os.path.join(im_outpath, "left"))
    if not os.path.exists(os.path.join(im_outpath, "right")):
        os.makedirs(os.path.join(im_outpath, "right"))
    if not os.path.exists(os.path.join(im_outpath, "grid")):
        os.makedirs(os.path.join(im_outpath, "grid"))


    # Image Processing 
    with open(sample_list) as infile:
        im_list = infile.readlines()
        total = len(im_list)

    for count, info in enumerate(im_list):

        progressbar = "".join(["\033[41m%s\033[0m" % '   '] * int(count/total * 20))
        progressbar = "\r" + progressbar + f" {count}|{total}"
        print(progressbar, end = "", flush=True)

        # Read image info
        im_info, which_eye = info.strip().split(" ")
        day, im_name = im_info.split("/")
        im_number = int(im_name.split(".")[0])        

        # Read image annotation and image
        im_path = os.path.join(im_root, day, im_name)
        im = cv2.imread(im_path)
        annotation = anno_dict[im_info]
        annotation = AnnoDecode(annotation) 

        # Read face' box
        facebox = GetFaceBox(annotation)  
        leftbox, rightbox = GetEyeBox(annotation)  

        # Crop images
        face_img = gtools.CropImg(im, 
                      facebox.begin[0], 
                      facebox.begin[1], 
                      facebox.width, 
                      facebox.height) 
        face_img = cv2.resize(face_img, (224, 224))

        left_img = gtools.CropImg(im, 
                      leftbox.begin[0], 
                      leftbox.begin[1], 
                      leftbox.width, 
                      leftbox.height) 
        left_img = cv2.resize(left_img, (60, 36))

        right_img = gtools.CropImg(im, 
                      rightbox.begin[0], 
                      rightbox.begin[1], 
                      rightbox.width, 
                      rightbox.height) 
        right_img = cv2.resize(right_img, (60, 36))

        image = edict()
        image.width = screen.width_p
        image.height = screen.height_p
        grid_img = GetGrid(image, facebox)

        # Acquire essential info
        gaze = annotation.gazepoint
        headRot = dpc.HeadTo2d(annotation.headrotvectors)
        headtrans = annotation.headtransvectors
   
        # Save the acquired info
        cv2.imwrite(os.path.join(im_outpath, "face", str(count+1)+".jpg"), face_img)
        cv2.imwrite(os.path.join(im_outpath, "left", str(count+1)+".jpg"), left_img)
        cv2.imwrite(os.path.join(im_outpath, "right", str(count+1)+".jpg"), right_img)
        cv2.imwrite(os.path.join(im_outpath, "grid", str(count+1)+".jpg"), grid_img)
        
        save_name_face = os.path.join(person, "face", str(count+1) + ".jpg")
        save_name_left = os.path.join(person, "left", str(count+1) + ".jpg")
        save_name_right = os.path.join(person, "right", str(count+1) + ".jpg")
        save_name_grid = os.path.join(person, "grid", str(count+1) + ".jpg")

        save_origin = im_info
        save_flag = which_eye
        save_gaze = ",".join(gaze.astype("str"))
        save_headrot = ",".join(headRot.astype("str"))
        save_headtrans = ",".join(headtrans.astype("str"))
        save_ratio = ",".join([str(screen.width_r), str(screen.height_r)])
        save_face = ",".join(list(map(str, [facebox.begin[0], facebox.begin[1], facebox.begin[0]+facebox.width, facebox.begin[1]+facebox.height])))
        save_left = ",".join(list(map(str, [leftbox.begin[0], leftbox.begin[1], leftbox.begin[0]+leftbox.width, leftbox.begin[1]+leftbox.height])))
        save_right = ",".join(list(map(str, [rightbox.begin[0], rightbox.begin[1], rightbox.begin[0]+rightbox.width, rightbox.begin[1]+rightbox.height])))


        save_str = " ".join([save_name_face, save_name_left, save_name_right, save_name_grid, save_origin, save_flag, save_gaze, save_headrot, save_headtrans, save_ratio, save_face, save_left, save_right])
        
        outfile.write(save_str + "\n")
    print("")
    outfile.close()

def AnnoDecode(anno_info):
	annotation = np.array(anno_info).astype("float32")
	out = edict()
	out["gazepoint"] = annotation[0:2]
	out["left_left_corner"] = annotation[2:4]
	out["left_right_corner"] = annotation[4:6]
	out["right_left_corner"] = annotation[6:8]
	out["right_right_corner"] = annotation[8:10]
	out["bottom_left_corner"] = annotation[10:12]
	out["bottom_right_corner"] = annotation[12:14]
	out["headrotvectors"] = annotation[14:17]
	out["headtransvectors"] = annotation[17:20]
	return out

def GetFaceBox(out):
  face = edict()

  points = [out.left_left_corner, out.left_right_corner, out.right_left_corner, out.right_right_corner, out.bottom_left_corner, out.bottom_right_corner]
  points = np.array(points)
  face.center = np.mean(points, 0)
  width = np.abs(points[3,0] - points[0, 0])
  hight = np.max(points[:,1]) - np.min(points[:,0])
  length = np.max([width, hight])
  length = 1.5*length
  
  face.begin = (face.center - np.array([0.5*length, 0.5*length])).astype("int")
  face.width = int(length)
  face.height = int(length)
  return bound(face)

def GetEyeBox(out):
  points = [out.left_left_corner, out.left_right_corner, out.right_left_corner, out.right_right_corner, out.bottom_left_corner, out.bottom_right_corner]
  points = np.array(points)

  # left eyes
  left = edict()
  left.center = (points[0] + points[1])/2
  length = np.abs(points[1, 0] - points[0, 0])
  left.width = int(1.5 * length)
  times = left.width / 60
  left.height = int(36*times)
  left.begin = (left.center - np.array([0.5*left.width, 0.5*left.height])).astype("int")

  # left eyes
  right = edict()
  right.center = (points[2] + points[3])/2
  length = np.abs(points[2, 0] - points[3, 0])
  right.width = int(1.5 * length)
  times = right.width / 60
  right.height = int(36 * times)
  right.begin = (right.center - np.array([0.5*right.width, 0.5*right.height])).astype("int")
  return bound(left), bound(right)

def GetGrid(image, face):
  img = np.zeros((image.width, image.height))
  img[face.begin[0]:face.begin[0] + face.width, face.begin[1]:face.begin[1]+face.height] = np.ones([face.width,face.height])
  img = cv2.resize(img, (25, 25))
  return img

def bound(axis):
  image = edict()
  image.width = 1280
  image.height = 800
  axis.begin[0] = max(axis.begin[0], 0)
  axis.begin[1] = max(axis.begin[1], 0)

  if axis.begin[0] + axis.width > image.width:
    axis.begin[0] = image.width - axis.width

  if axis.begin[1] + axis.height > image.height:
    axis.begin[1] = image.height - axis.height
  return axis


if __name__ == "__main__":
    ImageProcessing_MPII()
