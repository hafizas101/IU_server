#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import the necessary packages
from imutils import paths
import face_recognition, cv2, os, time, pickle
import numpy as np
import pandas as pd
from keras.models import load_model
from utils.datasets import get_labels
from utils.preprocessor import preprocess_input

current_path = os.getcwd()

# Generate output directory
faces_path = os.path.join(current_path, "faces")
if not os.path.exists(faces_path):
    os.mkdir(faces_path)
result_dir = os.path.join(current_path, "results")
dataset_path = os.path.join(current_path, "dataset")
encodings_path = os.path.join(current_path, "encodings.pickle")
input_video_path = os.path.join(current_path, "inp.mp4")
output_video_path = os.path.join(current_path, "output.mp4")
emotion_model_path = os.path.join(current_path, "emotion_model.hdf5")
output_FPS = 3

emotion_code_mapping = {0:'angry',1:'disgust',2:'fear',3:'happy', 4:'sad',5:'surprise',6:'neutral'}
# Pre-trained YOLOv3 network configuration and weight file paths for face recognition
face_rec_cfg = os.path.join(current_path, "yolo_files/yolov3-face.cfg")
face_rec_weights = os.path.join(current_path, "yolo_files/yolov3-wider_16000.weights")
net = cv2.dnn.readNetFromDarknet(face_rec_cfg, face_rec_weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


boundary_color = (0, 255, 0)   # (Blue, Green, Red)
boundary_thickness = 2
font_color = (0, 255, 0)       # (Blue, Green, Red)
font_size = 0.8
font_thickness = 2

"""
num is a very important variable. It controls how much frames per second of the video we are extracting so
basically it is very computationally expensive and time consuming for long videos to perform frame by frame
detection and processing. Hence we can periodically skip some frames. This is controlled by num variable. Setting
num=1 would mean extract all frames. num = 2 means skip 1 frame and then extract the second and so on. num = 4
would mean skip 3 frames and then extract 1 and then again skip 3 and so on.
"""
num = 2
tolerance_threshold = 0.65
size=64


# In[2]:


def convert_frames_to_video(frames, output_FPS):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
    height, width, channels = frames[0].shape
    out = cv2.VideoWriter(output_video_path, fourcc, output_FPS, (width, height))
    for i, ff in enumerate (frames):
        out.write(ff)
    out.release
    cv2.destroyAllWindows()
    return

def get_outputs_names(net):
    # Get the names of all the layers in the network
    layers_names = net.getLayerNames()

    # Get the names of the output layers, i.e. the layers with unconnected
    # outputs
    return [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]


def detect_faces(image):
    conf_threshold = 0.5
    nms_threshold = 0.4
    frame_height = image.shape[0]
    frame_width = image.shape[1]
    IMG_WIDTH = 416
    IMG_HEIGHT = 416
    blob = cv2.dnn.blobFromImage(image, 1 / 255, (IMG_WIDTH, IMG_HEIGHT),[0, 0, 0], 1, crop=False)
    net.setInput(blob)
    outs = net.forward(get_outputs_names(net))
    # Scan through all the bounding boxes output from the network and keep only the ones with high confidence scores. Assign the box's class label as the
    # class with the highest score.
    confidences = []
    boxes = []
    final_boxes = []
    faces = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x = int(detection[0] * frame_width)
                center_y = int(detection[1] * frame_height)
                width = int(detection[2] * frame_width)
                height = int(detection[3] * frame_height)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant
    # overlapping boxes with lower confidences.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold,
                               nms_threshold)
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        right = left + box[2]
        bottom = top + box[3]
        final_boxes.append((top, right, bottom, left))
    return final_boxes


# # Mod-001 ==> Video Splitting in frames

# In[3]:


cap = cv2.VideoCapture(input_video_path)
FPS = round(cap.get(cv2.CAP_PROP_FPS))
print("Frames Per Second: "+str(FPS))

frames = []
start = time.time()
count = 0
frame_originals = []
while (cap.isOpened()):
    ret, frame = cap.read()
    count = count + 1
    if not ret:
        break
    if count % num == 0:
#         frame = frame[100:700, 100:700]
        frames.append(frame)
#         cv2.imwrite(name+".png", frame)
#         cv2.waitKey(0)
end = time.time()
print("Total number of frames read: "+str(len(frames)))
print("Time taken in reading the frames: {} seconds".format(end-start))
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


# # Mod-002 ==> Face detection per frame

# In[5]:


data = pickle.loads(open(encodings_path, "rb").read())
labelled = []
emotion_classifier = load_model(emotion_model_path)
emotion_target_size = emotion_classifier.input_shape[1:3]
for i, ff in enumerate(frames):
    image = ff.copy()
#     image = cv2.imread(ff, 1)
    image = cv2.resize(image, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_AREA)
#     print(image.shape)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes = detect_faces(image)
    encodings = face_recognition.face_encodings(rgb, boxes)

    # initialize the list of names for each face detected
    names = []
    labels = []
    bbox = []
    scores = []
    emotions = []
    # loop over the facial embeddings
    for kk, encoding in enumerate(encodings):
        top, right, bottom, left = boxes[kk]
        face = image[top:bottom, left:right]
        gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        gray_face = cv2.resize(gray_face, (emotion_target_size))
        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        emotion_prediction = emotion_classifier.predict(gray_face)
#         emotion_probability = np.max(emotion_prediction)
        emotion_label_arg = np.argmax(emotion_prediction)
        
        emotion = emotion_code_mapping[emotion_label_arg]
        
        matches = face_recognition.compare_faces(data["encodings"], encoding, tolerance=tolerance_threshold)
        dist = face_recognition.face_distance(data["encodings"], encoding)
        name = "Unknown"
        
        best_match_index = np.argmin(dist)
        if matches[best_match_index]:
            name = data["names"][best_match_index]
        names.append(name)

        if name!="Unknown":
            dd = (dist - np.amin(dist))/(np.amax(dist) - np.amin(dist))
            s = 1-np.mean(dd)
            s = "{0:1.3f}".format(s)
            labels.append(name)
            scores.append(s)
            bbox.append([left, top, right, bottom])

        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        txt = name+' : '+str(emotion)
        cv2.putText(image, txt, (left, y), cv2.FONT_HERSHEY_SIMPLEX,0.75, (0, 255, 0), 2)
    labelled.append(image)
#     cv2.imshow("Image", image)
#     cv2.waitKey(0)
    
cv2.destroyAllWindows() 


# In[ ]:





# In[ ]:





# In[ ]:


convert_frames_to_video(labelled, output_FPS)


# In[ ]:




