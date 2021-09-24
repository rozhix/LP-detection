import cv2
import numpy as np
import argparse
obj_classes = []


def open_obj_file(obj_file):
    with open(obj_file, 'rt') as f:
        obj_classes = f.read().rstrip('\n').split('\n')
    return obj_classes


def make_darknet(net_config, net_weights):
    """make darknet framework, use opencv for backend and cpu for target"""
    net = cv2.dnn.readNetFromDarknet(net_config, net_weights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net

def find_license_plate(output, img):
    obj_classes = open_obj_file("obj.names")
    img_h, img_w, img_c = img.shape
    bboxes = []
    class_ids = []
    confidences = []

    confidence_threshold = 0.25
    nms_threshold = 0.3
    # output[0].shape = (300, 7)
    # (320*320)/(32*32)= 10*10 --> (RGB) 10*10*3 = 300
    # output[1].shape = (1200, 7)
    # (320*320)/(16*16)= 20*20 --> (RGB) 20*20*3 = 1200
    # output[2].shape = (4800, 7)
    # (320*320)/(8*8)= 40*40 --> (RGB) 40*40*3 = 4800
    for cell in output:
        for detect_vector in cell:
            scores = detect_vector[5:]
            #find the index of the class with the highest probability
            class_id = np.argmax(scores)
            #find the probability
            confidence = scores[class_id]
            if confidence > confidence_threshold:
                #find w,h,x,y coordinates
                w,h = int(detect_vector[2] * img_w), int(detect_vector[3] * img_h)
                x,y = int((detect_vector[0] * img_w) - w/2), int((detect_vector[1] * img_h) - h/2)
                bboxes.append([x,y,w,h])
                class_ids.append(class_id)
                confidences.append(float(confidence))
    #choose the best box with non max suppression
    indices = cv2.dnn.NMSBoxes(bboxes, confidences, confidence_threshold, nms_threshold)

    for i in indices:
        i = i[0]
        bbox = bboxes[i]
        x,y,w,h = bbox[0], bbox[1], bbox[2], bbox[3]
        #draw the bounding box
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(img, f'{obj_classes[class_ids[i]].upper()} {int(confidences[i] * 100)}%',
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        return img

def main(img_path):
    net_config = "yolov3_training.cfg"
    net_weights = "yolov3_training_final.weights"
    blob_size = 320

    net = make_darknet(net_config, net_weights)
    frame = cv2.imread(img_path)
    #blobFromImage creates 4-dimensional blob from image (1, 3, 320, 320)
    #blob = binary large object
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1 / 255, size=(blob_size, blob_size), mean=(0, 0, 0)
                                 , swapRB=True, crop=False)
    net.setInput(blob)
    #names of layers with unconnected outputs
    out_names = net.getUnconnectedOutLayersNames()
    #runs forward pass to compute output of layer
    output = net.forward(out_names)
    frame = find_license_plate(output, frame)

    cv2.imshow('photo', frame)
    cv2.waitKey(0)


if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="image path")
    args = vars(ap.parse_args())

    main(args['image'])





