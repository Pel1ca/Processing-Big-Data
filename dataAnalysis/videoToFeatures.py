import os
import cv2
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np


def v2f(input, output):

    cap = cv2.VideoCapture()
    cap.open(input)
    if not cap.isOpened():
        return 1

    frameCount = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    frameId = 0

    while frameId < frameCount:
        ret, frame = cap.read()
        # print frameId, ret, frame.shape
        if not ret:
            print("Failed to get the frame {f}".format(f=frameId))
            continue

        fname = "frame_" + str(frameId) + ".jpg"
        ofname = os.path.join(output, fname)
        ret = cv2.imwrite(ofname, frame)
        if not ret:
            print("Failed to write the frame {f}".format(f=frameId))
            continue

        frameId += int(1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frameId)

    return frameCount

def getFeatures(output_path,frameCount):
   
    features = np.array([[],[]])

    for i in range(frameCount):
        # get image
        img = Image.open('frame_'+i+'.jpg')
    # plt.imshow(img)
    # plt.show()

        # get pretrained model
        fullmodel = models.vgg16(pretrained=True)
        model = fullmodel.features

        # prepare image and model for inference
        img_transforms = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        img = img_transforms(img).unsqueeze(0)
        if torch.cuda.is_available():
            model = model.cuda()
            img = img.cuda()
        model.eval()

        # inference
        with torch.no_grad():
            features[i] = torch.flatten(model(img)).cpu().numpy()
    
    return features
