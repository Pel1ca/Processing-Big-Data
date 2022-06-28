# Big Data Processing Project

## Objective

Get videos from Giro Italia
Classify frames of those videos

## Inputs

* Embedings of the video frames
* Skeleton data of the frame (might be missing)

## Output

* List classifying every frame

## The code

videoToSkel.py => gets skeleton data of a video
videoToFeatures.py => gets features data of a video
scrapsVideos.py => web scrapes the internet for more videos, runs in the background
classifyVideo.py => gets the classification of a video
showVideo.py => shows a video with the classification of every frame (needs CV2)

### DEMO

1. Place a folder with skeleton data and feature data in the dataset folder
2. docker compose up -d
3. docker compose exec jupyterpbdjpc python3 dataAnalysis/classifyVideo.py datasets/EurosportCut/girosmallslow_cut.mp4_features.mat datasets/EurosportCut/esqueletosmallslow_cut.mat
4. python .\showVideo.py .\datasets\Classifications\videoClass.cl .\datasets\EurosportCut\girosmallslow_cut.mp4
5. Press "q" when you are pleased :)