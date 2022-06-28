import numpy as np
import cv2
# Para os dockers
import grpc
# Code (my code)
import sys
import open_pose_pb2 as pb
import open_pose_pb2_grpc as pb_grpc
import tqdm.auto
import typing
from scipy.io import savemat
from concurrent import futures


def read_cap(cap):
    """Returns a generator over the capture with indexed frames.
    
    The generator can be used in a for cycle:
    for idx, frame in read_cap(...):
        ...
        
    Or it can be collected into a list:
    frames = [(idx, frame) for idx, frame in read_cap(...)]
    ...
    """
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print(str(frame_count)+'frames')
            break
        
        yield frame_count, frame
        frame_count += 1
    cap.release()

class KeypointInfo(typing.TypedDict):
    x: float
    y: float
    score: float


KeypointId = int

Keypoint = typing.Dict[KeypointId, KeypointInfo]

DetectedPose = typing.List[Keypoint]

DetectedPoses = typing.List[DetectedPose]

class OpenPoseClient:
    """Detects poses by calling an OpenPose grpc server.
    
    This class should be used with Python's with block, as follows:
    
    with OpenPoseClient(address=<address>) as openpose:
        ...
    
    This allows the channel to be released when it is no longer used.
    To achieve this, tha class implements the __enter__ and __exit__ methods.

    Args:
        address: address where the server is running.
    """

    def __init__(self, address: str):
        self.__address = address
        self.__channel = None
        self.__stub = None
    
    def __enter__(self):
        self.__channel = grpc.insecure_channel(self.__address)
        self.__stub = pb_grpc.OpenPoseEstimatorStub(self.__channel)
        return self
    
    def __exit__(self, exception_type, exception_value, traceback):
        self.__channel.close()

    def get_keypoints(self, *images: bytes) -> typing.List[DetectedPoses]:
        """Detects poses calling an OpenPose grpc server.

        Args:
            images: List of images as bytes.

        Returns: Detected poses for each image. Returns a List where each element is
        another list with the detected poses for the respective image. Each detected
        pose is a dictionary with the keypoint id as key. The value for each key is a
        dictionary with the keys "x", "y" and "score", for the relative x coordinate
        (between 0 and 1), relative y coordinate (between 0 and 1) and a confidence
        score for the system.
        """
        if not self.__stub:
            raise ValueError("This class should be used in a 'with' block")
        requests = [pb.Image(data=image) for image in images]
        replies = [self.__stub.estimate(request) for request in requests]
        return [self.reply_pb_to_dicts(reply) for reply in replies]

    @staticmethod
    def reply_pb_to_dicts(reply: pb.DetectedPoses) -> DetectedPoses:
        poses: DetectedPoses = []
        for pb_pose in reply.poses:
            pose: DetectedPose = {}
            for pb_kp in pb_pose.key_points:
                pose[pb_kp.index] = {
                    "x": pb_kp.x,
                    "y": pb_kp.y,
                    "score": pb_kp.score,
                }
            poses.append(pose)
        return poses
    
def predict_poses(openpose, image):
    encoded = cv2.imencode('.jpg', image)[1]
    return openpose.get_keypoints(encoded.tobytes())


def pose_to_array(idx, pose):
    """Constructs an array representation for a pose.
    
    The representation is as follows:
    
    [frame_idx]
    [   x_0   ]
    [   y_0   ]
    [   s_0   ]
    [   ...   ]
    [   x_18  ]
    [   y_18  ]
    [   s_18  ]
    
    where x_{i}, y_{i} and s_{i} are the x coordinate, the
    y coordinate and the confidence score for the keypoint i.
    
    Returns: A numpy.array of shape (55, 1) with the described 
    representation.
    """
    arr = np.zeros((1 + 18 * 3,))
    arr[0] = idx
    for kp in pose:
        # First entry is the frame index
        # Keypoint 0 goes to the positions 2, 3, 4
        # Keypoint 1 goes to the positions 4, 5, 6
        # Keypoint 2 goes to the positions 7, 8, 9
        # ...
        # Keypoint 18 goes to 52, 53, 54
        kp_idx = kp * 3 + 1
        arr[kp_idx] = pose[kp]["x"]
        arr[kp_idx+1] = pose[kp]["y"]
        arr[kp_idx+2] = pose[kp]["score"]
    return arr.reshape(-1, 1)

def videoToPose(video_path, out_path):
    _NUM_THREADS = 1
    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name
    cap = cv2.VideoCapture(video_path)
    indexed_frames = read_cap(cap)
    # Automatically connects and disconnects to avoid too many open connections
    with OpenPoseClient(address='openpose1:8061') as openpose:
        
        def compute_and_build_arrays(args):
            index, frame = args
            # Define inner function to use openpose client here
            poses = predict_poses(openpose, frame)[0]
            # No poses detected.
            if len(poses) == 0:
                return None
            arrays = [pose_to_array(index, p) for p in poses]
            return np.hstack(arrays)
        
        with futures.ThreadPoolExecutor(max_workers = _NUM_THREADS) as executor:
            frames_with_indexes = [f for f in read_cap(cap)]
            frames_matrices = executor.map(compute_and_build_arrays, frames_with_indexes)
            frames_matrices_with_progress = tqdm.auto.tqdm(frames_matrices, total=len(frames_with_indexes))
            # Some frames don't have poses. We can filter them.
            matrix = np.hstack([f for f in frames_matrices_with_progress if f is not None])
            skel={"skeldata":matrix}
            savemat(out_path,skel)

videoToPose('../datasets/EurosportCut/girosmallslow_cut.mp4', 'out.mat')