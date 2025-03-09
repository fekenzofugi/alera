import dlib
import cv2
from skimage import io
import cv2
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import imutils

def find_landmarks(path, predictor_model="shape_predictor_68_face_landmarks.dat"):

    # Take the image file name from the command line
    file_name = path

    # Create a HOG face detector using the built-in dlib class
    face_detector = dlib.get_frontal_face_detector()
    face_pose_predictor = dlib.shape_predictor(predictor_model)

    # Load the image
    image = io.imread(file_name)
    # image = imutils.resize(image, width=800)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Run the HOG face detector on the image data
    detected_faces = face_detector(image_rgb, 1)

    print("Found {} faces in the image file {}".format(len(detected_faces), file_name))

    # Loop through each face we found in the image
    for i, face_rect in enumerate(detected_faces):
        # Detected faces are returned as an object with the coordinates 
        # of the top, left, right and bottom edges
        print("- Face #{} found at Left: {} Top: {} Right: {} Bottom: {}".format(i, face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom()))

        # Draw a box around each face we found
        cv2.rectangle(image_rgb, (face_rect.left(), face_rect.top()), (face_rect.right(), face_rect.bottom()), (0, 255, 0), 2)

        # Get the the face's pose
        pose_landmarks = face_pose_predictor(image_rgb, face_rect)

        # Draw the face landmarks on the screen.
        for point in pose_landmarks.parts():
            cv2.circle(image_rgb, (point.x, point.y), 2, (0, 0, 255), -1)
                
    # Display the image with landmarks
    cv2.imshow('Landmarks', image_rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def align_image(path, predictor_model):
    # Take the image file name from the command line
    file_name = path

    # Create a HOG face detector using the built-in dlib class
    face_detector = dlib.get_frontal_face_detector()
    face_pose_predictor = dlib.shape_predictor(predictor_model)

    # Load the image
    image = io.imread(file_name)
    image = imutils.resize(image, width=800)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    shape_predictor = dlib.shape_predictor(predictor_model)
    face_aligner = FaceAligner(shape_predictor, desiredFaceWidth=256)

    # Run the HOG face detector on the image data
    detected_faces = face_detector(image_rgb, 1)

    print("Found {} faces in the image file {}".format(len(detected_faces), file_name))
    face_aligned = None
    # Loop through each face we found in the image
    for i, face_rect in enumerate(detected_faces):

        (x, y, w, h) = rect_to_bb(face_rect)
        face_original = imutils.resize(image_rgb[y:y + h, x:x + w], width=256)
        face_aligned = face_aligner.align(image_rgb, image_gray, face_rect)

        # Detected faces are returned as an object with the coordinates 
        # of the top, left, right and bottom edges
        print("- Face #{} found at Left: {} Top: {} Right: {} Bottom: {}".format(i, face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom()))

        # Draw a box around each face we found
        cv2.rectangle(image_rgb, (face_rect.left(), face_rect.top()), (face_rect.right(), face_rect.bottom()), (0, 255, 0), 2)

        # Get the the face's pose
        pose_landmarks = face_pose_predictor(image_rgb, face_rect)

        # Draw the face landmarks on the screen.
        for point in pose_landmarks.parts():
            cv2.circle(image_rgb, (point.x, point.y), 2, (0, 0, 255), -1)
        # display the output images

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(cv2.cvtColor(face_original, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Original")
        axes[0].axis("off")

        axes[1].imshow(cv2.cvtColor(face_aligned, cv2.COLOR_BGR2RGB))
        axes[1].set_title("Aligned")
        axes[1].axis("off")

        plt.show()
        
    return face_aligned

#For dlib’s 68-point facial landmark detector:
FACIAL_LANDMARKS_IDXS = OrderedDict([
	("mouth", (48, 68)),
	("inner_mouth", (60, 68)),
	("right_eyebrow", (17, 22)),
	("left_eyebrow", (22, 27)),
	("right_eye", (36, 42)),
	("left_eye", (42, 48)),
	("nose", (27, 36)),
	("jaw", (0, 17))
])

class FaceAligner:
    def __init__(self, predictor, desiredLeftEye=(0.35, 0.35),
        desiredFaceWidth=256, desiredFaceHeight=None):
        # store the facial landmark predictor, desired output left
        # eye position, and desired output face width + height
        self.predictor = predictor
        self.desiredLeftEye = desiredLeftEye
        self.desiredFaceWidth = desiredFaceWidth
        self.desiredFaceHeight = desiredFaceHeight
        # if the desired face height is None, set it to be the
        # desired face width (normal behavior)
        if self.desiredFaceHeight is None:
            self.desiredFaceHeight = self.desiredFaceWidth

    def align(self, image, gray, rect):
        # convert the landmark (x, y)-coordinates to a NumPy array
        shape = self.predictor(gray, rect)
        shape = shape_to_np(shape)
        # extract the left and right eye (x, y)-coordinates
        (lStart, lEnd) = FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = FACIAL_LANDMARKS_IDXS["right_eye"]
        leftEyePts = shape[lStart:lEnd]
        rightEyePts = shape[rStart:rEnd]

        # compute the center of mass for each eye
        leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
        rightEyeCenter = rightEyePts.mean(axis=0).astype("int")
        # compute the angle between the eye centroids
        dY = rightEyeCenter[1] - leftEyeCenter[1]
        dX = rightEyeCenter[0] - leftEyeCenter[0]
        angle = np.degrees(np.arctan2(dY, dX)) - 180

        # compute the desired right eye x-coordinate based on the
        # desired x-coordinate of the left eye
        desiredRightEyeX = 1.0 - self.desiredLeftEye[0]
        # determine the scale of the new resulting image by taking
        # the ratio of the distance between eyes in the *current*
        # image to the ratio of distance between eyes in the
        # *desired* image
        dist = np.sqrt((dX ** 2) + (dY ** 2))
        desiredDist = (desiredRightEyeX - self.desiredLeftEye[0])
        desiredDist *= self.desiredFaceWidth
        scale = desiredDist / dist

        # compute center (x, y)-coordinates (i.e., the median point)
        # between the two eyes in the input image
        eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2.0,
            (leftEyeCenter[1] + rightEyeCenter[1]) // 2.0)
        # grab the rotation matrix for rotating and scaling the face
        M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)
        # update the translation component of the matrix
        tX = self.desiredFaceWidth * 0.5
        tY = self.desiredFaceHeight * self.desiredLeftEye[1]
        M[0, 2] += (tX - eyesCenter[0])
        M[1, 2] += (tY - eyesCenter[1])

        # apply the affine transformation
        (w, h) = (self.desiredFaceWidth, self.desiredFaceHeight)
        output = cv2.warpAffine(image, M, (w, h),
            flags=cv2.INTER_CUBIC)
        # return the aligned face
        return output


def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((shape.num_parts, 2), dtype=dtype)

	# loop over all facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, shape.num_parts):
		coords[i] = (shape.part(i).x, shape.part(i).y)

	# return the list of (x, y)-coordinates
	return coords

def rect_to_bb(rect):
	# take a bounding predicted by dlib and convert it
	# to the format (x, y, w, h) as we would normally do
	# with OpenCV
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y

	# return a tuple of (x, y, w, h)
	return (x, y, w, h)