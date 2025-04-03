from skimage.feature import hog
from skimage import exposure
import cv2
from facenet_pytorch import MTCNN
import torch

def histogram_of_oriented_gradients(path, plot=False):
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fd, hog_image = hog(gray, orientations=8, pixels_per_cell=(16, 16),
                        cells_per_block=(1, 1), visualize=True)
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    if plot:
        cv2.imshow('HOG Image', hog_image_rescaled)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return hog_image_rescaled

# import dlib
def detect_faces(path):
    pass
#     # Take the image file name from the command line
#     file_name = path

#     # Create a HOG face detector using the built-in dlib class
#     face_detector = dlib.get_frontal_face_detector()

#     # Load the image into an array
#     image = cv2.imread(file_name)

#     # Run the HOG face detector on the image data.
#     # The result will be the bounding boxes of the faces in our image.
#     detected_faces = face_detector(image, 1)

#     print("I found {} faces in the file {}".format(len(detected_faces), file_name))

#     # Loop through each face we found in the image
#     for i, face_rect in enumerate(detected_faces):
#         # Detected faces are returned as an object with the coordinates 
#         # of the top, left, right and bottom edges
#         print("- Face #{} found at Left: {} Top: {} Right: {} Bottom: {}".format(i, face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom()))

#         # Draw a box around each face we found
#         cv2.rectangle(image, (face_rect.left(), face_rect.top()), (face_rect.right(), face_rect.bottom()), (0, 255, 0), 2)

#     # Display the image with the detected faces
#     cv2.imshow('Detected Faces', image)

#     # Wait until the user hits <enter> to close the window
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

def detect_faces_webcam():
    pass
#     # Open a connection to the webcam
#     video_capture = cv2.VideoCapture(0)

#     # Create a HOG face detector using the built-in dlib class
#     face_detector = dlib.get_frontal_face_detector()

#     while True:
#         # Capture frame-by-frame
#         ret, frame = video_capture.read()

#         # Convert the frame to grayscale
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#         # Run the HOG face detector on the grayscale frame
#         detected_faces = face_detector(gray, 1)

#         # Loop through each face we found in the frame
#         for i, face_rect in enumerate(detected_faces):
#             # Draw a box around each face we found
#             cv2.rectangle(frame, (face_rect.left(), face_rect.top()), (face_rect.right(), face_rect.bottom()), (0, 255, 0), 2)

#         # Display the resulting frame
#         cv2.imshow('Video', frame)

#         # Break the loop on 'q' key press
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     # Release the capture and close any OpenCV windows
#     video_capture.release()
#     cv2.destroyAllWindows()


def detect_faces_mtcnn(dataset, loader):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))

    # Create an MTCNN face detector
    mtcnn = MTCNN(
        image_size=160, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        device=device
    )

    aligned = []
    names = []
    for x, y in loader:
        x_aligned, prob = mtcnn(x, return_prob=True)
        if x_aligned is not None:
            print('Face detected with probability: {:8f}'.format(prob))
            aligned.append(x_aligned)
            names.append(dataset.idx_to_class[y])

    return aligned, names

def detect_single_face_mtcnn(image_path):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))

    mtcnn = MTCNN(
        image_size=160, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        device=device
    )
    image = cv2.imread(image_path)
    x_aligned, prob = mtcnn(image, return_prob=True)
    if x_aligned is not None:
        print('Face detected with probability: {:8f}'.format(prob))

    return x_aligned