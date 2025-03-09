import os
import cv2
import uuid

def get_data(path):
    # Create directories
    os.makedirs(path, exist_ok=True)

    # Establish a connection to the webcam
    cap = cv2.VideoCapture(0)
    while cap.isOpened(): 
        ret, frame = cap.read()
    
        # Cut down frame to 250x250px
        # frame = frame[120:120+250,200:200+250, :]
        
        # Collect images 
        if cv2.waitKey(1) & 0XFF == ord('p'):
            # Create the unique file path 
            imgname = os.path.join(path, '{}.jpg'.format(uuid.uuid1()))
            # Write out anchor image
            cv2.imwrite(imgname, frame)
        
        # Show image back to screen
        cv2.imshow('Image Collection', frame)
        
        # Breaking gracefully
        if cv2.waitKey(1) & 0XFF == ord('q'):
            break

        # Print the number of jpg files in each directory
        num_images = len(os.listdir(path))
        print(f'Number of images: {num_images}')
            
    # Release the webcam
    cap.release()
    # Close the image show frame
    cv2.destroyAllWindows()

def get_files_info(directory):
    """
    Get information about files in a specified directory.
    This function ensures the specified directory exists, retrieves a list of all files
    in the directory, and returns the total number of files along with their names.
    Args:
        directory (str): The path to the directory to inspect.
    Returns:
        tuple: A tuple containing:
            - num_files (int): The total number of files in the directory.
            - img_ids (list of str): A list of file names in the directory.
    """
    # Ensure the directory exists
    os.makedirs(directory, exist_ok=True)

    # Get the list of all files and directories
    file_list = os.listdir(directory)

    # Filter only files
    file_list = [file for file in file_list if os.path.isfile(os.path.join(directory, file))]

    # Get the number of files
    num_files = len(file_list)

    img_ids = file_list  # Directly assign the list of file names
    
    return num_files, img_ids

if __name__ == "__main__":
    data_path = "images"
    user_path = "images/elidia"
    os.makedirs(data_path, exist_ok=True)

    get_data(user_path)