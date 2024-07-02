import cv2
import os
import shutil
import random

# Skip frames interval for each video
skip = [1, 3, 2, 2, 6, 2]

def convert_video_to_imgs():
    """
    This function processes video files, extracts frames at specific intervals,
    saves the frames in different orientations, and stores them in designated folders.
    """
    # Loop through each video file
    for i in range(6):
        video_file = 'videos/' + str(i) + '.mkv'
        output_folder = 'train/' + str(i) + '/'

        # Ensure the output folder exists
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Open the video file
        cap = cv2.VideoCapture(video_file)

        frame_count = 0
        saved_frame_count = 0

        # Check if the video was successfully opened
        if not cap.isOpened():
            print("Error: Could not open video file.")
        else:
            while True:
                # Read a video frame
                ret, frame = cap.read()

                # If the frame was successfully read, ret is True, continue processing the frame
                if ret:
                    # Only read every k-th frame
                    if frame_count % skip[i] == 0:
                        # Define output filename template
                        base_filename = os.path.join(output_folder, f"{saved_frame_count:06d}")

                        # Save the original frame
                        original_file = f"{base_filename}_000.png"
                        cv2.imwrite(original_file, frame)
                        print(f"Saved frame {saved_frame_count:06d}_000 to {original_file}")

                        # Rotate 90 degrees
                        rotated_90 = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                        rotated_90_file = f"{base_filename}_090.png"
                        cv2.imwrite(rotated_90_file, rotated_90)
                        print(f"Saved frame {saved_frame_count:06d}_090 to {rotated_90_file}")

                        # Rotate 180 degrees
                        rotated_180 = cv2.rotate(frame, cv2.ROTATE_180)
                        rotated_180_file = f"{base_filename}_180.png"
                        cv2.imwrite(rotated_180_file, rotated_180)
                        print(f"Saved frame {saved_frame_count:06d}_180 to {rotated_180_file}")

                        # Rotate 270 degrees
                        rotated_270 = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                        rotated_270_file = f"{base_filename}_270.png"
                        cv2.imwrite(rotated_270_file, rotated_270)
                        print(f"Saved frame {saved_frame_count:06d}_270 to {rotated_270_file}")

                        # Increase saved frame count
                        saved_frame_count += 1

                    # Increase total frame count
                    frame_count += 1
                else:
                    # If no more frames can be read, exit the loop
                    break

        # Release the video capture object
        cap.release()

        print(f"All frames of Video {i:01d} have been saved.")

def move_random_files(source_dir, destination_dir, proportion=0.1):
    """
    Randomly select a proportion of files from source_dir and move them to destination_dir.

    Parameters:
    source_dir: Source directory path
    destination_dir: Destination directory path
    proportion: Proportion of files to move (float between 0 and 1)
    """
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # Get all files in the source directory
    files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]

    # Calculate the number of files to move
    num_files_to_move = int(len(files) * proportion)

    # Randomly select files
    files_to_move = random.sample(files, num_files_to_move)

    # Move files
    for file in files_to_move:
        src_file_path = os.path.join(source_dir, file)
        dest_file_path = os.path.join(destination_dir, file)
        shutil.move(src_file_path, dest_file_path)
        print(f"Moved {file} to {destination_dir}")

    print(f"Moved {len(files_to_move)} files from {source_dir} to {destination_dir}")

if __name__ == "__main__":
    # convert_video_to_imgs()

    # Example usage
    for i in range(6):
        train = 'train/' + str(i) + '/'
        dev = 'dev/' + str(i) + '/'
        test = 'test/' + str(i) + '/'

        move_random_files(train, dev)
        move_random_files(train, test)
