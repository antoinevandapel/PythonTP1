import cv2
import numpy as np


# retourne la moyenne du carré des différences
def get_error_distance(im1, im2):
    err = np.sum((im1.astype("float") - im2.astype("float")) ** 2)
    err /= float(im1.shape[0] * im2.shape[1])
    return err


def save_webcam(out, fps, mirror=False):
    # Capturing video from webcam:
    cap = cv2.VideoCapture(0)
    current_frame = 0
    # Get current width of frame
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    # Get current height of frame
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    # Define the codec and create VideoWriter object
    force = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(out, force, fps, (int(width), int(height)))
    _, previous_frame = cap.read()
    previous_frame = cv2.flip(previous_frame, 1)
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            if mirror:
                # Mirror the output video frame
                frame = cv2.flip(frame, 1)
            # Saves for video
            out.write(frame)
            sim = get_error_distance(frame, previous_frame)
            if sim > 25:
                print("Vous avez bougé : " + sim.__str__())
            previous_frame = frame
            # Display the resulting frame
            cv2.imshow('frame', frame)
        else:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):  # if 'q' is pressed then quit
            break
        # To stop duplicate images
        current_frame += 1
    # When everything done, release the capture
    cap.release()
    out.release()
    cv2.destroyAllWindows()


def main():
    save_webcam('output.avi', 30.0, mirror=True)


if __name__ == '__main__':
    main()