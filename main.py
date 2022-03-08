import cv2
from tracker import MyTracker
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Object Tracking')
    parser.add_argument('--feature', metavar='-F', type=str, default='resnet18',
                    help='Which feature is gonna use')
    parser.add_argument('--kernel', metavar='-K', type=str, default='linear',
                    help='Which feature is gonna use')
    args = parser.parse_args()
    cap = cv2.VideoCapture("./car.avi")
    tracker = MyTracker(model_name=args.feature, kernel=args.kernel)
    ok, frame = cap.read()
    if not ok:
        print("error reading video")
        exit(-1)
    #roi = cv2.selectROI("tracking", frame, False, False)
    roi = (218, 302, 148, 108)
    tracker.init(frame, roi)
    count = 0
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        x, y, w, h = tracker.update(frame)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 1)
        cv2.imshow('tracking', frame)
        c = cv2.waitKey(1) & 0xFF
        if c==27 or c==ord('q'):
            break
        if count >= 4:
            tracker.init(frame, [x, y, w, h])
            count = 0
        count += 1

    cap.release()
    cv2.destroyAllWindows()

