import cv2
from tracker import MyTracker

if __name__ == '__main__':
    cap = cv2.VideoCapture("./car.avi")
    tracker = MyTracker()
    ok, frame = cap.read()
    if not ok:
        print("error reading video")
        exit(-1)
    #roi = cv2.selectROI("tracking", frame, False, False)
    roi = (218, 302, 148, 108)
    tracker.init(frame, roi)
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
    cap.release()
    cv2.destroyAllWindows()

