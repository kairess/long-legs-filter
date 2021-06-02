import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

video_path = 'videos/04.mp4'
cap = cv2.VideoCapture(video_path)

W, H = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out = cv2.VideoWriter(f'{video_path}_input.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (W, H))
out2 = cv2.VideoWriter(f'{video_path}_output.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (W, H))

with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:

    leg_ys = []

    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        output = img.copy()

        if results.pose_landmarks:
            leg_y_now = (results.pose_landmarks.landmark[23].y + results.pose_landmarks.landmark[24].y) / 2 * H
            leg_ys.append(leg_y_now)

            leg_y_avg = int(leg_y_now)
            if len(leg_ys) > 10:
                leg_y_avg = int(sum(leg_ys[-10:]) / 10)

            img_long = cv2.resize(img[leg_y_avg:], dsize=None, fx=1.0, fy=1.5)
            img_long = cv2.resize(img_long[:int(img_long.shape[0] * 0.75)], dsize=(W, H - leg_y_avg))

            output[leg_y_avg:] = img_long

            mp_drawing.draw_landmarks(
                img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow('img', img)
        cv2.imshow('output', output)
        out.write(img)
        out2.write(output)
        if cv2.waitKey(1) == ord('q'):
            break

out.release()
cap.release()
