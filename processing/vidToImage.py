def process_video(video_path, output_folder):
    import cv2, os, math

    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(5)

    count = 0

    while cap.isOpened():
        frame_id = cap.get(1)
        ret, frame = cap.read()

        if not ret:
            break

        if frame_id % math.floor(frame_rate) == 0:
            width = frame.shape[1]

            if width < 300:
                scale_ratio = 2
            elif width > 1900:
                scale_ratio = 0.33
            elif width > 1000:
                scale_ratio = 0.5
            else:
                scale_ratio = 1

            new_frame = cv2.resize(
                frame,
                (int(width * scale_ratio), int(frame.shape[0] * scale_ratio))
            )

            filename = os.path.join(output_folder, f"frame_{count}.png")
            cv2.imwrite(filename, new_frame)
            count += 1

    cap.release()
    return output_folder