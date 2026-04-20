import os
import sys
import cv2


def get_filename_only(file_path):
    return os.path.splitext(os.path.basename(file_path))[0]


def process_video(video_path, frames_base_path):
    """Convert a single video to sampled PNG frames."""
    try:
        if not os.path.isfile(video_path):
            print(f"Error: Video file not found: {video_path}")
            return False

        frames_base_path = os.path.abspath(frames_base_path)
        if not frames_base_path:
            print("Error: No frames output directory provided.")
            return False

        os.makedirs(frames_base_path, exist_ok=True)

        video_name = get_filename_only(video_path)
        if not video_name:
            print(f"Error: Unable to determine video name from path: {video_path}")
            return False

        output_dir = os.path.join(frames_base_path, video_name)
        os.makedirs(output_dir, exist_ok=True)
        print(f"Creating directory: {output_dir}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video {video_path}")
            return False

        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            sample_every = max(1, total_frames // 8) if total_frames > 0 else 1
            count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                if frame_id % sample_every != 0 or count >= 8:
                    continue

                h, w = frame.shape[:2]
                if w < 300:
                    scale = 2
                elif w > 1900:
                    scale = 0.33
                elif 1000 < w <= 1900:
                    scale = 0.5
                else:
                    scale = 1

                new_w = max(1, int(w * scale))
                new_h = max(1, int(h * scale))
                new_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

                out_filename = f"{video_name}-{count:03d}.png"
                out_path = os.path.join(output_dir, out_filename)
                if not cv2.imwrite(out_path, new_frame):
                    print(f"Warning: Failed to write frame {out_path}")
                    continue

                count += 1
        finally:
            cap.release()

        print(f"Done: {video_path} -> {count} frames saved to {output_dir}")
        return count > 0
    except Exception as e:
        print(f"Error while processing video '{video_path}': {e}")
        return False


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python 00-convert_video_to_image.py <video_path> <output_frames_dir>")
        sys.exit(1)

    try:
        video_path = sys.argv[1]
        frames_base_path = sys.argv[2]
        success = process_video(video_path, frames_base_path)
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)
