import cv2
import numpy as np

def readVideo(
    videoFile,
    consecutive_frame_index=None,
    selected_frames=None,
    frame_fps=None,
    num_frames=None,
    interval="full",
):
    cap = cv2.VideoCapture(str(videoFile))

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fcount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_fps == None:
        ratio = 1
    else:
        if frame_fps > fps:
            raise Exception(
                f"frame_fps ({frame_fps}) needs to be lower than video fps ({fps})"
            )

        ratio = round(fps / frame_fps)
    f_counter = 0

    if interval == "selected":
        allowed_frame_indices = np.array(selected_frames)
        allowed_frame_indices = allowed_frame_indices[::ratio]
        num_frames = len(allowed_frame_indices)
    else:
        if num_frames is not None:
            if interval == "full":
                allowed_frame_indices = np.arange(0, fcount, ratio)
                allowed_frame_indices = [
                    x[0]
                    for x in np.array_split(allowed_frame_indices, num_frames)
                    if len(x) > 0
                ]
                if len(allowed_frame_indices) < num_frames:
                    allowed_frame_indices = np.concatenate(
                        [
                            allowed_frame_indices,
                            np.array([-1] * (num_frames - len(allowed_frame_indices))),
                        ]
                    )
            elif interval == "consecutive":
                allowed_frame_indices = np.arange(0, fcount, ratio)

                temp_num_frames = num_frames
                index = consecutive_frame_index
                if consecutive_frame_index < 0:
                    temp_num_frames = num_frames + index
                    index = 0
                allowed_frame_indices = allowed_frame_indices[
                    index : index + temp_num_frames
                ]
                if (
                    len(allowed_frame_indices) < num_frames
                    and consecutive_frame_index < 0
                ):
                    f_counter = num_frames - len(allowed_frame_indices)
                    allowed_frame_indices = np.concatenate(
                        [
                            np.array([-1] * (num_frames - len(allowed_frame_indices))),
                            allowed_frame_indices,
                        ]
                    )
                if len(allowed_frame_indices) < num_frames:
                    allowed_frame_indices = np.concatenate(
                        [
                            allowed_frame_indices,
                            np.array([-1] * (num_frames - len(allowed_frame_indices))),
                        ]
                    )
            else:
                raise Exception(
                    "interval needs to be 'full' / 'consecutive' / 'selected' when num_frames is not None"
                )
        else:
            allowed_frame_indices = np.arange(0, fcount, ratio)

    frames = np.zeros((len(allowed_frame_indices), height, width, 3), dtype=np.uint8)
    f = 0

    while f < fcount:
        grabbed = cap.grab()
        if f in allowed_frame_indices:
            ret, frame = cap.retrieve()
            if ret != False:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames[f_counter] = frame
            f_counter += 1
            if num_frames is not None and f_counter >= num_frames:
                break
        f += 1
    cap.release()
    return frames, allowed_frame_indices