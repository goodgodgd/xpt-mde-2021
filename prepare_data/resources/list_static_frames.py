import os.path as op
from config import opts
from glob import glob
import pykitti
import cv2
import numpy as np

'''
References
https://github.com/opencv/opencv/blob/master/samples/python/opt_flow.py
def draw_flow(img, flow, step=16):
    ...

https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html
https://eehoeskrap.tistory.com/124
'''


def list_kitti_odom_static_frames():
    with open("kitti_odom_staic_frames.txt", "w") as fw:
        for drive_id in range(22):
            drive_path = op.join(opts.KITTI_ODOM_PATH, "sequences", f"{drive_id:02d}")
            pattern = op.join(drive_path, "image_2", "*.png")
            frame_files = glob(pattern)
            frame_files.sort()
            for fi in range(1, len(frame_files)-1):
                frame_bef = cv2.imread(frame_files[fi-1])
                frame_bef = cv2.cvtColor(frame_bef, cv2.COLOR_RGB2GRAY)
                frame_cur = cv2.imread(frame_files[fi])
                frame_cur = cv2.cvtColor(frame_cur, cv2.COLOR_RGB2GRAY)
                frame_aft = cv2.imread(frame_files[fi+1])
                frame_aft = cv2.cvtColor(frame_aft, cv2.COLOR_RGB2GRAY)

                flow1 = cv2.calcOpticalFlowFarneback(frame_bef, frame_cur,
                                            flow=None, pyr_scale=0.5, levels=3, winsize=10,
                                            iterations=3, poly_n=5, poly_sigma=1.1, flags=0)
                flow2 = cv2.calcOpticalFlowFarneback(frame_cur, frame_aft,
                                            flow=None, pyr_scale=0.5, levels=3, winsize=15,
                                            iterations=3, poly_n=5, poly_sigma=1.1, flags=0)
                # vis = draw_flow(frame_cur, flow1)
                # cv2.imshow("flow", vis)
                # if cv2.waitKey(500) == ord('q'):
                #     return
                # if cv2.waitKey(500) == ord('s'):
                #     break

                flow1_dist = np.sqrt(flow1[:, :, 0] * flow1[:, :, 0] + flow1[:, :, 1] * flow1[:, :, 1])
                flow2_dist = np.sqrt(flow2[:, :, 0] * flow2[:, :, 0] + flow2[:, :, 1] * flow2[:, :, 1])
                img_size = flow1.shape[0] * flow1.shape[1]
                valid1 = np.count_nonzero((2 < flow1_dist) & (flow1_dist < 50)) / img_size
                valid2 = np.count_nonzero((2 < flow2_dist) & (flow2_dist < 50)) / img_size
                print(f"drive: {drive_id:02d} {fi:06d}, ratio valid flows: {valid1:0.3f}, {valid2:0.3f}")
                if valid1 < 0.4 or valid2 < 0.4:
                    print(f"===== write: {drive_id:02d} {frame_files[fi][-10:-4]} {frame_files[fi]}")
                    fw.write(f"{drive_id:02d} {frame_files[fi][-10:-4]}\n")


def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (_x2, _y2) in lines:
        dist = np.sqrt(np.square(x1 - _x2) + np.square(y1 - _y2))
        if dist > 50:
            cv2.circle(vis, (x1, y1), 1, (0, 0, 0), -1)
        elif dist < 2:
            cv2.circle(vis, (x1, y1), 1, (255, 0, 0), -1)
        else:
            cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis


if __name__ == "__main__":
    list_kitti_odom_static_frames()