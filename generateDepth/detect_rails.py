import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

def show_images(images, cmap=None):
    cols = 2
    rows = (len(images) + 1) // cols

    plt.figure(figsize=(10, 11))
    for i, image in enumerate(images):
        plt.subplot(rows, cols, i + 1)
        # use gray scale color map if there is only one channel
        cmap = 'gray' if len(image.shape) == 2 else cmap
        plt.imshow(image, cmap=cmap)
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout(pad=0, h_pad=0, w_pad=0)
    plt.show()

# convert color
def convert_hsv(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)


def convert_hls(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2HLS)


def convert_gray_scale(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def select_color():
    # if needed
    pass


def apply_smoothing(image, kernel_size=15):
    """
    kernel_size must be postivie and odd
    """
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def detect_edges(image, low_threshold=50, high_threshold=150):
    return cv2.Canny(image, low_threshold, high_threshold)


def hough_lines(image):
    """
    `image` should be the output of a Canny transform.

    Returns hough lines (not the image with lines)
    """
    return cv2.HoughLinesP(image, rho=1, theta=np.pi / 180, threshold=20, minLineLength=5, maxLineGap=10)


def average_slope_intercept(lines):
    left_lines = []  # (slope, intercept)
    left_weights = []  # (length,)
    right_lines = []  # (slope, intercept)
    right_weights = []  # (length,)

    for line in lines:
        # print(line)
        for x1, y1, x2, y2 in line:
            if x2 == x1:
                continue  # ignore a vertical line
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            length = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
            if slope < 0:  # y is reversed in image
                left_lines.append((slope, intercept))
                left_weights.append((length))
            else:
                right_lines.append((slope, intercept))
                right_weights.append((length))

    # add more weight to longer lines
    left_lane = np.dot(left_weights, left_lines) / np.sum(left_weights) if len(left_weights) > 0 else None
    right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights) if len(right_weights) > 0 else None

    return left_lane, right_lane  # (slope, intercept), (slope, intercept)


def make_line_points(y1, y2, line):
    """
    Convert a line represented in slope and intercept into pixel points
    """
    if line is None:
        return None

    slope, intercept = line

    # make sure everything is integer as cv2.line requires it
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    y1 = int(y1)
    y2 = int(y2)

    return (x1, y1, x2, y2)


def lane_lines(image, lines):
    left_lane, right_lane = average_slope_intercept(lines)

    y1 = image.shape[0]  # bottom of the image
    y2 = y1 * 0.6  # slightly lower than the middle

    left_line = make_line_points(y1, y2, left_lane)
    right_line = make_line_points(y1, y2, right_lane)

    return left_line, right_line


def filter_region(image, vertices):
    """
    Create the mask using the vertices and apply it to the input image
    """
    mask = np.zeros_like(image)
    if len(mask.shape) == 2:
        cv2.fillPoly(mask, vertices, 255)
    else:
        cv2.fillPoly(mask, vertices, (255,) * mask.shape[2])  # in case, the input image has a channel dimension
    return cv2.bitwise_and(image, mask)


def select_region(image):
    """
    It keeps the region surrounded by the `vertices` (i.e. polygon).  Other area is set to 0 (black).
    """
    # first, define the polygon by vertices
    # rows, cols = image.shape[:2]
    rows = image.shape[0]
    cols = 1280
    # bottom_left = [cols * 0.5, rows * 1]
    # top_left = [cols * 0.7, rows * 0.5]
    # bottom_right = [cols * 1, rows * 1]
    # top_right = [cols * 0.8, rows * 0.5]

    bottom_left = [cols * 0.5, rows * 0.92]
    top_left = [cols * 0.7, rows * 0.6]
    bottom_right = [cols * 1, rows * 0.92]
    top_right = [cols * 0.8, rows * 0.6]

    # the vertices are an array of polygons (i.e array of arrays) and the data type must be integer
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    return filter_region(image, vertices)


def delete_horizontal_lines():
    pass


if __name__ == '__main__':

    # vidcap = cv2.VideoCapture('/home-local/datasetvideo/9.mp4')
    count = 0
    s = 0
    success = True
    # fps = int(vidcap.get(cv2.CAP_PROP_FPS))

    # while success:
        # success, image = vidcap.read()
        # print('read a new frame:', success)
        # if count % (90) == 0:
    for files in sorted(glob.glob("/home-local/video_data/NoInsidesorStopvideos/0/image_03/data/*.jpg"), key=lambda x: int(os.path.splitext(os.path.basename(x))[0])): 
        print(files)
        image = cv2.imread(files)
        # cv2.imwrite('/home-local/datasetvideo/9'+'/%d.jpg' % (count // 90), image)
        img = cv2.GaussianBlur(image, (9, 9), 0)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 75, 150)
        edges = cv2.GaussianBlur(edges, (5, 5), 0)
        roi_images = select_region(edges)
        lines = cv2.HoughLinesP(roi_images, rho=1, theta=np.pi / 180, threshold=50, minLineLength=270, maxLineGap=7)
        # lines = cv2.HoughLinesP(roi_images, rho=1, theta=np.pi / 180, threshold=50, minLineLength=100, maxLineGap=1)

        # lines = cv2.HoughLinesP(roi_images, rho=1, theta=np.pi / 180, threshold=50, minLineLength=320, maxLineGap=3)
        # lines = cv2.HoughLinesP(roi_images, rho=1, theta=np.pi / 180, threshold=20, minLineLength=20, maxLineGap=2)
        if lines is not None:
            print('hello')
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
                xx = cv2.cvtColor(roi_images, cv2.COLOR_GRAY2RGB)
                cv2.line(xx, (x1, y1), (x2, y2), (0, 255, 0), 1)
                # cv2.imshow("frame", image)
                # cv2.imwrite('./%d.jpg' % s, image)
                cv2.imwrite('./%d.jpg' % s, xx)
                # cv2.imshow("edges", xx)
                # key = cv2.waitKey(1)
                # if key == 27:
                #     break
                # sort the frames

        cv2.imwrite('/home-local/video_data/NoInsidesorStopvideos/0/image_03/groundtruth/detect_rails/%d.jpg' % s, image)
        # cv2.imwrite('/home-local/datasetvideo/2/detect_rails/mask%d.jpg' % s, xx)
        s += 1
    # do this on frames
    # img1 = cv2.imread('../test_images/frame2004.jpg')
    # cv2.imwrite('canny1frame2004.jpg', edges)
    # cv2.imwrite('cannyframe2004.jpg', img1)



#  change it to read files instead of videos
# and detect the rails (tnajam tzid fel pixelsmen tawa)