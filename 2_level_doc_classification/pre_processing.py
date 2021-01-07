"""IMAGE PREPROCESSING FUNCTIONS
"""
import cv2
import numpy as np
from scipy.ndimage.filters import rank_filter

# from sbox.utils.sbox_logger import logger
import pytesseract
import re
import imutils
from PIL import Image

# print("error") # = logger(__name__)


class PagePreprocess(object):
    def __init__(self, im):
        self.err = False
        self.orig_im = im
        self.orig_shape = self.orig_im.shape
        self.image = im

    def crop(self):
        try:
            self.image, self.num_tries = process_image(self.orig_im)
            self.crop_shape = self.image.shape
            return self.image
        except Exception as e:
            print("crop_obj_Error")  # (f"Error: {e}", exc_info=True)

    def deskew(self):
        try:
            self.image, self.theta_est = process_skewed_crop(self.image)
            return self.image
        except Exception as e:
            print("deskew_obj_Error")  # (f"Error: {e}", exc_info=True)


def auto_canny(image, sigma=0.33):
    try:
        v = np.median(image)
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edged = cv2.Canny(image, lower, upper, True)
        return edged
    except Exception as e:
        print("auto_canny_Error")  # (f"Error: {e}", exc_info=True)


def dilate(image, kernel, iterations):
    dilated_image = cv2.dilate(image, kernel, iterations=iterations)
    return dilated_image


def downscale_image(im, max_dim=2048):
    try:
        a, b = im.shape[:2]
        if max(a, b) <= max_dim:
            return 1.0, im

        scale = 1.0 * max_dim / max(a, b)
        new_im = cv2.resize(im, (int(b * scale), int(a * scale)), cv2.INTER_AREA)
        return scale, new_im
    except Exception as e:
        print("error")  # (f"Error: {e}", exc_info=True)


def find_components(im, max_components=16):
    try:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        dilation = dilate(im, kernel, 6)

        count = 21
        n = 0
        sigma = 0.000

        while count > max_components:
            n += 1
            sigma += 0.005
            result = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if len(result) == 3:
                _, contours, hierarchy = result
            elif len(result) == 2:
                contours, hierarchy = result
            possible = find_likely_rectangles(contours, sigma)
            count = len(possible)

        return (dilation, possible, n)
    except Exception as e:
        print("comp_error")  # (f"Error: {e}", exc_info=True)


def find_likely_rectangles(contours, sigma):
    try:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        possible = []
        for c in contours:

            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, sigma * peri, True)
            box = make_box(approx)
            possible.append(box)

        return possible
    except Exception as e:
        print("likely_rec_error")  # (f"Error: {e}", exc_info=True)


def make_box(poly):
    try:
        x = []
        y = []
        for p in poly:
            for point in p:
                x.append(point[0])
                y.append(point[1])
        xmax = max(x)
        ymax = max(y)
        xmin = min(x)
        ymin = min(y)
        return (xmin, ymin, xmax, ymax)
    except Exception as e:
        print("bbox_error")  # (f"Error: {e}", exc_info=True)


def rect_union(crop1, crop2):
    x11, y11, x21, y21 = crop1
    x12, y12, x22, y22 = crop2
    return min(x11, x12), min(y11, y12), max(x21, x22), max(y21, y22)


def rect_area(crop):
    x1, y1, x2, y2 = crop
    return max(0, x2 - x1) * max(0, y2 - y1)


def crop_image(im, rect, scale):
    try:
        xmin, ymin, xmax, ymax = rect
        crop = [xmin, ymin, xmax, ymax]
        xmin, ymin, xmax, ymax = [int(x / scale) for x in crop]
        if ((ymax - ymin) * (xmax - xmin)) > 0.25 * im.size:
            cropped = im[ymin:ymax, xmin:xmax]
        else:
            cropped = im
        return cropped
    except Exception as e:
        print("crop_error_1")  # (f"Error: {e}", exc_info=True)


def reduce_noise_raw(im):
    bilat = cv2.bilateralFilter(im, 4, 75, 75)
    blur = cv2.medianBlur(bilat, 1)
    return blur


def reduce_noise_edges(im):
    try:
        structuring_element = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        opening = cv2.morphologyEx(im, cv2.MORPH_OPEN, structuring_element)
        maxed_rows = rank_filter(opening, -4, size=(1, 20))
        maxed_cols = rank_filter(opening, -4, size=(20, 1))
        debordered = np.minimum(np.minimum(opening, maxed_rows), maxed_cols)
        return debordered
    except Exception as e:
        print("noise_red_Error")  # (f"Error: {e}", exc_info=True)


def rects_are_vertical(rect1, rect2, rect_align=2):
    try:
        xmin1, ymin1, xmax1, ymax1 = rect1
        xmin2, ymin2, xmax2, ymax2 = rect2

        midpoint1 = (xmin1 + xmax1) / 2
        midpoint2 = (xmin2 + xmax2) / 2
        dist = abs(midpoint1 - midpoint2)

        rectarea1 = rect_area(rect1)
        rectarea2 = rect_area(rect2)
        if rectarea1 > rectarea2:
            thres = (xmax1 - xmin1) * rect_align
        else:
            thres = (xmax2 - xmin2) * rect_align

        if thres > dist:
            align = True
        else:
            align = False
        return align
    except Exception as e:
        print("vert_rec_Error")  # (f"Error: {e}", exc_info=True)


def find_final_crop(im, rects, orig_im):
    try:
        current = None
        for rect in rects:
            if current is None:
                current = rect
                continue

            aligned = rects_are_vertical(current, rect)

            if not aligned:
                continue

            current = rect_union(current, rect)
        if current is not None:
            return current
        else:
            return (0, 0, orig_im.shape[0], orig_im.shape[1])
    except Exception as e:
        print("crop_Error")  # (f"Error: {e}", exc_info=True)


def process_image(orig_im):
    try:
        scale, im = downscale_image(orig_im)

        blur = reduce_noise_raw(im.copy())

        edges = auto_canny(blur.copy())

        debordered = reduce_noise_edges(edges.copy())

        dilation, rects, num_tries = find_components(debordered, 16)

        final_rect = find_final_crop(dilation, rects, orig_im)

        cropped = crop_image(orig_im, final_rect, scale)
        # kernel = np.ones((3, 3), np.float32) / 25
        # smooth2d = cv2.filter2D(cropped, -1, kernel=kernel)
        return (cropped, num_tries)
    except Exception as e:
        print("process")  # (f"Error: {e}", exc_info=True)


def rad_to_deg(theta):
    return theta * 180 / np.pi


def rotate(image, theta):
    try:
        (h, w) = image.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, theta, 1)
        rotated = cv2.warpAffine(
            image,
            M,
            (int(w), int(h)),
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )
        return rotated
    except Exception as e:
        print("rotation_error")  # (f"Error: {e}", exc_info=True)


def angle_calculation(gray):
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    coords = np.column_stack(np.where(thresh > 0))
    # print(coords, coords.shape)

    min_y = coords[0][0]
    max_y = coords[-1][0]
    min_x = coords[0][0]
    max_x = coords[-1][1]

    left_most = coords[0]
    right_most = coords[0]
    top_most = coords[0]
    bottom_most = coords[0]
    # print(coords[0], coords[-1])
    for i in range(1, coords.shape[0]):
        y, x = coords[i][0], coords[i][1]
        if y <= min_y:
            min_y = y
            top_most = coords[i]
        elif y >= max_y:
            max_y = y
            bottom_most = coords[i]
        if x <= min_x:
            min_x = x
            left_most = coords[i]
        elif x >= max_x:
            max_x = x
            right_most = coords[i]
    # print(top_most, left_most, bottom_most, right_most)

    slopes = []
    edge_coor = [top_most, left_most, bottom_most, right_most]
    for i in range(0, len(edge_coor)):
        if i == len(edge_coor) - 1:
            if abs((edge_coor[0][1] - edge_coor[i][1])) >= 10:
                angle = (
                    (
                        (edge_coor[0][0] - edge_coor[i][0])
                        / (edge_coor[0][1] - edge_coor[i][1])
                    )
                    * 180
                ) / 3.14
                slopes.append(angle)
            else:
                slopes.append(0.0)
        else:
            if abs((edge_coor[i + 1][1] - edge_coor[i][1])) >= 10:
                angle = (
                    (
                        (edge_coor[i + 1][0] - edge_coor[i][0])
                        / (edge_coor[i + 1][1] - edge_coor[i][1])
                    )
                    * 180
                ) / 3.14
                slopes.append(angle)
            else:
                slopes.append(0.0)
        # img = cv2.circle(thresh, (edge_coor[i][1], edge_coor[i][0]), 5, (255, 0, 0), 2)

    slopes = np.asarray(slopes)
    if len(np.where(slopes == 0.0)[0]) >= 2:
        # print("error") #(f"Error: {e}", exc_info=True)don't rotate")
        return None
    else:
        # print("error") #(f"Error: {e}", exc_info=True)rotate")
        neg_slope = (slopes[0] + slopes[2]) / 2
        pos_slope = (slopes[1] + slopes[3]) / 2
        # print(pos_slope, neg_slope)
        new_pos_slope = pos_slope
        new_neg_slope = neg_slope
        if pos_slope > 90:
            if pos_slope < 180:
                new_pos_slope = 180 - pos_slope
            else:
                new_pos_slope = pos_slope - ((pos_slope // 180) * 180)
                # print(new_pos_slope)
        if neg_slope < -90:
            new_neg_slope = 180 + neg_slope
        # print(new_pos_slope, new_neg_slope)
        if new_pos_slope <= new_neg_slope:
            fin_angle = pos_slope
        else:
            fin_angle = neg_slope

        if fin_angle < -90:
            rot_angle = 180 + fin_angle
        elif fin_angle > 90:
            rot_angle = -(180 - fin_angle)
        elif -90 < fin_angle < 0:
            rot_angle = fin_angle
        elif 0 < fin_angle < 90:
            rot_angle = fin_angle
        return rot_angle


def estimate_skew(image):
    try:
        osd = pytesseract.image_to_osd(image)
        angle = float(re.search("(?<=Rotate: )\d+", osd).group(0))
        if angle == 0:
            # fin_image = rotate(image_gray, angle)
            edges = auto_canny(image)
            # print(edges.shape)
            # print("error") #(f"Error: {e}", exc_info=True)edges found: ", edges)
            lines = cv2.HoughLines(edges, 1, np.pi / 270, 400)
            # print("error") #(f"Error: {e}", exc_info=True)lines found: ", lines)
            if lines is not None:
                new = edges.copy()
                thetas = []
                for line in lines:
                    for rho, theta in line:
                        a = np.cos(theta)
                        b = np.sin(theta)
                        x0 = a * rho
                        y0 = b * rho
                        x1 = int(x0 + 1000 * (-b))
                        y1 = int(y0 + 1000 * (a))
                        x2 = int(x0 - 1000 * (-b))
                        y2 = int(y0 - 1000 * (a))
                        if theta > np.pi / 3 and theta < np.pi * 2 / 3:
                            thetas.append(theta)
                            new = cv2.line(new, (x1, y1), (x2, y2), (255, 255, 255), 1)

                theta_mean = np.mean(thetas)
                theta = -(90 - (rad_to_deg(theta_mean) if len(thetas) > 0 else 0))
            else:
                # theta = angle_calculation(image)
                theta = 0.0
        else:
            theta = angle
        return theta
    except Exception as e:
        print("theta_error")  # (f"Error: {e}", exc_info=True)


def process_skewed_crop(image):
    try:
        theta = estimate_skew(image)
        # print(theta)
        # ret, thresh = cv2.threshold(image, 0, 127, cv2.THRESH_OTSU)
        # print(thresh)
        if theta is not None and (theta % 90) != 0:
            rotated = rotate(image, theta)
        elif (theta % 90) == 0:
            rotated = imutils.rotate_bound(image, theta)
        else:
            rotated = image
        # print(rotated)
        return rotated, theta
    except Exception as e:
        print("skew_Error")  # (f"Error: {e}", exc_info=True)


def preprocess_image(file_path: str):
    try:
        gray_page = cv2.imread(file_path, 0)
        process_page = PagePreprocess(gray_page)
        _ = process_page.crop()
        deskewed_page = process_page.deskew()
        # cv2.imwrite(file_path, deskewed_page)
        return deskewed_page
    except Exception as e:
        print("process_image_error")  # (f"Error: {e}", exc_info=True)


def preprocess_image_file(img):
    try:
        # converted_image = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        gray_page = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
        # gray_page = cv2.cvtColor(gray_page, cv2.COLOR_BGR2RGB)
        process_page = PagePreprocess(gray_page)
        _ = process_page.crop()
        deskewed_page = process_page.deskew()
        return deskewed_page
    except Exception as e:
        print("error")  # (f"Error: {e}", exc_info=True)
