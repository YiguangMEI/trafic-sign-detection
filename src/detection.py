def pyramid(image, scale=1.5, min_size=(64, 64)):
    yield image
    while True:
        w = int(image.shape[1] / scale)
        image = cv2.resize(image, (w, int(w * image.shape[0] / image.shape[1])))
        if image.shape[0] < min_size[1] or image.shape[1] < min_size[0]:
            break
        yield image


def sliding_window(image, step_size, window_size):
    for y in range(0, image.shape[0] - window_size[1], step_size):
        for x in range(0, image.shape[1] - window_size[0], step_size):
            yield (x, y, image[y : y + window_size[1], x : x + window_size[0]])


def detect_traffic_signs(
    image, model, win_size=(64, 64), step_size=16, pyramid_scale=1.5
):
    detections = []
    for resized in pyramid(image, scale=pyramid_scale, min_size=win_size):
        for x, y, window in sliding_window(resized, step_size, win_size):
            if window.shape[0] != win_size[1] or window.shape[1] != win_size[0]:
                continue
            ##this place need to
            gray_window = cv2.cvtColor(window, cv2.COLOR_BGR2GRAY)
            hog_features = hog(
                gray_window,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
                feature_vector=True,
            )
            hog_features = hog_features.reshape(1, -1)

            prediction = model.predict(hog_features)

            if prediction[0] == 1:  # Assuming '1' indicates a traffic sign
                x_orig = int(x * (image.shape[1] / resized.shape[1]))
                y_orig = int(y * (image.shape[0] / resized.shape[0]))
                w_orig = int(win_size[0] * (image.shape[1] / resized.shape[1]))
                h_orig = int(win_size[1] * (image.shape[0] / resized.shape[0]))
                detections.append((x_orig, y_orig, w_orig, h_orig, prediction[0]))

    return detections


def non_max_suppression(boxes, overlap_thresh):
    if len(boxes) == 0:
        return []

    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    pick = []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(
            idxs, np.concatenate(([last], np.where(overlap > overlap_thresh)[0]))
        )

    return boxes[pick].astype("int")
