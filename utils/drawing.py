import cv2

def draw_bbox(img, bboxs, class_names, colors):
    h, w = img.shape[0:2]
    n = bboxs.size()[0]
    bboxs = bboxs.detach().numpy()
    print(bboxs)
    for i in range(n):
        p1 = (int((bboxs[i, 0] - bboxs[i, 2] / 2) * w), int((bboxs[i, 1] - bboxs[i, 3] / 2) * h))
        p2 = (int((bboxs[i, 0] + bboxs[i, 2] / 2) * w), int((bboxs[i, 1] + bboxs[i, 3] / 2) * h))
        class_name = class_names[int(bboxs[i, 5])]
        # confidence = bboxs[i, 4]
        cv2.rectangle(img, p1, p2, color=colors[int(bboxs[i, 5])], thickness=2)
        cv2.putText(img, class_name, p1, cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORS[int(bboxs[i, 5])])
    return img