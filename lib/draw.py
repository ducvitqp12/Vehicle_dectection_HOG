import cv2

font = cv2.FONT_HERSHEY_SIMPLEX
fontscale = .5
fontwd = 1
linetype = cv2.LINE_AA


def rect(img, box, color=(0, 255, 0), txt='', thick=2):
    ''' Returns result of cv2.rectangle()
    box: ((x0,y0),(x1,y1)) - box[0] and [1] are passed to cv2.rectangle
    txt: text label to be drawn on top left of box
    '''
    if txt:
        _w, _h = cv2.getTextSize(txt, font, fontscale, fontwd)[0]
        wd, ht = int(_w), int(_h)
        x0 = box[0][0] - (thick - 1)
        y0 = box[0][1] - ht - 4
        if x0 < 0:
            x0 = 0
        if y0 < 0:
            y0 = 0
        img[y0:y0 + ht + 4, x0:x0 + wd + 8] = color
        cv2.putText(img, txt, (x0 + 4, y0 + ht + 1), font, fontscale, (0, 0, 0), fontwd, linetype)
    return cv2.rectangle(img, box[0], box[1], color, thick)
