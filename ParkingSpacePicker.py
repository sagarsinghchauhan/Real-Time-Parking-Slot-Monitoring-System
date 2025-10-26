from opcode import opname

import cv2
import pickle

try:
    with open('CarParkPos', 'rb') as f:
        posList = pickle.load(f)
        # pickle.dump(posList, f)
except:
    posList = []

# Load the image
# img = cv2.imread('carParkImg.png')

# ✅ Create the window first
cv2.namedWindow("Image")

width, height = 107, 48

# Mouse click event
def mouseClick(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        posList.append((x, y))
        print("Added position:", (x, y))
    if event == cv2.EVENT_RBUTTONDOWN:
        for i,pos in enumerate(posList):
            x1, y1 = pos
            if x1<x<x1+width and y1<y<y1+height:
                posList.pop(i)

    with open('CarParkPos', 'wb'    ) as f:
        pickle.dump(posList, f)



while True:
    # Make a copy of the image to redraw rectangles each time
    img = cv2.imread('carParkImg.png')

    # Draw rectangles for all saved positions
    for pos in posList:
        cv2.rectangle(img, pos, (pos[0] + width, pos[1] + height), (255, 0, 255), 2)

    # ✅ Use the same name here as above
    cv2.imshow("Image", img)

    # ✅ Set the callback (best done once, but this also works)
    cv2.setMouseCallback("Image", mouseClick)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cv2.destroyAllWindows()
