import cv2
import numpy as np
import HandTrackingModule as htm
import time
import pyautogui

# ------------------- CONFIGURATIONS -------------------

# Camera and Screen Configurations
wCam, hCam = 640, 480
wScr, hScr = pyautogui.size()

# Frame Configurations
frameR = 200  # Frame Reduction

# Mouse Movement Configurations
smoothening = 4
move_distance_threshold = 48
durationCloseThreshold = 0.25

# Click Configurations
click_distance_threshold = 40
left_click_landmarks = (8, 4)  # Landmarks for left click
right_click_landmarks = (12, 4)  # Landmarks for right click

# Scroll Configurations
scroll_speed = 6
scroll_threshold = 50

# ------------------- INITIALIZATIONS -------------------
detector = htm.handDetector(maxHands=1)
pyautogui.FAILSAFE = False
plocX, plocY = 0, 0
clocX, clocY = 0, 0
heldStartTime = None
isHeldClick = False
isLeftClick = False
isRightClick = False
pTime = 0

# Function to move the mouse cursor
def move_mouse(x1, y1):
    x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
    y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))
    clocX = plocX + (x3 - plocX) / smoothening
    clocY = plocY + (y3 - plocY) / smoothening
    pyautogui.moveTo(wScr - clocX, clocY)
    return clocX, clocY

# Function to handle mouse clicks
def handle_mouse_clicks(length, lineInfo, click_type):
    global isHeldClick, isLeftClick, isRightClick, heldStartTime

    # Check if landmarks are close enough for a click action
    if length < click_distance_threshold:
        if heldStartTime is None:
            heldStartTime = time.time()
        durationClose = time.time() - heldStartTime
        if durationClose > durationCloseThreshold and not isHeldClick:
            pyautogui.mouseDown()
            isHeldClick = True
        elif durationClose < durationCloseThreshold and not isHeldClick:
            if click_type == 'left' and not isLeftClick:
                pyautogui.click(button='left')
                isLeftClick = True
            elif click_type == 'right' and not isRightClick:
                pyautogui.click(button='right')
                isRightClick = True
    else:
        heldStartTime = None
        if isHeldClick and length > (click_distance_threshold * 1.5):  # Adjust multiplier as needed
            pyautogui.mouseUp()
            isHeldClick = False
        isLeftClick = False
        isRightClick = False

# Function to handle scrolling
def handle_scrolling(y1, y2, fingers):
    scroll_distance = abs(y1 - y2)
    if scroll_distance > scroll_threshold:
        scroll_amount = scroll_speed * (scroll_distance - scroll_threshold)
        if all(fingers[i] == 1 for i in range(1, 5)):
            pyautogui.scroll(scroll_amount)
        elif all(fingers[i] == 0 for i in range(2, 5)):
            pyautogui.scroll(-scroll_amount)

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    if len(lmList) != 0:
        # Displaying coordinates for landmarks 5, 8, 12 and the rectangle's corners
        lm5 = lmList[5][1:]
        lm8 = lmList[8][1:]
        lm12 = lmList[12][1:]
        cv2.putText(img, f'5: {lm5}', (lm5[0] + 10, lm5[1]), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
        cv2.putText(img, f'8: {lm8}', (lm8[0] + 10, lm8[1]), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
        cv2.putText(img, f'12: {lm12}', (lm12[0] + 10, lm12[1]), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)

        rect_coords = [(frameR, frameR), (wCam - frameR, frameR), (wCam - frameR, hCam - frameR), (frameR, hCam - frameR)]
        for coord in rect_coords:
            cv2.putText(img, str(coord), (coord[0] + 10, coord[1]), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)

        x1, y1 = lmList[5][1:]
        x2, y2 = lmList[4][1:]
        fingers = detector.fingersUp()
        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 1)

        if fingers[1] == 1 and fingers[0] == 1:
            length, img, lineInfo = detector.findDistance(8, 4, img)
            if length > move_distance_threshold:
                plocX, plocY = move_mouse(x1, y1)

        length, img, lineInfo = detector.findDistance(*left_click_landmarks, img)
        handle_mouse_clicks(length, lineInfo, 'left')
        length, img, lineInfo = detector.findDistance(*right_click_landmarks, img)
        handle_mouse_clicks(length, lineInfo, 'right')

        handle_scrolling(y1, y2, fingers)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
