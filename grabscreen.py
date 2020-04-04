import cv2
import numpy as np
import win32gui, win32ui, win32con, win32api


def roi(x1,y1,x2,y2, n=150):
    # returns n*2 by n*2 region of interest in the center of the screen
    xdiff = x2 - x1
    ydiff = y2 - y1
    xcenter = xdiff / 2
    ycenter = ydiff / 2
    region =  [int(xcenter - n), int(ycenter - n), int(xcenter + n), int(ycenter + n)]
    img = grab_screen(region=region)
    return img


def get_coords(x1, y1, x2, y2, objectRegion, n=150):
    # returns the real location of the boxes relative to the gameplay screen
    roixorgin = int(((x2 - x1) / 2) - n)
    roiyorgin = int(((y2 - y1) / 2) - n)

    objectRegion = [objectRegion[0] + roixorgin,
                    objectRegion[1] + roiyorgin,
                    objectRegion[2] + roixorgin,
                    objectRegion[3] + roiyorgin]

    return objectRegion

def grab_screen(region=None):

    hwin = win32gui.GetDesktopWindow()

    if region:
            left,top,x2,y2 = region
            width = x2 - left + 1
            height = y2 - top + 1
    else:
        width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
        height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)
        left = win32api.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN)
        top = win32api.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN)

    hwindc = win32gui.GetWindowDC(hwin)
    srcdc = win32ui.CreateDCFromHandle(hwindc)
    memdc = srcdc.CreateCompatibleDC()
    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(srcdc, width, height)
    memdc.SelectObject(bmp)
    memdc.BitBlt((0, 0), (width, height), srcdc, (left, top), win32con.SRCCOPY)
    
    signedIntsArray = bmp.GetBitmapBits(True)
    img = np.fromstring(signedIntsArray, dtype='uint8')
    img.shape = (height,width,4)

    srcdc.DeleteDC()
    memdc.DeleteDC()
    win32gui.ReleaseDC(hwin, hwindc)
    win32gui.DeleteObject(bmp.GetHandle())

    return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)



