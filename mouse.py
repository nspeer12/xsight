import pyautogui
import time
def click(x, y):
    pyautogui.click(x, y)

def get_position():
    return pyautogui.position()

def quickscope():
    # hold right for 1 second
    aim(.5)
    pyautogui.click(button='left')
    time.sleep(1)

def gratata(n):
    pyautogui.click(button='left', clicks=n, interval=0.1)


def full_auto(n):
    # aim for n seconds
    pyautogui.dragTo(1, 1, n, button='left')


def aim(n):
    # aim for n seconds
    pyautogui.dragTo(1, 1, n, button='right')

def lock_on_target(objectRegion):
    xcenter = int((objectRegion[2] - objectRegion[0]) / 2) + objectRegion[0]
    ycenter = int((objectRegion[3] - objectRegion[1]) / 2) + objectRegion[1]

    # bad coordinates
    if ycenter < 0 or xcenter < 0:
        print('invalid coordinates')
        return
    else:
        x, y = get_position()
        pyautogui.move(x - xcenter, y - ycenter)
        