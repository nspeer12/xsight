import pyautogui
import time
def click(x, y):
    pyautogui.click(x, y)

def get_position():
    return pyautogui.position()

def quickscope():
    # hold right for 1 second
    pyautogui.dragTo(None, None, 1, button='right')
    pyautogui.click(button='left')

def gratata(n):
    pyautogui.click(button='left', clicks=n, interval=0.2)