from __future__ import annotations


def get_screen_size() -> tuple[int, int]:
    try:
        import pyautogui

        size = pyautogui.size()
        return int(size.width), int(size.height)
    except Exception:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        try:
            return int(root.winfo_screenwidth()), int(root.winfo_screenheight())
        finally:
            root.destroy()
