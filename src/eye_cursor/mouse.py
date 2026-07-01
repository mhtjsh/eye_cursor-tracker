from __future__ import annotations


class MouseController:
    def __init__(self, failsafe_margin: int = 8) -> None:
        try:
            import pyautogui
        except ImportError as exc:
            raise RuntimeError("PyAutoGUI is not installed. Run: python -m pip install -e .") from exc

        self._pyautogui = pyautogui
        self._pyautogui.PAUSE = 0
        self._pyautogui.FAILSAFE = True
        self.width, self.height = pyautogui.size()
        self.failsafe_margin = max(0, int(failsafe_margin))

    def clamp(self, x: float, y: float) -> tuple[int, int]:
        margin = self.failsafe_margin
        max_x = max(int(self.width) - margin - 1, margin)
        max_y = max(int(self.height) - margin - 1, margin)
        px = min(max(int(round(x)), margin), max_x)
        py = min(max(int(round(y)), margin), max_y)
        return px, py

    def move_to(self, x: float, y: float) -> tuple[int, int]:
        px, py = self.clamp(x, y)
        self._pyautogui.moveTo(px, py, duration=0)
        return px, py

    def left_click(self) -> None:
        self._pyautogui.click(button="left")

    def right_click(self) -> None:
        self._pyautogui.click(button="right")
