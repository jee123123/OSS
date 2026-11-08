import unittest

from PIL import Image, ImageDraw

from score2midi.preprocess import extend_short_final_system


class ShortFinalSystemTest(unittest.TestCase):
    def _page(self, short_last: bool) -> Image.Image:
        page = Image.new("L", (800, 1000), 255)
        draw = ImageDraw.Draw(page)

        for system_index, top in enumerate((100, 400, 700)):
            right = 200 if short_last and system_index == 2 else 750
            for staff_top in (top, top + 70):
                for line_index in range(5):
                    y = staff_top + line_index * 7
                    draw.line((70, y, right, y), fill=0)
            draw.line((right, top, right, top + 98), fill=0, width=3)

        return page

    def test_extends_short_final_system(self) -> None:
        result = extend_short_final_system(self._page(True))

        self.assertLess(result.getpixel((740, 700)), 128)
        self.assertLess(result.getpixel((740, 798)), 128)

    def test_keeps_regular_final_system(self) -> None:
        page = self._page(False)
        result = extend_short_final_system(page)

        self.assertIs(result, page)


if __name__ == "__main__":
    unittest.main()
