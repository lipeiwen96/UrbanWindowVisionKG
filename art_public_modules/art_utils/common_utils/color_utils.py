"""
颜色工具
@建筑研究 李沛文
"""


class ColorMethods:
    @staticmethod
    def hex_to_rgb(color_hex):
        """
        将输入的16进制的颜色转化为rgb的表示方法
        "#ff01e4" --> (255,1,228)
        """
        r = int(color_hex[1:3], 16)
        g = int(color_hex[3:5], 16)
        b = int(color_hex[5:7], 16)
        return r, g, b

    @staticmethod
    def rgb_to_hex(r: int, g: int, b: int, upper: bool = False):
        color = "#"
        # 将R、G、B分别转化为16进制拼接转换并大写  hex() 函数用于将10进制整数转换成16进制，以字符串形式表示
        if upper:
            color += str(hex(r))[-2:].replace('x', '0').upper()
            color += str(hex(g))[-2:].replace('x', '0').upper()
            color += str(hex(b))[-2:].replace('x', '0').upper()
        else:
            color += str(hex(r))[-2:].replace('x', '0').lower()
            color += str(hex(g))[-2:].replace('x', '0').lower()
            color += str(hex(b))[-2:].replace('x', '0').lower()
        return color

    @staticmethod
    def hex_to_rgb_alpha(color_hex, alpha):
        r = int(color_hex[1:3], 16)
        g = int(color_hex[3:5], 16)
        b = int(color_hex[5:7], 16)
        return r, g, b, alpha


if __name__ == "__main__":
    print(ColorMethods.hex_to_rgb("#ff4242"))
    print(ColorMethods.rgb_to_hex(255, 66, 66, upper=True))
