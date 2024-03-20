import random

from PIL import Image, ImageDraw, ImageFont

width, height = 1100, 2000


def generate(color, digit1, digit2):
    # 创建一个白色背景的图像
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    # 画一个边长为1m的正方形
    square_side = 1000
    square_top_left = (0, 1000)
    square_bottom_right = (square_top_left[0] + square_side, square_top_left[1] + square_side)
    draw.rectangle([square_top_left, square_bottom_right], fill=color)

    # 画一个边长为1m的等边三角形
    triangle_height = (3 ** 0.5 / 2) * square_side  # 通过正方形边长计算等边三角形的高度
    triangle_top = (square_top_left[0] + square_side / 2, square_top_left[1] - triangle_height)
    triangle_bottom_left = (square_top_left[0], square_top_left[1])
    triangle_bottom_right = (square_bottom_right[0], square_top_left[1])
    draw.polygon([triangle_top, triangle_bottom_left, triangle_bottom_right], fill=color)

    # 画靶标
    font_path = "SimHei.ttf"  # 替换为你的加粗黑体字体文件路径
    font_size = 500  # 调整字体大小
    font = ImageFont.truetype(font_path, font_size)

    center_x = square_side / 2
    center_y = square_side * 1.4

    text_width, text_height = draw.textsize(digit1, font)
    draw.rectangle([(center_x - 310, center_y - 300), (center_x - 10, center_y + 300)], fill="white")
    draw.text((center_x - 310 + (300 - text_width) / 2, center_y - 300 + (600 - text_height) / 2), digit1, font=font,
              fill='black')

    text_width, text_height = draw.textsize(digit2, font)
    draw.rectangle([(center_x + 10, center_y - 300), (center_x + 310, center_y + 300)], fill="white")
    draw.text((center_x + 10 + (300 - text_width) / 2, center_y - 300 + (600 - text_height) / 2), digit2, font=font,
              fill='black')

    return image


if __name__ == '__main__':
    num = 4
    digits = []
    for i in range(2 * num):
        digits.extend(range(10))

    data = []
    while len(digits) > 0:
        print(len(digits))

        i, j = random.choice(range(len(digits))), random.choice(range(len(digits)))
        while i == j:
            i, j = random.choice(range(len(digits))), random.choice(range(len(digits)))

        i = digits[i]
        j = digits[j]

        k = f'{i}{j}'

        if k not in data:
            data.append(k)
            digits.remove(i)
            digits.remove(j)

        else:
            continue

    print('----------------------')
    print(data)
    print(len(data))

    images = []
    for i in data:
        images.append(generate(random.choice(['blue', 'red']), i[0], i[1]))

    col = 10
    row = 4

    image = Image.new("RGB", (width * col, height * row), "white")

    for i in range(row):
        for j in range(col):
            image.paste(images[i * col + j], (width * j, height * i))

    image.save('1.png')
