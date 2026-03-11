def calculate_brightness(img):
    if not img or not img[0]:
        return -1

    height, width = len(img), len(img[0])
    total = 0

    for row in img:
        if len(row) != width:
            return -1

        for pixel in row:
            if not 0 <= pixel <= 255:
                return -1

            total += pixel

    return round(total / (height * width), 2)


img = [[100, 200], [50, 150]]
print(calculate_brightness(img))
