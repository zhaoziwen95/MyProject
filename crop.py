def crop(img):
    """crop a picture
        return this picture"""

    L = 800
    H = 800
    pos_y = 357
    pos_x = 673
    img = img[pos_y:pos_y + L, pos_x:pos_x + H]
    return img
