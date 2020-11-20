'''Helper functions for plotting functionality.

Coordinate transforms, text positioning etc.
'''

def coord_height_to_pixels(ax, height):
    p1 = ax.transData.transform((0, height))
    p2 = ax.transData.transform((0, 0))

    pix_height = p1[1] - p2[1]
    return pix_height

def hbar_text_position(rect, x_pos = 0.5, y_pos=0.5):
    rx, ry = rect.get_xy()
    width = rect.get_width()
    height = rect.get_height()

    tx = rx + (width * x_pos)
    ty = ry + (height * y_pos)
    return (tx, ty)