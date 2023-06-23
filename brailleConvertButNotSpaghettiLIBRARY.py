from PIL import Image
import cv2
import numpy
charMap = {'      ': ' ', '.     ': '⠁', '  .   ': '⠂', '. .   ': '⠃', '    . ': '⠄', '.   . ': '⠅', '  . . ': '⠆', '. . . ': '⠇', ' .    ': '⠈', '..    ': '⠉', ' ..   ': '⠊', '...   ': '⠋', ' .  . ': '⠌', '..  . ': '⠍', ' .. . ': '⠎', '... . ': '⠏', '   .  ': '⠐', '.  .  ': '⠑', '  ..  ': '⠒', '. ..  ': '⠓', '   .. ': '⠔', '.  .. ': '⠕', '  ... ': '⠖', '. ... ': '⠗', ' . .  ': '⠘', '.. .  ': '⠙', ' ...  ': '⠚', '....  ': '⠛', ' . .. ': '⠜', '.. .. ': '⠝', ' .... ': '⠞', '..... ': '⠟', '     .': '⠠', '.    .': '⠡', '  .  .': '⠢', '. .  .': '⠣', '    ..': '⠤', '.   ..': '⠥', '  . ..': '⠦', '. . ..': '⠧', ' .   .': '⠨', '..   .': '⠩', ' ..  .': '⠪', '...  .': '⠫', ' .  ..': '⠬', '..  ..': '⠭', ' .. ..': '⠮', '... ..': '⠯', '   . .': '⠰', '.  . .': '⠱', '  .. .': '⠲', '. .. .': '⠳', '   ...': '⠴', '.  ...': '⠵', '  ....': '⠶', '. ....': '⠷', ' . . .': '⠸', '.. . .': '⠹', ' ... .': '⠺', '.... .': '⠻', ' . ...': '⠼', '.. ...': '⠽', ' .....': '⠾', '......': '⠿'}






def matchChar(pixels):
    string = ""
    for i in pixels:
        if i > 0:
            string += ' '
        else:
            string += '.'
    return charMap.get(string)



def prepImage(name, size):
    og = Image.open(name)
    gray = og.convert('L')
    gray = gray.resize((int(gray.size[0] * size), int(gray.size[1] * size)), Image.Resampling.LANCZOS)

    avgColor = 0
    for i in range(gray.height):
        for j in range(gray.width):
            avgColor += gray.getpixel((j, i))

    avgColor /= (gray.height * gray.width)
    BWfunc = lambda x: 255 if x > avgColor else 0
    BW = gray.convert('L').point(BWfunc, mode='1')
    yCount = BW.height // 3
    xCount = BW.width // 2
    return [BW, xCount, yCount]

def prepImageContours(name, size):
    og = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
    og = cv2.flip(og, 0)
    og = cv2.rotate(og, cv2.ROTATE_90_CLOCKWISE)
    gray = cv2.resize(og, (int(og.shape[0] * size), int(og.shape[1] * size)))
    edges = cv2.Canny(gray, 255/3, 255)
    coords = numpy.where(edges != [0])
    coordsSet = set(zip(coords[0], coords[1]))

    yCount = gray.shape[1] // 3
    xCount = gray.shape[0] // 2
    coordsDict = {}
    for i in range(gray.shape[1]):
        for j in range(gray.shape[0]):
            if (j, i) not in coordsSet:
                coordsDict[(j, i)] = 1
            else:
                coordsDict[(j, i)] = 0

    return [coordsDict, xCount, yCount]



def generate(BW, xCount, yCount):
    lines = []
    for i in range(yCount):
        line = ''
        for j in range(xCount):
            cropRect = (j * 2, i * 3, (j + 1) * 2, (i + 1) * 3)
            cropped = BW.crop(cropRect)
            input = [cropped.getpixel((0, 0)), cropped.getpixel((1, 0)), cropped.getpixel((0, 1)),
                     cropped.getpixel((1, 1)), cropped.getpixel((0, 2)), cropped.getpixel((1, 2))]
            line += matchChar(input)
        line += '\n'
        lines.append(line)

    return lines

def generateContour(coords, xCount, yCount):
    lines = []
    for i in range(yCount):
        line = ''
        for j in range(xCount):
            x = j * 2
            # xMax = (j+1) * xCount
            y = i * 3
            # yMax = (i+1) * yCount
            input = [coords.get((x, y)), coords.get((x+1, y)), coords.get((x, y+1)),
                     coords.get((x+1, y+1)), coords.get((x, y+2)), coords.get((x+1, y+2))]

            line += matchChar(input)
        line += '\n'
        lines.append(line)
    return lines