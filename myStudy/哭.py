from PIL import Image, ImageQt
from PyQt5.QtCore import QBuffer
buffer = QBuffer()
# QImage转Image
qimage = Image.open(r"C:\Users\ntbgy\PycharmProjects\ai-play-wzry\myStudy\img.png")
image = ImageQt.fromqimage(qimage)

# Image转QImage
qimage = ImageQt.ImageQt(image)