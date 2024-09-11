from PIL import Image, ImageQt
from PyQt5.QtCore import QBuffer
buffer = QBuffer()
# QImage转Image
qimage = Image.open(r"/工具/img.png")
image = ImageQt.fromqimage(qimage)

# Image转QImage
qimage = ImageQt.ImageQt(image)