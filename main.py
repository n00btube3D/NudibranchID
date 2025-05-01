import sys

# UI Windows
from ui_windows.MainWindow import Ui_MainWindow
from ui_windows.Results import Ui_Form

# QT headers
from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from PySide6.QtWidgets import (
    QDialog,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsSceneMouseEvent,
    QGraphicsRectItem,
)
from PySide6.QtGui import QImage, QPixmap, QPainter
from PySide6.QtCore import QStandardPaths, QPointF, QRectF, QRect, Qt

import json

# Pillow
from PIL import ImageQt

# pytorch
import torch
import torchvision.transforms as transforms
import torch.jit as jit

# Modify model here as needed
classes = json.load(open("pytorch_model/classes.txt"))
model = jit.load("pytorch_model/yolov11.torchscript")


def cropAndPredict(
    pixmap: QPixmap, rect: QGraphicsRectItem | None, confidence: float = 0.5, k: int = 5
):
    """
    Crop the given pixmap using the provided rectangle and predict the top k classes.

    Parameters:
    pixmap (QPixmap): The source pixmap to be cropped and predicted.
    rect (QGraphicsRectItem | None): The rectangle item defining the crop area. If None, the entire pixmap is used.
    confidence (float): Minimum probability required to determine a species.
    k: Maximum Top-K predictions.

    Returns:
    tuple: A tuple containing the cropped pixmap and a list of the top 5 predicted classes
    """
    if rect is not None:
        pixmap = extract_pixmap(pixmap, rect.rect().toRect())

    image = ImageQt.fromqpixmap(pixmap)
    image = image.convert("RGB")
    transform = transforms.Compose([transforms.Resize(640), transforms.ToTensor()])
    image = transform(image).unsqueeze(0)
    outputs = model(image)  # type: torch.Tensor
    outputs = outputs.flatten()

    # Set elements lower than confidence threshold to 0
    outputs = torch.where(outputs < confidence, 0, outputs)

    non_zero_indices = torch.nonzero(outputs, as_tuple=True)[0]

    non_zero_values = outputs[non_zero_indices]

    predictions = non_zero_indices[
        torch.argsort(non_zero_values, descending=True)[:k]
    ].tolist()

    # Numerical Indices loaded from json are strings
    predictions = [str(idx) for idx in predictions]
    return pixmap, [classes[idx] for idx in predictions]


class CustomGraphicsScene(QGraphicsScene):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Guard against mouseDoubleClick event consuming mousePress event
        self.clicks = 0

        self.topLeft = QPointF()
        self.bottomRight = QPointF()

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent):
        self.topLeft = event.scenePos()

        # Guard against mouseDoubleClick event consuming mousePress event
        self.clicks = 1
        super().mousePressEvent(event)
        return

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent):
        if self.clicks == 1:
            self.bottomRight = event.scenePos()

            # Do not add rectangle if there is no image loaded.
            # The image itself is an item.
            if len(self.items()) != 0:
                # Remove existing rectangle if exists.
                # This for loop will not take long because there will only be 2 items max.
                for item in self.items():
                    if isinstance(item, QGraphicsRectItem):
                        self.removeItem(item)
                self.createRect()
            self.clicks = 0
        super().mouseReleaseEvent(event)
        return

    def createRect(self):
        """
        Create a rectangle from the points saved in self, and add it to the scene.
        """
        self.limitPointPos()
        rect = QRectF(self.topLeft, self.bottomRight)
        self.addRect(rect)

    def limitPointPos(self):
        """
        Make sure the rectangle will not be outside the image itself by modifying the stored point information.
        Doesn't return anything because modification is done in-place (no new points are created).
        """

        if self.topLeft.x() < 0.0:
            self.topLeft.setX(0.0)
        if self.topLeft.x() > self.width():
            self.topLeft.setX(self.width())
        if self.topLeft.y() < 0.0:
            self.topLeft.setY(0.0)
        if self.topLeft.y() > self.height():
            self.topLeft.setY(self.height())

        if self.bottomRight.x() < 0.0:
            self.bottomRight.setX(0.0)
        if self.bottomRight.x() > self.width():
            self.bottomRight.setX(self.width())
        if self.bottomRight.y() < 0.0:
            self.bottomRight.setY(0.0)
        if self.bottomRight.y() > self.height():
            self.bottomRight.setY(self.height())
        return


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.setWindowTitle("Nudibranch ID")

        self.pushButton.setCheckable(True)
        self.pushButton.clicked.connect(self.submitToModel)

        self.actionOpen_File.triggered.connect(self.opendialog)
        self.actionClose.triggered.connect(self.terminate)

        self.graphicScene = CustomGraphicsScene(parent=self)
        self.graphicsView.setScene(self.graphicScene)

    def submitToModel(self):
        if len(self.graphicScene.items()) == 0:
            QMessageBox.critical(
                self, "No Image Loaded!", "Load an image first, then try again."
            )
            return

        rectangle = None
        # Find the items
        for item in self.graphicScene.items():
            if isinstance(item, QGraphicsPixmapItem):
                pixmap = item
            if isinstance(item, QGraphicsRectItem):
                rectangle = item

        pixmap, textPred = cropAndPredict(pixmap.pixmap(), rectangle, 0.5, 5)

        # textPred looks like this
        # ['species 1', ...]
        # Transform it into:
        # ['<a href="google.com/search?q=species+1">species 1</a>', ...]

        finalTextList = list(
            map(
                lambda x: f"<a href='google.com/search?q={x.replace(' ', '+')}&tbm=isch'>{x}</a>",
                textPred,
            )
        )

        # Consolidate results and send them to the subwindow.
        resultWindow = SubWindow(self)
        item = QGraphicsPixmapItem(pixmap)
        resultWindow.graphicsScene.addItem(item)
        resultWindow.setWindowTitle("Result")

        if (n_pred := len(finalTextList)) > 0:
            if n_pred == 1:
                resultWindow.label.setText(
                    f"Top 1 prediction: {str(finalTextList)[1:-1]}"
                )
            else:
                resultWindow.label.setText(
                    f"Top {n_pred} predictions: {str(finalTextList)[1:-1]}"
                )
        else:
            resultWindow.label.setText(
                "The model is not able to confidently identify any nudibranchs."
            )

        resultWindow.show()
        resultWindow.graphicsView.fitInView(
            resultWindow.graphicsScene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio
        )

    def opendialog(self):
        fileSelect = QFileDialog(self)
        fileSelect.setFileMode(QFileDialog.FileMode.ExistingFile)
        # A mouthful...
        fileSelect.setDirectory(
            QStandardPaths.standardLocations(
                QStandardPaths.StandardLocation.DesktopLocation
            )[-1]
        )
        fileSelect.show()

        if fileSelect.exec():
            # Full path of selected file
            chosenFile = fileSelect.selectedFiles()
            if len(chosenFile) > 1:
                QMessageBox.information(
                    self, "Too many images selected", "Please select one image only."
                )
            self.load_image(chosenFile[0])

    def load_image(self, filePath: str):
        self.image = QImage()
        if self.image.load(filePath):
            # Remove loaded image and annotations if any.
            for item in self.graphicScene.items():
                self.graphicScene.removeItem(item)

            # Create pixmap item and load the image.
            pic = QGraphicsPixmapItem()
            pic.setPixmap(QPixmap.fromImage(self.image))
            self.graphicScene.addItem(pic)

            self.graphicsView.fitInView(
                self.graphicScene.itemsBoundingRect(),
                Qt.AspectRatioMode.KeepAspectRatio,
            )

    def terminate(self):
        app.quit()


class SubWindow(QDialog, Ui_Form):
    def __init__(self, parent):
        super().__init__(parent=parent)
        self.setupUi(self)
        self.label.setTextInteractionFlags(
            Qt.TextInteractionFlag.LinksAccessibleByMouse,
        )
        self.label.setOpenExternalLinks(True)
        self.graphicsScene = QGraphicsScene(parent=self)
        self.graphicsView.setScene(self.graphicsScene)


def extract_pixmap(source_pixmap: QPixmap, rect: QRect):
    # Create an empty pixmap with the size of the rectangle
    extracted_pixmap = QPixmap(rect.size())

    # Create a painter to draw on the extracted pixmap
    painter = QPainter(extracted_pixmap)

    # Draw the part of the source pixmap defined by the rectangle onto the extracted pixmap
    painter.drawPixmap(
        0, 0, source_pixmap, rect.x(), rect.y(), rect.width(), rect.height()
    )

    return extracted_pixmap


app = QApplication(sys.argv)
window = MainWindow()

window.show()
app.exec()
