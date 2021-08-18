import sys
from mainWindow import Window
from PyQt5.QtWidgets import QApplication

#Opening the main window
def main():
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    app.exec()

if __name__ == '__main__':
    main()