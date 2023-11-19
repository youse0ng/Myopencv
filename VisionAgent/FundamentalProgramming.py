from PyQt5.QtWidgets import *
import sys
import winsound # 삑소리를 내는데 사용하는 패키지

class BeepSound(QMainWindow): # QmainWindow (윈도우를 생성하고 관리하는 함수)
    def __init__(self):
        super().__init__()
        self.setWindowTitle('삑 소리 내기') # 윈도우 이름과 위치 저장
        self.setGeometry(400,400,500,100) # 윈도우를 (400,400) 위치에 초기 배치하고 너비와 높이를 500,100으로 설정


        shortBeepButton=QPushButton('짧게 삑',self) # 버튼 생성 위젯
        longBeepButton=QPushButton('길게 삑',self) # 버튼 생성 위젯
        quitButton=QPushButton('나가기',self) # 버튼 생성 위젯
        
        self.label=QLabel('환영합니다.',self) # 레이블 위젯

        shortBeepButton.setGeometry(10,10,100,30) # 버튼 위치와 크기 지정
        longBeepButton.setGeometry(110,10,100,30)
        quitButton.setGeometry(210,10,100,30)
        self.label.setGeometry(10,40,500,70)

        shortBeepButton.clicked.connect(self.shortBeepFunction) # 콜백 함수 지정
        longBeepButton.clicked.connect(self.longBeepFunction) 
        quitButton.clicked.connect(self.quitFunction)

    def shortBeepFunction(self):
        self.label.setText('주파수 1000으로 0.5초동안 삑소리를 낸다.')
        winsound.Beep(1000,500)
    
    def longBeepFunction(self):
        self.label.setText('주파수 1000으로 3초동안 삑 소리를 낸다.')
        winsound.Beep(1000,3000)
    
    def quitFunction(self):
        self.close()

app=QApplication(sys.argv)
win=BeepSound()
win.show()
app.exec_()