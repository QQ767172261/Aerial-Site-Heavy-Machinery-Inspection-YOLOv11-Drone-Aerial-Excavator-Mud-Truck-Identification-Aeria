
航拍视角工程车辆检测数据集   基于YOLOv11航拍工地重型机械设备检测系统![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/8bf2a33e8dde4ca4af0c13a25b9ffc37.png)
，1264张，yolo和voc两种标注方式
1类，标注数量：
construct heavy equipment: 1496
image num: 1264
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/905c123c13fa493bb9aee7df3bc56b3b.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/756f5fac3e704eb19bf9667f05bd36f4.png)

2.模型代码：模型训练使用yolov11n训练，30个epoch训练结果，map如描述图所示。![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/2697d2ac2a8d46e7aa4c34019921d466.png)

3.qt界面：运行界面采用qt编写，本项目已经训练好模型，配置好环境后可直接使用，运行效果见描述图像![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/cf684a9d5c094722969b00e55ca3f3fb.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/dad746a282384bed83c2a92e09a7eaba.png)
# 基于YOLOv11航拍工地重型机械设备检测系统完整代码
## 1、数据集配置文件 `construct_heavy.yaml`
```yaml
path: ./construct_dataset
train: images/train
val: images/val
test: images/test

names:
  0: construct heavy equipment
```

## 2、训练脚本 `train_construct.py`
```python
from ultralytics import YOLO

if __name__ == '__main__':
    # 加载YOLOv11n模型
    model = YOLO("yolo11n.pt")
    # 开始训练
    train_result = model.train(
        data="construct_heavy.yaml",
        epochs=80,
        imgsz=640,
        batch=8,
        device=0,
        project="construct_output",
        name="yolo11_construct",
        conf=0.25
    )
```

## 3、推理脚本 `predict_construct.py`
```python
from ultralytics import YOLO

model = YOLO("./construct_output/yolo11_construct/weights/best.pt")

# 图片推理
res = model.predict("test.jpg", save=True, conf=0.25)
for r in res:
    print("检测目标数量：", len(r.boxes))
    for box in r.boxes:
        cls = model.names[int(box.cls)]
        conf = float(box.conf)
        xyxy = box.xyxy.tolist()[0]
        print(f"类别:{cls},置信度:{conf:.3f},坐标:{xyxy}")
```

## 4、PyQt5可视化界面（和截图界面一致）`construct_gui.py`
```python
import sys
import cv2
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel,
                             QFileDialog, QTableWidget, QTableWidgetItem, QComboBox, QLineEdit)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from ultralytics import YOLO

class ConstructDetectUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("基于YOLOv11的航拍工地重型机械设备检测系统")
        self.setGeometry(50,50,1300,900)
        self.model = YOLO("./construct_output/yolo11_construct/weights/best.pt")
        self.img_path = None
        self.result_img = None
        self.init_ui()

    def init_ui(self):
        # 图像显示区
        self.img_label = QLabel("图像预览", self)
        self.img_label.setGeometry(30, 80, 800, 600)
        self.img_label.setStyleSheet("border:1px solid #333;")
        self.img_label.setAlignment(Qt.AlignCenter)

        # 文件导入模块
        label_import = QLabel("文件导入", self)
        label_import.setGeometry(860,80,100,30)
        self.btn_img = QPushButton("选择图片", self)
        self.btn_img.setGeometry(860,110,160,35)
        self.btn_img.clicked.connect(self.load_image)
        self.btn_video = QPushButton("选择视频文件", self)
        self.btn_video.setGeometry(860,150,160,35)
        self.btn_cam = QPushButton("开启摄像头", self)
        self.btn_cam.setGeometry(860,190,160,35)

        # 检测结果面板
        label_result = QLabel("检测结果", self)
        label_result.setGeometry(860,250,100,30)
        self.label_time = QLabel("用时：0s", self)
        self.label_time.setGeometry(860,280,180,30)
        self.label_num = QLabel("目标数目：0", self)
        self.label_num.setGeometry(860,310,180,30)

        self.combo_target = QComboBox(self)
        self.combo_target.addItems(["全部","construct heavy equipment"])
        self.combo_target.setGeometry(860,350,160,30)
        self.label_combo = QLabel("目标选择：", self)
        self.label_combo.setGeometry(860,328,80,30)

        self.label_conf_text = QLabel("置信度", self)
        self.label_conf_text.setGeometry(860,390,80,30)
        self.line_conf = QLineEdit("0.25", self)
        self.line_conf.setGeometry(920,390,80,30)

        self.label_pos = QLabel("目标位置", self)
        self.label_pos.setGeometry(860,430,100,30)
        self.label_xmin = QLabel("xmin: ", self)
        self.label_xmin.setGeometry(860,460,80,30)
        self.label_ymin = QLabel("ymin: ", self)
        self.label_ymin.setGeometry(940,460,80,30)
        self.label_xmax = QLabel("xmax: ", self)
        self.label_xmax.setGeometry(860,490,80,30)
        self.label_ymax = QLabel("ymax: ", self)
        self.label_ymax.setGeometry(940,490,80,30)

        # 操作按钮
        self.btn_detect = QPushButton("开始检测", self)
        self.btn_detect.setGeometry(860,540,160,40)
        self.btn_detect.clicked.connect(self.run_detect)
        self.btn_save = QPushButton("保存结果", self)
        self.btn_save.setGeometry(860,590,160,40)
        self.btn_exit = QPushButton("退出", self)
        self.btn_exit.setGeometry(860,640,160,40)

        # 结果表格
        self.table = QTableWidget(self)
        self.table.setGeometry(30,700,1220,160)
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["序号","文件路径","类别","置信度","坐标位置"])

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "打开图片", "", "*.jpg;*.png;*.bmp")
        if file_path:
            self.img_path = file_path
            self.img_label.setPixmap(QPixmap(file_path).scaled(self.img_label.size(), Qt.KeepAspectRatio))

    def run_detect(self):
        if not self.img_path:
            return
        conf_thresh = float(self.line_conf.text())
        results = self.model(self.img_path, conf=conf_thresh)[0]
        img_out = results.plot()
        self.result_img = img_out
        h,w,c = img_out.shape
        qimg = QImage(img_out.data, w, h, c*w, QImage.Format_RGB888).rgbSwapped()
        self.img_label.setPixmap(QPixmap.fromImage(qimg).scaled(self.img_label.size(), Qt.KeepAspectRatio))

        # 填充表格
        self.table.setRowCount(0)
        target_filter = self.combo_target.currentText()
        box_list = []
        for box in results.boxes:
            cls_name = self.model.names[int(box.cls)]
            if target_filter != "全部" and cls_name != target_filter:
                continue
            box_list.append(box)
        self.label_num.setText(f"目标数目：{len(box_list)}")

        for idx,box in enumerate(box_list):
            cls_name = self.model.names[int(box.cls)]
            conf = float(box.conf)
            xyxy = [round(i.item(),1) for i in box.xyxy[0]]
            xmin,ymin,xmax,ymax = xyxy
            self.label_xmin.setText(f"xmin: {xmin}")
            self.label_ymin.setText(f"ymin: {ymin}")
            self.label_xmax.setText(f"xmax: {xmax}")
            self.label_ymax.setText(f"ymax: {ymax}")

            self.table.insertRow(idx)
            self.table.setItem(idx,0, QTableWidgetItem(str(idx+1)))
            self.table.setItem(idx,1, QTableWidgetItem(self.img_path))
            self.table.setItem(idx,2, QTableWidgetItem(cls_name))
            self.table.setItem(idx,3, QTableWidgetItem(f"{conf:.3f}"))
            self.table.setItem(idx,4, QTableWidgetItem(str(xyxy)))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = ConstructDetectUI()
    win.show()
    sys.exit(app.exec_())
```

## 环境安装
```bash
pip install ultralytics torch opencv-python pyqt5
```
关键词：航拍工地重型机械检测、YOLOv11、无人机航拍、挖掘机渣土车识别、工地设备智能监控

