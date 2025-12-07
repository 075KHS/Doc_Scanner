import sys
import cv2
import numpy as np
import time
from skimage.filters import threshold_sauvola
import pytesseract

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QFileDialog, QTextEdit, QMessageBox, QGroupBox,
    QSlider
)
from PyQt5.QtGui import QImage, QPixmap

# Tesseract OCR 프로그램 실행 파일 경로 설정
# PyQt에서 OCR 기능을 수행하기 위해 꼭 필요함.
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ESPCN ×4 Super Resolution 모델 (저장 시 고해상도 업스케일링)
# 문서를 고품질로 저장하기 위해 사용.
# 미리보기에서는 사용하지 않으며, 저장 시에만 적용됨.
SR_ENABLED = False
sr = None
try:
    from cv2 import dnn_superres
    print("[INFO] Loading ESPCN ×4 Super Resolution model...")
    sr = dnn_superres.DnnSuperResImpl_create()
    sr.readModel("ESPCN_x4.pb")          # 모델 파일 로드
    sr.setModel("espcn", 4)              # 업스케일 배율 설정
    SR_ENABLED = True
    print("[INFO] ESPCN model loaded successfully.")
except Exception as e:
    # 모델이 없을 경우 기능 비활성화
    print("[WARN] ESPCN model not loaded. SR disabled.", e)


# 문서 코너 4점을 정렬하기 위한 함수
# 좌상, 우상, 우하, 좌하 순서로 재배열해야
# 투시변환(perspective transform)이 정상적으로 작동함.
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)          # x+y 최소 → 좌상, 최대 → 우하
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)  # x−y 최소 → 우상, 최대 → 좌하
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


# 문서 투시 변환 (Perspective Transform)
# 입력: 원본 이미지, 4개 코너 점
# 출력: 문서가 정면으로 보이도록 보정된 이미지
def four_point_transform(img, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # 변환 후 문서의 너비와 높이를 계산
    wA = np.linalg.norm(br - bl)
    wB = np.linalg.norm(tr - tl)
    maxW = int(max(wA, wB))

    hA = np.linalg.norm(tr - br)
    hB = np.linalg.norm(tl - bl)
    maxH = int(max(hA, hB))

    # 변환 목적 좌표
    dst = np.array([
        [0, 0],
        [maxW - 1, 0],
        [maxW - 1, maxH - 1],
        [0, maxH - 1]
    ], dtype="float32")

    # 투시변환 행렬 계산
    M = cv2.getPerspectiveTransform(rect, dst)

    # 문서를 정면으로 펴서 반환
    return cv2.warpPerspective(img, M, (maxW, maxH))


# 문서 미리보기용 전처리 (빠른속도 중심)
# Sauvola Adaptive Thresholding 사용
def preview_doc(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    t = threshold_sauvola(gray, window_size=25, k=0.2)
    return (gray > t).astype(np.uint8) * 255


# 저장용 고품질 전처리
# - 노이즈 제거
# - CLAHE 대비 향상
# - Sauvola Thresholding
def enhance_document(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 노이즈 제거
    gray = cv2.fastNlMeansDenoising(gray, h=10)

    # CLAHE 로 글자 대비 강조
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Sauvola 적응형 이진화
    t = threshold_sauvola(gray, window_size=25, k=0.2)
    return (gray > t).astype(np.uint8) * 255


# ESPCN ×4 업스케일링 (저장 시 고해상도 출력)
def upscale_ESPCN(binary_img):
    if not SR_ENABLED:
        return binary_img

    # ESPCN은 컬러 입력만 지원하므로 GRAY → BGR 변환 후 업스케일
    img_color = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
    up = sr.upsample(img_color)

    # 다시 GRAY 변환
    return cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)


# PyQt 기반 문서 스캐너 메인 클래스
class ScannerApp(QMainWindow):
    def __init__(self):
        super().__init__()

        # UI 기본 설정
        self.setWindowTitle("Document Scanner (PyQt + ESPCN + OCR)")
        self.resize(1300, 700)

        # VideoCapture(카메라/영상파일) 객체
        self.cap = None

        # 주기적으로 프레임을 업데이트하는 Timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # 최근 감지된 문서 윤곽점 / 투시변환 결과
        self.last_doc = None
        self.last_warped = None

        # 매 프레임마다 edge 탐지를 하면 느려지므로 N프레임마다 한번만 문서 탐지 수행
        self.frame_counter = 0
        self.detect_interval = 5

        self.is_video = False  # True면 영상 파일, False면 카메라

        self.init_ui()         # UI 초기화

    # UI 구성 함수
    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        left = QVBoxLayout()

        # 원본 프레임 표시
        self.label_original = QLabel("Original / Camera")
        self.label_original.setFixedSize(800, 400)
        self.label_original.setStyleSheet("background:#222;")
        left.addWidget(self.label_original)

        # 투시보정된 문서 미리보기
        self.label_warped = QLabel("Warped Preview")
        self.label_warped.setFixedSize(800, 250)
        self.label_warped.setStyleSheet("background:#111;")
        left.addWidget(self.label_warped)

        layout.addLayout(left, 2)  # 비율 2

        # -------------------- 우측 패널 --------------------
        right = QVBoxLayout()

        # 입력 소스 선택 버튼 (카메라 / 영상)
        source_box = QGroupBox("Input Source")
        s_layout = QVBoxLayout()

        btn_cam = QPushButton("카메라 시작")
        btn_cam.clicked.connect(self.open_camera)
        s_layout.addWidget(btn_cam)

        btn_vid = QPushButton("영상 파일 열기")
        btn_vid.clicked.connect(self.load_video)
        s_layout.addWidget(btn_vid)

        source_box.setLayout(s_layout)
        right.addWidget(source_box)

        # 영상 프레임 이동 슬라이더
        self.seek_slider = QSlider(Qt.Horizontal)
        self.seek_slider.setEnabled(False)
        self.seek_slider.sliderReleased.connect(self.seek_video)

        right.addWidget(QLabel("Video Position"))
        right.addWidget(self.seek_slider)

        # 스캔 / OCR 버튼
        scan_box = QGroupBox("Scan / OCR")
        sc = QVBoxLayout()

        self.btn_scan = QPushButton("스캔 + 저장 + OCR")
        self.btn_scan.clicked.connect(self.save_scan)
        sc.addWidget(self.btn_scan)

        scan_box.setLayout(sc)
        right.addWidget(scan_box)

        # OCR 결과 출력창
        right.addWidget(QLabel("OCR 결과"))
        self.ocr_text = QTextEdit()
        right.addWidget(self.ocr_text, 1)

        layout.addLayout(right, 1)

        self.statusBar().showMessage("Ready.")

    # 카메라 입력 시작
    def open_camera(self):
        if self.cap:
            self.cap.release()

        self.cap = cv2.VideoCapture(0)
        self.is_video = False
        self.seek_slider.setEnabled(False)

        # 초기화
        self.frame_counter = 0
        self.last_doc = None
        self.last_warped = None

        self.timer.start(30)   # 약 33FPS
        self.statusBar().showMessage("Camera mode activated")

    # 영상 파일 로드
    def load_video(self):
        fname, _ = QFileDialog.getOpenFileName(
            self, "영상 선택", "", "Video (*.mp4 *.avi *.mov *.mkv)"
        )
        if not fname:
            return

        if self.cap:
            self.cap.release()

        self.cap = cv2.VideoCapture(fname)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Error", "영상 파일 열기 실패")
            return

        self.is_video = True
        frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 슬라이더 범위 설정 (0 ~ total-1)
        self.seek_slider.setRange(0, max(1, frame_count - 1))
        self.seek_slider.setEnabled(True)

        # 데이터 초기화
        self.last_doc = None
        self.last_warped = None
        self.frame_counter = 0

        self.timer.start(30)
        self.statusBar().showMessage(f"Video loaded ({frame_count} frames)")

    # 영상 seek (슬라이더로 프레임 이동)
    def seek_video(self):
        if not self.is_video:
            return
        frame_id = self.seek_slider.value()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        self.frame_counter = 0

    # 프레임 업데이트 (카메라/영상 공통)
    def update_frame(self):
        if self.cap is None:
            return

        ok, frame = self.cap.read()
        if not ok:
            return

        self.frame_counter += 1
        orig = frame.copy()

        # 영상일 경우 슬라이더와 연동하여 현재 위치 표시
        if self.is_video and self.seek_slider.isEnabled():
            cur = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            self.seek_slider.blockSignals(True)
            self.seek_slider.setValue(cur)
            self.seek_slider.blockSignals(False)

        # 5프레임마다 탐지
        if self.frame_counter % self.detect_interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)

            # 문서 윤곽 추출: Canny → 팽창 → Morph Close
            edges = cv2.Canny(blur, 50, 150)
            edges = cv2.dilate(edges, np.ones((3, 3)), 1)
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3)), 2)

            # Contour 크기 순으로 상위 10개만 검사 → 사각형(4점) 검사
            contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)

            doc = None
            for c in contours[:10]:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.015 * peri, True)
                if len(approx) == 4:
                    doc = approx.reshape(4, 2)
                    break

            # 문서가 감지되었다면,
            # 투시 변환 + 미리보기용 이진화 이미지 생성
            if doc is not None:
                self.last_doc = doc
                self.last_warped = four_point_transform(orig, doc.astype("float32"))

                preview = preview_doc(self.last_warped)
                self.update_label(self.label_warped, preview, gray=True)

        # 문서 윤곽선을 원본 화면에 표시
        if self.last_doc is not None:
            cv2.polylines(frame, [self.last_doc.astype(int)], True, (0, 255, 0), 2)

        # 원본 화면 출력
        self.update_label(self.label_original, frame, gray=False)

    # QLabel 이미지 업데이트 (RGB/GRAY 자동 처리)
    def update_label(self, label, img, gray=False):
        if gray:
            h, w = img.shape
            q = QImage(img.data, w, h, w, QImage.Format_Grayscale8)
        else:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            q = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)

        pix = QPixmap.fromImage(q)
        pix = pix.scaled(label.width(), label.height(), Qt.KeepAspectRatio)
        label.setPixmap(pix)

    # 스캔 + 저장 + OCR 수행
    # 투시변환 결과 → 고품질 전처리 → (옵션) SR 업스케일 → 저장 → OCR
    def save_scan(self):
        if self.last_warped is None:
            QMessageBox.warning(self, "문서 없음", "문서가 감지되지 않았습니다.")
            return

        # 고품질 전처리
        processed = enhance_document(self.last_warped)

        # Super Resolution 적용 (가능한 경우만)
        doc_sr = upscale_ESPCN(processed)

        # 파일명: 시간 기반
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        img_path = f"scan_{timestamp}.png"
        cv2.imwrite(img_path, doc_sr)

        # OCR 수행 (한국어 + 영어)
        text = pytesseract.image_to_string(doc_sr, lang="kor+eng", config="--psm 6")

        # OCR 결과 저장
        txt_path = f"scan_{timestamp}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)

        # 결과를 UI에 표시
        self.ocr_text.setPlainText(text)
        self.statusBar().showMessage(f"저장 완료: {img_path}")

    # 종료 시 실행
    # 카메라 종료 + OpenCV 창 모두 닫기
    def closeEvent(self, event):
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        event.accept()


# 실행부: PyQt 윈도우 실행
app = QApplication(sys.argv)
win = ScannerApp()
win.show()
sys.exit(app.exec_())
