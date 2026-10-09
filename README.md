<div align="center">

# 📄 Paper Scanner

**Turn your webcam into a document scanner: detect a sheet of paper, straighten it and save a clean scan.**

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)

</div>

---

## ✨ How it works

```mermaid
flowchart LR
    C[📷 Webcam frame] --> P[Pre-process<br/>grey · blur · edges]
    P --> K[Find largest 4-corner contour]
    K --> W[Perspective warp<br/>top-down view]
    W --> S[💾 output_image.jpg]
```

1. Each frame is converted to a thresholded edge image.
2. The **largest four-point contour** (the paper) is found and its corners are ordered.
3. A **perspective transform** flattens it to a 640 × 480 top-down view.
4. The result is shown live and saved as `output_image.jpg`. Press **Q** to quit.

## 🚀 Getting Started

```bash
git clone https://github.com/Arashomranpour/paper_scanner.git
cd paper_scanner
pip install opencv-python numpy
python app.py
```

> 📷 `app.py` opens camera index `1` (`cv2.VideoCapture(1)`); change it to `0` if you only have one camera.
> A PyInstaller build (`build/` and `dist/`) is included in the repository for Windows.

## 📁 Project Structure

```
.
├── app.py         # Scanner: contour detection + perspective warp
├── app.spec       # PyInstaller spec
├── build/  dist/  # Packaged Windows application
```

## 🛠️ Tech Stack

`OpenCV` · `NumPy` · `PyInstaller`
