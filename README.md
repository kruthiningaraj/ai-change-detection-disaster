# 🌍 AI-Based Change Detection for Disaster Identification  
### 🎓 Undergrad Capstone Project | Computer Vision & Pattern Recognition

This repository implements our **undergraduate capstone project** for **AI-driven disaster identification using bi-temporal satellite images**.

📄 **[Read Our Published Research Paper (IRJET)](research_paper/IRJET-V9I7530.pdf)**  

---

## 🧠 Approach:
- **Change Detection:** VGG-19 feature extraction + Reduced Sum of Squares (RSS) + Otsu thresholding  
- **Disaster Probability Estimation:** Based on white-pixel ratio from change maps  
- **Binary Classification:** InceptionV3 for classifying Hurricane vs Earthquake post-disaster images  

---

## 📂 Dataset:
We use the **xView2 (xBD) dataset**: [https://xview2.org/dataset](https://xview2.org/dataset)

### Structure after download:
```
data/xBD/
 ├── train/
 │    ├── pre_disaster/
 │    ├── post_disaster/
 │    └── labels/
 └── test/
      ├── pre_disaster/
      ├── post_disaster/
      └── labels/
```

---

## 🚀 Getting Started:
1️⃣ Clone repository:  
```bash
git clone https://github.com/your-username/ai-change-detection-disaster.git
cd ai-change-detection-disaster
```

2️⃣ Install dependencies:  
```bash
pip install -r requirements.txt
```

3️⃣ Download dataset from xView2 and organize under `data/xBD/`.

4️⃣ Train classifier:  
```bash
python src/train_classifier.py
```

5️⃣ Run change detection & inference:  
```bash
python src/inference.py --pre_img <path> --post_img <path> --classifier models/disaster_classifier.h5
```

---

## 🏆 Highlights:
- Undergrad Capstone in **Computer Vision & Pattern Recognition**
- Implements **VGG-19 change detection pipeline** with disaster probability computation
- Integrated **binary disaster classification (Hurricane vs Earthquake)**
- Includes published research paper

---

## 📜 License:
MIT License © 2025
