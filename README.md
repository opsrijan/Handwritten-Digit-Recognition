# Handwritten Digit Recognition with Neural Networks

This project implements a neural network using TensorFlow and Keras to recognize handwritten digits from the MNIST dataset.

---

## 📚 Dataset

The model uses the **MNIST dataset**, which contains:

- 60,000 training images
- 10,000 testing images

Each image is a 28x28 grayscale image representing a digit from 0 to 9.

---

## 🛠️ Tech Stack

- **Language:** Python  
- **Libraries:** TensorFlow, Keras, NumPy, Pandas, Matplotlib

---

## ⚙️ Data Preprocessing

1. **Normalization:** Scales pixel values to the range [0, 1].  
2. **Flattening:** Converts each 28x28 image into a flat 784-dimensional vector.  
3. **One-Hot Encoding:** Converts digit labels (0–9) into binary class matrices.  
4. **Train-Validation Split:** 80-20 split from the training dataset for validation.

---

## 🧠 Model Architecture

A simple fully connected feedforward neural network with the following structure:

- **Input Layer:** 784 neurons (flattened 28x28 image)
- **Hidden Layer 1:** Dense layer with 512 neurons, ReLU activation
- **Hidden Layer 2:** Dense layer with 256 neurons, ReLU activation
- **Output Layer:** Dense layer with 10 neurons, Softmax activation

### 🏋️ Training Configuration

- **Loss Function:** Categorical Crossentropy  
- **Optimizer:** Adam  
- **Epochs:** 10  
- **Batch Size:** 128  

---

## 📈 Results

- Achieved approximately **98% accuracy** on the test set.
- Training and validation accuracy/loss plotted over epochs.

---

## 🔍 Visualization

- Displays loss/accuracy graphs to monitor model performance.
- Visualizes predictions on test images, comparing **true vs predicted** labels.

---

## 🧪 How to Run

1. Install required packages:

```bash
pip install numpy pandas matplotlib tensorflow keras
```

2. Run the script:

```bash
python digit_recognition.py
```

---

## 👨‍💻 Author

**Srijan Kumar**  
Project under self-study in machine learning and neural networks.
