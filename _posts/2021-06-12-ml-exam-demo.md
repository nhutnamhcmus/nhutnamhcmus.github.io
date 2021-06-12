---
title: "Giải đề mẫu Nhập môn Học Máy"
categories:
  - Exam
  - Machine Learning
tags:
  - Exam
  - Machine Learning
toc: true
comments: true
---

Đề thi mẫu môn Nhập môn Học Máy - K2018

# Mô hình tuyến tính

## Câu 01:
Cho mô hình perceptron $y=sign(w_0 + w_1x)$ với $w_0 = 1$, $w_1 = 1$ và bộ dữ liệu kiểm tra $D = {(x_i, y_i)} = {(2, -1), (3, 1), (-2, -1), (-4, 1)}$. Hãy tính **độ chính xác** của mô hình?

**Hướng đi**

Gọi biến toàn cục correct = 0 là số lần dự đoán đúng của mô hình

Với $x_1 = 2$, ta có $y_{\text{pred}} = sign(1 + 2 \times 1) = 1 \ne y_1 = -1$

Với $x_2 = 3$, ta có $y_{\text{pred}} = sign(1 + 3 \times 1) = 1 = y_2 = 1$, correct += 1

Với $x_3 = -2$, ta có $y_{\text{pred}} = sign(1 + (-2) \times 1) = -1 = y_3 = -1$, correct += 1

Với $x_4 = -4$, ta có $y_{\text{pred}} = sign(1 + (-4) \times 1) = -1 \ne y_4 = 1$

Độ chính xác mô hình

$$\text{accuracy} = \frac{1}{n}\sum_{i=1}^{n}\left[sign(w_0 + w_1x_i) == y_i\right] = \frac{1}{4} \times 2 = 0.5$$


## Câu 02:
Cho mô hình logistic regression:
$$y = \frac{1}{1+ exp(w_0 + w_1x)}$$ với $w_0 = 1$, $w_1 = 1$ và bộ dữ liệu kiểm tra $D = {(x_i, y_i)} = {(-3, -1), (-2, 1), (2, -1), (4, 1)}$. Hãy tính **độ lỗi** của mô hình?

**Hướng đi**

Gọi biến toàn cục error = 0 là số lần dự đoán bị sai của mô hình

Với $x_1 = -3$, ta có $y_{\text{pred}} = \left[\frac{1}{1+ exp(1 + 1(-3))} >= 0.5\right] = 1 \ne y_1 = -1$, error += 1

Với $x_2 = -2$, ta có $y_{\text{pred}} = \left[\frac{1}{1+ exp(1 + 1(-2))} >= 0.5\right] = 1 = y_2 = 1$

Với $x_3 = 2$, ta có $y_{\text{pred}} = \left[\frac{1}{1+ exp(1 + 1(2))} >= 0.5\right] = -1 = y_3 = -1$

Với $x_4 = 4$, ta có $y_{\text{pred}} = \left[\frac{1}{1+ exp(1 + 1(4))} >= 0.5\right] = -1 \ne y_4 = 1$, error += 1

Độ lỗi của mô hình 

$$\text{error rate} = \frac{1}{n}\sum_{i=1}^{n}\left[\frac{1}{1+ exp(w_0 + w_1x_i)} \ne y_i\right] = \frac{1}{4} \times 2 = 0.5 = 1 - \text{accuracy}$$

## Câu 03:
Cho mô hình linear regression:
$$y = f(x) = w_0 + w_1x$$ và bộ dữ liệu D, hãy xác định mô hình và trực quan mô hình

|  x 	|   y	|
|---	|---	|
|   1	|   2	|
|   2	|   1	|
|   4	|   2	|

$$ X = 
\begin{bmatrix}
    1       & 1\\
    1       & 2\\
    1       & 4
\end{bmatrix}
$$

$$ y = 
\begin{bmatrix}
    2 \\
    1 \\
    2 
\end{bmatrix}
$$

Tính ma trận chuyển vị của X 

$$ X^T = 
\begin{bmatrix}
    1 & 1 & 1\\
    1 & 2 & 4
\end{bmatrix}
$$

Mô hình tốt nhất

$$W = \left(X^TX\right)^{-1}X^Ty$$

```python
import numpy as np 
import matplotlib.pyplot as plt 

xs = np.array([1, 2, 4])
ys = np.array([2, 1, 2])

one = np.ones(xs.shape[0])
X = np.stack((one, xs)).T

W = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), ys)
print(W)

[Output] [1.5        0.07142857]

x = np.linspace(1,4)
y_predict = W[0] + W[1]*x 
plt.scatter(xs, ys)
plt.plot(x, y_predict, c='red')
plt.show()
```

|![](/assets/images_posts/ml-testt.png)|


# Mô hình cây quyết định và thống kê

Dữ liệu huấn luyện bảng sau 𝐷 có 3 thuộc tính Snow\_Dist, Weekend, Sun và một thuộc tính quyết
định Skiing. Câu 4, 5 và 6 sẽ sử dụng dữ liệu này.

## Câu 04:
Tìm và vẽ tất cả các cây stump sử dụng độ đo **gini** (không cần chạy từng bước)

## Câu 05: 
Tìm và vẽ cây quyết định sử dụng độ đo **entropy** (không cần chạy từng bước)

## Câu 06:
Tìm mô hình **naïve bayes** (không cần chạy từng bước)

# Mạng Neural network 

## Câu 07:
Cho biểu thức $y = (ax+b)(cx+d) + sin(c+d)$ hãy chuyển biểu thức thành đồ thị tính toán và
vẽ đồ thị này

**Hướng đi**: Mình nghĩ vẽ ra cũng dễ :))))


## Câu 08:

|![](/assets/images_posts/com_graph_00.png)|
|:--:| 
| |

Tính giá trị biến ouput $y$ nếu các biến input có giá trị $x_1$ = 3, $x_2$ = −2

**Hướng đi**

Ta có: $x_1 = 3$, $x_2 = −2$, $w_1 = 3$, $w_2 = 4$

$$a \leftarrow w_1 \times x_1 = 3 \times 3 = 9$$

$$b \leftarrow w_2 \times x_2 = -2 \times 4 = -8$$

$$c \leftarrow a \times b = 9 \times -8 = -72$$

$$y \leftarrow \sigma(c) = \frac{1}{1 + exp(c)} = \frac{1}{1 + exp(-72)} = 1$$


## Câu 09:
Tính toán đạo hàm riêng 

$$\frac{\partial y}{\partial x_1}, \frac{\partial y}{\partial x_2}, \frac{\partial y}{\partial w_1}, \frac{\partial y}{\partial w_2}$$

**Hướng đi**

$$\frac{\partial y}{\partial x_1} = \frac{\partial y}{\partial c}\frac{\partial c}{\partial a}\frac{\partial a}{\partial x_1} = \sigma(c)\left[1-\sigma(c)\right]bw_1$$

$$\frac{\partial y}{\partial x_2} = \frac{\partial y}{\partial c}\frac{\partial c}{\partial b}\frac{\partial b}{\partial x_2} = \sigma(c)\left[1-\sigma(c)\right]aw_2$$

$$\frac{\partial y}{\partial w_1} = \frac{\partial y}{\partial c}\frac{\partial c}{\partial a}\frac{\partial a}{\partial w_1} = \sigma(c)\left[1-\sigma(c)\right]bx_1$$

$$\frac{\partial y}{\partial w_2} = \frac{\partial y}{\partial c}\frac{\partial c}{\partial b}\frac{\partial b}{\partial w_2} = \sigma(c)\left[1-\sigma(c)\right]ax_2$$