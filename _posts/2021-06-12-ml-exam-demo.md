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
header:
  teaser: "/assets/img/neural_ml.png"
---

Đề thi mẫu môn Nhập môn Học Máy - K2018

# Mô hình tuyến tính

## Câu 01: Perceptron Learing Algorithm

Cho mô hình perceptron $y=sign(w_0 + w_1x)$ với $w_0 = 1$, $w_1 = 1$ và bộ dữ liệu kiểm tra $D = {(x_i, y_i)} = {(2, -1), (3, 1), (-2, -1), (-4, 1)}$. Hãy tính **độ chính xác** của mô hình?

**Hướng đi** Gọi biến toàn cục correct = 0 là số lần dự đoán đúng của mô hình

Với $x_1 = 2$, ta có $y_{\text{pred}} = sign(1 + 2 \times 1) = 1 \ne y_1 = -1$

Với $x_2 = 3$, ta có $y_{\text{pred}} = sign(1 + 3 \times 1) = 1 = y_2 = 1$, correct += 1

Với $x_3 = -2$, ta có $y_{\text{pred}} = sign(1 + (-2) \times 1) = -1 = y_3 = -1$, correct += 1

Với $x_4 = -4$, ta có $y_{\text{pred}} = sign(1 + (-4) \times 1) = -1 \ne y_4 = 1$

Độ chính xác mô hình

$$\text{accuracy} = \frac{1}{n}\sum_{i=1}^{n}\left[sign(w_0 + w_1x_i) == y_i\right] = \frac{1}{4} \times 2 = 0.5$$


## Câu 02: Logistic Regression

Cho mô hình logistic regression:
$$y = \frac{1}{1+ exp(w_0 + w_1x)}$$ với $w_0 = 1$, $w_1 = 1$ và bộ dữ liệu kiểm tra $D = {(x_i, y_i)} = {(-3, -1), (-2, 1), (2, -1), (4, 1)}$. Hãy tính **độ lỗi** của mô hình?

**Hướng đi** Gọi biến toàn cục error = 0 là số lần dự đoán bị sai của mô hình

Với $x_1 = -3$, ta có $y_{\text{pred}} = \left[\frac{1}{1+ exp(1 + 1(-3))} >= 0.5\right] = 1 \ne y_1 = -1$, error += 1

Với $x_2 = -2$, ta có $y_{\text{pred}} = \left[\frac{1}{1+ exp(1 + 1(-2))} >= 0.5\right] = 1 = y_2 = 1$

Với $x_3 = 2$, ta có $y_{\text{pred}} = \left[\frac{1}{1+ exp(1 + 1(2))} >= 0.5\right] = -1 = y_3 = -1$

Với $x_4 = 4$, ta có $y_{\text{pred}} = \left[\frac{1}{1+ exp(1 + 1(4))} >= 0.5\right] = -1 \ne y_4 = 1$, error += 1

Độ lỗi của mô hình:

$$\text{error rate} = \frac{1}{n}\sum_{i=1}^{n}\left[\frac{1}{1+ exp(w_0 + w_1x_i)} \ne y_i\right] = \frac{1}{4} \times 2 = 0.5 = 1 - \text{accuracy}$$

## Câu 03: Linear Regression

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

|  # 	|  Snow\_Dist 	|  Weekend 	|  Sun 	|  Skiing 	|
|---	|---	|---	|---	|---	|
|  1 	|  $\leq 100$ 	|  yes 	|  yes 	|  yes 	|
|  2 	|  $\leq 100$ 	|  yes 	|  yes 	|  yes 	|
|  3 	|  $\leq 100$ 	|  yes 	|  no 	|  yes 	|
|  4 	|  $\leq 100$ 	|   no	|  yes 	|  yes 	|
|  5 	|  $> 100$ 	|  yes 	|  yes 	|  yes 	|
|  6 	|  $> 100$ 	|  yes 	|  yes 	|  yes 	|
|  7 	|  $> 100$ 	|  yes 	|  yes 	|  no 	|
|  8 	|  $> 100$ 	|  yes 	|  no 	|  no 	|
|  9 	|  $> 100$ 	|  no 	|  yes 	|  no 	|
|  10 	| $> 100$  	|  no 	|  yes 	|  no 	|

## Câu 04: Decision Tree with Gini

Tìm và vẽ tất cả các cây stump sử dụng độ đo **gini** (không cần chạy từng bước)

**Hướng đi** 

Ta có:

$$\text{Gini}(D) = 1 - \sum_{i=1}^{m}p_i^2$$

$$gini_{A}(D) = \frac{|D_1|}{|D|}gini(D_1)+\frac{|D_2|}{|D|}gini(D_2)$$

$$\Delta gini(A) = gini(D) - gini_A(D)$$

Tổng số bộ: 10

Lớp P (Positive) = 6: Skiing = "yes"

Lớp N (Negative) = 4: Skiiing = "no"

$$gini(D) = 1 - \left(\frac{6}{10}\right)^2 - \left(\frac{4}{10}\right)^2 = 0.48$$

Xem xét thuộc tính **Snow\_Dist**:{$\leq 100$, $> 100$}

$$gini_{\text{Snow_Dist}}(D) = \frac{|D_1|}{10}gini(D_1)+\frac{|D_2|}{10}gini(D_2)$$

$$= \frac{4}{10}\left(1-\left(\frac{4}{4}\right)^2\right) + \frac{6}{10}\left(1-\left(\frac{2}{6}\right)^2-\left(\frac{4}{6}\right)^2\right) = \frac{4}{15} = 0.267$$

Xem xét thuộc tính **Weekend** {yes, no}

$$gini_{\text{Weekend}}(D) = \frac{|D_1|}{10}gini(D_1)+\frac{|D_2|}{10}gini(D_2)$$

$$= \frac{7}{10}\left(1-\left(\frac{5}{7}\right)^2-\left(\frac{2}{7}\right)^2\right) + \frac{3}{10}\left(1-\left(\frac{1}{3}\right)^2-\left(\frac{2}{3}\right)^2\right) = \frac{44}{105} = 0.41905$$

Xem xét thuộc tính **Sun** {yes, no}

$$gini_{\text{Sun}}(D) = \frac{|D_1|}{10}gini(D_1)+\frac{|D_2|}{10}gini(D_2)$$

$$= \frac{8}{10}\left(1-\left(\frac{5}{8}\right)^2-\left(\frac{3}{8}\right)^2\right) + \frac{2}{10}\left(1-\left(\frac{1}{2}\right)^2-\left(\frac{1}{2}\right)^2\right) = \frac{19}{40} = 0.475$$

| Attribute  	|  Split 	|  Gini index 	|  Reduction in impurity $\Delta gini(A) = gini(D) - gini_A(D)$ 	|
|---	|---	|---	|---	|
| Snow_Dist  	| Binary  	|  0.267 	|  0.48 - 0.267 =  0.213	|
| Weekend  	|  Binary 	|  0.367 	| 0.48 - 0.41905 =  0.06095	|
|  Sun 	|  Binary 	|  0.3167 	|  0.48 - 0.475 = 0.005	|

Thuộc tính Snow_Dist được chọn vì có Gini index nhỏ nhất và Reduction in impurity lớn nhất

## Câu 05: Decision Tree with Entropy

Tìm và vẽ cây quyết định sử dụng độ đo **entropy** (không cần chạy từng bước)

**Độ đo Entropy**

$$\text{Entropy}(S) = -\sum_{i=1}^{c}p_ilog_2p_i$$

**Average entropy** on attribute A

$$AE(S, A) = \sum_{v \in \text{Values}(A)}\frac{|S_v|}{|S|}\text{Entropy}(S_v)$$

**Information gain**

$$Gain(S, A) = Entropy(S) - AE(S, A)$$

Tổng số bộ: 10

Lớp P (Positive) = 6: Skiing = "yes"

Lớp N (Negative) = 4: Skiiing = "no"

Áp dụng:

$$Entropy(S) = -\frac{6}{10}log_2\left(\frac{6}{10}\right) -\frac{4}{10}log_2\left(\frac{4}{10}\right) = 0.97095$$

Xem xét thuộc tính **Snow\_Dist**:{$\leq 100$, $> 100$}

- **Snow\_Dist** = $\leq 100$

$$Info([4, 0]) = Entropy([4, 0]) = -\frac{4}{4}log_2\left(\frac{4}{4}\right) -\frac{0}{4}log_2\left(\frac{0}{4}\right) = 0$$

- **Snow\_Dist** = $> 100$

$$Info([2, 4]) = Entropy([2, 4]) = -\frac{2}{6}log_2\left(\frac{2}{6}\right) -\frac{4}{6}log_2\left(\frac{4}{6}\right) = 0.918$$

- Thông tin của thuộc tính **Snow\_Dist**

$$Info([4, 0], [2, 4]) = \frac{4}{10} \times 0 + \frac{6}{10} \times 0.918 = 0.5508$$

- Độ lợi thông tin của thuộc tính **Snow\_Dist**

$$Info([6, 4]) - Info([4, 0], 2, 4]) = Entropy(S) - Info([4, 0], 2, 4]) = 0.42015$$

Xem xét thuộc tính **Weekend** {yes, no}

- **Weekend** = yes

$$Info([5, 2]) = Entropy([5, 2]) = -\frac{5}{7}log_2\left(\frac{5}{7}\right) -\frac{2}{7}log_2\left(\frac{2}{7}\right) = 0.86312$$

- **Weekend** = no

$$Info([1, 2]) = Entropy([1, 2]) = -\frac{1}{3}log_2\left(\frac{1}{3}\right) -\frac{2}{3}log_2\left(\frac{2}{3}\right) = 0.918296$$

- Thông tin của thuộc tính **Weekend**

$$Info([5, 2], [1, 2]) = \frac{7}{10} \times 0.86312 + \frac{3}{10} \times 0.918296 = 0.879673$$

- Độ lợi thông tin của thuộc tính **Weekend**

$$Info([6, 4]) - Info([5, 2], [1, 2]) = Entropy(S) - Info([5, 2], [1, 2]) = 0.09128$$

Xem xét thuộc tính **Sun** {yes, no}

- **Sun** = yes

$$Info([5, 3]) = Entropy([5, 3]) = -\frac{5}{8}log_2\left(\frac{5}{8}\right) -\frac{3}{8}log_2\left(\frac{3}{8}\right) = 0.95443$$

- **Sun** = no

$$Info([1, 1]) = Entropy([1, 1]) = -\frac{1}{2}log_2\left(\frac{1}{2}\right) -\frac{1}{2}log_2\left(\frac{1}{2}\right) = 1$$

- Thông tin của thuộc tính **Sun**

$$Info([5, 3], [1, 1]) = \frac{8}{10} \times  0.95443 + \frac{2}{10} \times 1 = 0.963544$$

- Độ lợi thông tin của thuộc tính **Sun**

$$Info([6, 4]) - Info([5, 3], [1, 1]) = Entropy(S) - Info([5, 3], [1, 1]) = 7.406 \times 10^{-3}$$

Chọn thuộc tính **Snow\_Dist** do có Information Gain cao nhất

## Câu 06: Naive Bayes

Tìm mô hình **naïve bayes** (không cần chạy từng bước)

Mô hình Naive Bayes

- P(Skiing)

|  	|   	|
|---	|---	|
| yes  	|  6/10 	|
| no  	|  4/10 	|

- P(Snow_Dist \| Skiing)

|   	|   	|  Snow_Dist 	|  Snow_Dist 	|
|---	|---	|---	|---	|
|   	|   	|   $\leq 100$ 	|  $> 100$ 	|
| Skiing  	|  yes 	|  4/6 	|  2/6 	|
| Skiing  	|  no 	|  0/4 	|  4/4 	|


- P(Weekend \| Skiing)

|   	|   	|  Snow_Dist 	|  Snow_Dist 	|
|---	|---	|---	|---	|
|   	|   	|   yes 	|  no 	|
| Skiing  	|  yes 	|  5/6 	|  1/6 	|
| Skiing  	|  no 	|  2/4 	|  2/4 	|


- P(Sun \| Skiing)

|   	|   	|  Snow_Dist 	|  Snow_Dist 	|
|---	|---	|---	|---	|
|   	|   	|   yes 	|  no 	|
| Skiing  	|  yes 	|  5/6 	|  1/6 	|
| Skiing  	|  no 	|  3/4 	|  1/4 	|

# Mạng Neural network 

## Câu 07: Convert expression to computational graph
Cho biểu thức $y = (ax+b)(cx+d) + sin(c+d)$ hãy chuyển biểu thức thành đồ thị tính toán và
vẽ đồ thị này

**Hướng đi**: Mình nghĩ vẽ ra cũng dễ :))))


## Câu 08: Caluate on Computational Graph

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


## Câu 09: Derivatives with Computational Graph
Tính toán đạo hàm riêng 

$$\frac{\partial y}{\partial x_1}, \frac{\partial y}{\partial x_2}, \frac{\partial y}{\partial w_1}, \frac{\partial y}{\partial w_2}$$

**Hướng đi**

$$\frac{\partial y}{\partial x_1} = \frac{\partial y}{\partial c}\frac{\partial c}{\partial a}\frac{\partial a}{\partial x_1} = \sigma(c)\left[1-\sigma(c)\right]bw_1$$

$$\frac{\partial y}{\partial x_2} = \frac{\partial y}{\partial c}\frac{\partial c}{\partial b}\frac{\partial b}{\partial x_2} = \sigma(c)\left[1-\sigma(c)\right]aw_2$$

$$\frac{\partial y}{\partial w_1} = \frac{\partial y}{\partial c}\frac{\partial c}{\partial a}\frac{\partial a}{\partial w_1} = \sigma(c)\left[1-\sigma(c)\right]bx_1$$

$$\frac{\partial y}{\partial w_2} = \frac{\partial y}{\partial c}\frac{\partial c}{\partial b}\frac{\partial b}{\partial w_2} = \sigma(c)\left[1-\sigma(c)\right]ax_2$$