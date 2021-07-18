---
layout: single
permalink: /msa/linear_regression/multiple_linear_regression
title: "[MSA] Multiple Linear Regression"
author_profile: true
toc: true
comments: True
---
Phần trình bày về mô hình hồi quy tuyến tính bội - multiple linear regression model

## 1) Động lực nghiên cứu khoa học

Trong lĩnh vực nghiên cứu Máy học (Machine Learning), người ta mong muốn đưa ra các giá trị dự đoán với những thông tin mới được đưa vào mô hình nào đó, thông qua những dữ kiện đã biết trước đó sao cho các giá trị dự đoán càng gần với giá trị thực, và đó là bài toán Regression hay được gọi là hồi quy.

Với trường hợp đơn biết, ta có thể mô phỏng mô hình bằng một đường thẳng có dạng $y = ax + b$, rất quen thuộc khi chúng ta học cấp 3. Với trường hợp hai biến, ta có thể mô phỏng mô hình bằng một mặt phẳng có dạng $y = ax + by + c$, và khi số lượng biến của chúng ta lớn hơn hai, mô hình có thể được mô phỏng bằng một siêu phẳng (hyperplane)

## 2) Phát biểu bài toán

Đầu vào bài toán: ta nhận các biến $(X_1, X_2, ..., X_p)$ và các giá trị thực tương ứng với quan sát thứ $i$ đó $y_i \in \mathbb{R}$

Đầu ra của bài toán: $b_0 \in \mathbb{R}$ hệ số chặn hồi quy (regression intercept) và $b_j \in \mathbb{R}$ là các hệ số hồi quy (regression slope) của giá trị dự đoán thứ $j$

Mô hình hồi quy tuyến tính bội như sau:

$$y_i = b_0 + \sum_{j=1}^pb_jx_{ij} + e_i, \forall i \in \{1,...,n\}$$

Trong đó:
- $y_i \in \mathbb{R}$ là giá trị thực tương ứng với quan sát thứ $i$
- $b_0 \in R$ là hệ số chặn hồi quy (regression intercept)
- $b_j \in \mathbb{R}$ là các hệ số hồi quy (regression slope) của giá trị dự đoán thứ $j$
- $x_{ij} \in \mathbb{R}$ là giá trị dự đoán thứ $j$ cho quan sát thứ $i$
- $e_i \overset{\underset{\mathrm{iid}}{}}{\sim} \mathcal{N}(0, \sigma^2)$ một Gaussian Error

Hay viết gọn hơn bằng dạng ma trận như sau

$$\mathbf{y = Xb + e}$$

Trong đó:
- $\mathbf{y} = (y_1, y_2, ..., y_n)' \in \mathbb{R}^n$ có kích thước $n \times 1$ là vector các giá trị tương ứng với biến quan sát
- $\mathbf{X = [1_n, x_1, x_2, ..., x_p]} \in \mathbb{R}^{n\times (p+1)}$ có kích thước  $n \times (p + 1)$ là ma trận biến quan sát
- $\mathbf{b} = (b_0, b_1, b_2, ..., b_p)' \in \mathbb{R}^{p+1}$ có kích thước $(p + 1) \times n$ là vector các hệ số hồi quy (coefficient vector)
- $\mathbf{e} = (e_1, e_2, ..., e_n)' \in \mathbb{R}^n$ là vector độ lỗi (error vector)

Mô hình là một mô hình hồi quy và tuyến tính (linear) với những tham số $b_0, b_1, ..., b_p$ vì chúng ta mô hình hóa một biến (Y) tương ứng như một hàm dự đoán các giá trị $(X_1, X_2, ..., X_p)$. Hơn nữa, mô hình là bội (multiple) vì chúng ta có nhiều hơn một bộ dự đoán, nếu chỉ có một bộ dự đoán, mô hình trở thành một mô hình Simple Linear Regression

Các giả định cơ sở của mô hình hồi quy tuyến tính bội
- Mối quan hệ giữa $X_j$ và $Y$ là tuyến tính
- $x_{ij}$ và $y_i$ là những biến ngẫu nhiên đã biết
- $e_i \overset{\underset{\mathrm{iid}}{}}{\sim} \mathcal{N}(0, \sigma^2)$ là biến ngẫu nhiên chưa biết
- $b_0, b_1, ..., b_p$ là các hằng số chưa biết
- $(y_i \|\ x_{i1}, x_{i2}, ..., x_{ip}) \overset{\underset{\mathrm{iid}}{}}{\sim} \mathcal{N}(b_0 + \sum_{j=1}^pb_jx_{ij} , \sigma^2)$

## 3) Phương pháp

### 3.1 Ước lượng tham số mô hình bằng phương pháp bình phương tối tiểu (Ordinary Least Squares)

Bài toán bình phương tối tiểu (OLS)

$$\underset{\mathbf{b} \in \mathbb{R}^{p+1}}{min} ||\mathbf{y -Xb}||^2 = \underset{\mathbf{b} \in \mathbb{R}^{p+1}}{min} \sum_{i=1}^{n}\left(y_i - b_0 - \sum_{j=1}^pb_jx_{ij}\right)^2$$

Với ma trận $\mathbf{X} \in \mathbb{R}^{p+1}$, $p+1 \leq n$, $rank(\mathbf{X}) = n$. Xét hàm số:

$$f: \mathbb{R}^n \rightarrow \mathbb{R}$$

$$f(x) =  ||\mathbf{y -Xb}||^2$$

Ta có:

$$
\begin{gather*}
f(x) = \left< \mathbf{y -Xb}, \mathbf{y -Xb}\right>\\
= (\mathbf{y -Xb})^\text{T}(\mathbf{y -Xb})\\
= (\mathbf{y}^{\text{T}} - \mathbf{b}^{\text{T}}\mathbf{X}^{\text{T}})(\mathbf{y - Xb})\\
= \mathbf{b}^{\text{T}}\mathbf{X}^{\text{T}}\mathbf{Xb} - \mathbf{y}^{\text{T}}\mathbf{Xb} - \mathbf{b}^{\text{T}}\mathbf{X}^{\text{T}}\mathbf{y} + \mathbf{y}^{\text{T}}\mathbf{y}\\
= \mathbf{b}^{\text{T}}(\mathbf{X}^{\text{T}}\mathbf{X})\mathbf{b} - (2 \mathbf{y}^{\text{T}}\mathbf{X})\mathbf{b} + \mathbf{y}^{\text{T}}\mathbf{y}
\end{gather*}
$$

Vì $\mathbf{y}^{\text{T}}\mathbf{X}\mathbf{b} = \mathbf{b}^{\text{T}}\mathbf{A}^{\text{T}}\mathbf{y}$ nên ta có:

$$
\begin{gather*}
\nabla f(x) = 2(\mathbf{X}^{\text{T}}\mathbf{X})\mathbf{b} - 2\mathbf{X}^{\text{T}}\mathbf{y}\\
\nabla^2 f(x) = 2(\mathbf{X}^{\text{T}}\mathbf{X})
\end{gather*}
$$

Mà do $\mathbf{b}  \in \mathbb{R}^{p+1}$

$$
\begin{gather*}
\mathbf{b}\nabla^2 f(x) \mathbf{b} = 2[\mathbf{b}^{\text{T}}(\mathbf{X}^{\text{T}}\mathbf{X})\mathbf{b}]\\
= 2[\mathbf{b}^{\text{T}}(\mathbf{X}\mathbf{b})]\\
= 2||\mathbf{X}\mathbf{b}||^2 \geq 0
\end{gather*}
$$

Vì thế mà $\nabla^2 f(x)$ là ma trận xác định dương mặc khác $rank(\mathbf{X}^{\text{T}}\mathbf{X}) = n$
nên $\mathbf{X}^{\text{T}}\mathbf{X}$ khả nghịch

Ta thu được nghiệm của bài toán $\hat{\mathbf{b}} = (\mathbf{X}^{\text{T}}\mathbf{X})^{-1}\mathbf{X}^{\text{T}}\mathbf{y}$

### 3.2 Ước lượng tham số mô hình bằng phương pháp triển vọng cực đại (maximum likelihood)

### 3.3 Đánh giá mô hình

### 3.4 Suy luận (inference) và dự đoán (prediction)

## 4) Ý nghĩa hình học

## 5) Ví dụ minh họa
