---
title: "Giải đề mẫu Nhận Dạng"
categories:
  - Exam
  - Recognition
tags:
  - Exam
  - Recognition

toc: true
comments: true
header:
  teaser: "/assets/images_posts/recognition.png"
---

Giải đề mẫu Nhận Dạng - K2018 HCMUS

# Đề 01

## Câu 01: Local Binary Pattern

a) Trình bày các kiến thức về đặc trưng mẫu nhị phân cục bộ $LBP_{(P, R)}$

- (1) Phương pháp xác định giá trị
- (2) Ưu điểm
- (3) Nhược điểm
- (4) Các vị dụ minh hoạ cho các kiến thức đẵ trình bày

b) Sự khác biệt giữa đặc trưng  $LBP_{(P, R)}$ so với đặc trưng $LBP_{(P, R)^{ri}}$. Trình bày ví dụ cho cách tính đặc trưng $LBP_{(P, R)^{ri}}$

**Hướng đi**
a) Trình bày các kiến thức về đặc trưng mẫu nhị phân cục bộ $LBP_{(P, R)}$

**(1) Phương pháp xác định giá trị**

Phương pháp xác định giá trị LBP sơ khai: Được tính toán bằng cách tại mỗi điểm ảnh, xét 8 điểm xung quanh nó

- Bước 01: Các điểm xung quanh có giá trị nhỏ hơn điểm đang xét sẽ được đánh dấu là 0, lớn hơn điểm đang xét sẽ được đánh dấu là 1
- Bước 02: Sau đó, các giá trị sau khi tính toán phân ngưỡng ở trên sẽ được nhân với ma trận trọng số và được sử dụng để tính giá trị LBP của điểm đang xét

$$\begin{bmatrix}1 & 2 & 4 \\128 & & 8\\64 & 32 & 16\end{bmatrix}$$

![](/assets/images_posts/orginal_lbp.png)

Phương pháp xác định giá trị LBP cải tiến: Xét các điểm thuộc đường tròn với tâm là điểm đang xét. Ta gọi $(P,R)$ là vùng lân cận gồm $P$ điểm trên một đường tròn bán kinh $R$

$$T = t(g_c, g_0, ..., g_{p-1})$$

Trong đó:

- $g_c$ và $(g_0, ..., g_{p-1})$ là giá trị trên ảnh xám của điểm trung tâm và các điểm trên đường tròn bán kính $R$

Ta có thể xấp xỉ công thức trên bằng cách lấy từng điểm trên đường tròn bán kinh $R$ trừ đi giá trị trung tâm $g_c$ do các giá trị chỉ thể hiện cường độ sáng của điểm ảnh.

$$T \approx t(g_0 - g_c, g_1 - g_c, ..., g_{p-1} - g_c)$$

Để không bị ảnh hưởng bởi các giá trị của các điểm ảnh, chuẩn hoá công thức trên bằng việc xét dấu

$$T \approx t(sign(g_0 - g_c), sign(g_1 - g_c), ..., sign(g_{p-1} - g_c))$$

Trong đó:

$$sign(x) = \begin{cases}1, x \geq 0\\ 0, x < 0\end{cases}$$

Giá trị LBP được tính như sau:

$$LBP_{P, R} = \sum_{i=0}^{p-1} sign(g_i-g_c) \times 2^i$$

**(2) Ưu điểm**
- Bất biến đối với bất kỳ phép biến đổi đơn điệu nào trên ảnh xám (grayscale) nên hiệu quả cho việc làm giảm tỷ lệ từ chối sai (FRR) cho mỗi đối tượng, khi đối tuợng đó được chụp tại những điều kiện sáng/ tối khác nhau
- Lượng tử hoá vector là không cần thiết
- Tính toán một cách đơn giản

![](/assets/images_posts/lbp_pr.png)

**(3) Nhược điểm**
- Chưa quan tâm đến phép quay, đặc trưng LBP áp dụng trên hình ảnh xoay của cùng một đối tượng sẽ tạo ra các giá trị khác nhau

![](/assets/images_posts/stuck_lbp_pr.png)

- Có vấn đề khi số điểm lân cận tăng quá cao, giá trị LBP sẽ tăng lên rất cao

b) Sự khác biệt giữa đặc trưng $LBP_{(P, R)}$ so với đặc trưng $LBP_{(P, R)^{ri}}$. Trình bày ví dụ cho cách tính đặc trưng $LBP_{(P, R)^{ri}}$

Đặc trưng $LBP_{(P, R)^{ri}}$ khắc phục nhược điểm của đặc trưng $LBP_{(P, R)}$: bất biến với phép xoay, chọ giá trị nhỏ nhất để đại diện cho tất cả các ảnh xoay (mỗi ảnh được xoay theo hướng của vòng tròn luợng giác một góc xác định cho trước)

Để kiểm tra một mẫu kết cấu có phải là "Uniform Patterns" hay không, ta áp dụng công thức

$$U(LBP_{P,R}) = |sign(g_{p-1} - g_c) - sign(g_0 - g_c)| + \sum_{i=1}^{p-1}|sign(g_i - g_c) - sign(g_{i-1} - g_c)|$$

Khi $U \leq 2$, mẫu kết cấu gọi là "Uniform Patterns", ngược lại là "Non-Uniform Patterns"

Công thức tính "Uniform Patterns"

$$LBP_{P, R}^{riu2} = \begin{cases}\sum_{i=0}^{p-1}(g_i - g_c), \text{ if } U(LBP_{P, R}) \leq 2\\P+1, \text{otherwise}\end{cases}$$

Để đạt được bất biến với phép quay, một hàm bất biến xoay của LBP được định nghĩa bằng 

$$LBP_{(P, R)^{ri}} = min(ROR(LBP_{(P, R)^{ri}}, i)i = 0, 1, ..., P-1)$$

## Câu 02: Principal Components Analysis

Trình bày các bước thực hiện của thuật toán phân tích thành phần chính PCA? Cho ví dụ minh hoạ?

**Hướng đi**
Thuật toán PCA:

Input: $D = {x_1, x_2, ..., x_n}, x_i \in R^{D}$

Output: W

Bước 1. Xây dựng vector trung bình $\mu$ 

$$\mu = \frac{1}{n}\sum_{i=1}^{n}x_i$$

Bước 2. Xây dựng ma trận hiệp phương sai $S$ 

$$S = \frac{1}{n}\sum_{i=1}^{n}(x_i - \mu_i)(x_i - \mu_i)^T$$

Bước 3. Phân rã ma trận hiệp phương sai thành những cặp vector riêng và giá trị riêng

$$\{w_1, w_2, ..., w_D\}$$ 

và 

$$\lambda_1, \lambda_2, ..., \lambda_D$$

Bước 4. Sắp xếp các giá trị riêng theo thứ tự giảm dần tương ứng với các vectors riêng 

$$\{w_1, w_2, ..., w_D\}$$ 

và 

$$\lambda_1 \geq \lambda_2 \geq ... \geq \lambda_D$$

Bước 5. Chọn $k$ vector riêng mà tương ứng với $k$ giá trị riêng lớn nhất, với $k$ là số chiều đặc trưng mới  ($k \leq D$). Ở đây mình sẽ có một cách chọn k sao cho hợp lý với cách dựa vào threshold

$$\{w_1, w_2, ..., w_k\}$$ 

và 

$$\lambda_1 \geq \lambda_2 \geq ... \geq \lambda_k$$

Bước 6. Xây dưng ma trận hình chiếu $W$ từ $k$ vector riêng 

$$W = [w_1, w_2, ..., w_k]^T$$

Bước 7: Ta có một phép biến đổi tuyến tính (linear transformation) $R^N \rightarrow R^k$ thực hiện giảm chiều (dimensionality reduction)

$$
\begin{bmatrix}
b_1 \\
b_2 \\
... \\
b_k \\
\end{bmatrix} = 
\begin{bmatrix}
w_1^T \\
w_2^T \\
... \\
w_k^T \\
\end{bmatrix}(x- \bar{x}) = W^T(x- \bar{x})
$$

Để chọn được giá trị $k$, chúng ta có thể sử dụng tiêu chí sau:

$$\frac{\sum_{i=1}^k\lambda_i}{\sum_{i=1}^N\lambda_i} > \text{threshold}$$

Trong đó: threshold là một người mà chúng ta muốn, có thể là 0.9, 0.95

Ví dụ: Cho dữ liệu 

$$
X = \begin{bmatrix}
7 & 4& 6& 8& 8 &7& 5& 9 &7& 8 \\
4 &1 &3& 6& 5& 2& 3& 5& 4 &2\\
3 &8 &5 &1&7& 9& 3& 8& 5& 2
\end{bmatrix}
$$

Bước 01: Tính toán vector trung bình
$$\text{mean\_vector} = [6.9, 3.5, 5.1]$$

Bước 02: Xây dựng ma trận hiệp phương sai
$$
C = \begin{bmatrix}
2.32 & 1.61 & -0.43\\
1.61 & 2.5 & -1.278\\
-0.43 & -1.278 & 7.878\\
\end{bmatrix}
$$

Bước 03: Phân rã ma trận hiệp phương sai thành những cặp vector riêng và giá trị riêng
$$w_1 = [-0.7012, 0.7075, 0.0841], \lambda_1 = 0.7499$$
$$w_2 = [0.699, 0.6609, 0.2731], \lambda_2 = 3.6761$$
$$w_3 = [-0.1376, -0.2505, 0.9583], \lambda_3 = 8.2739$$

Bước 04: Sắp xếp các giá trị riêng theo thứ tự giảm dần tương ứng với các vectors riêng 
$$w_3 = [-0.1376, -0.2505, 0.9583], \lambda_3 = 8.2739$$
$$w_2 = [0.699, 0.6609, 0.2731], \lambda_2 = 3.6761$$
$$w_1 = [-0.7012, 0.7075, 0.0841], \lambda_1 = 0.7499$$

Bước 05: Chọn $k=2$ vector riêng mà tương ứng với $k$ giá trị riêng lớn nhất, với $k$ là số chiều đặc trưng mới  ($k \leq D$). Ở đây mình sẽ có một cách chọn k sao cho hợp lý với cách dựa vào threshold
$$w_3 = [-0.1376, -0.2505, 0.9583], \lambda_3 = 8.2739$$
$$w_2 = [0.699, 0.6609, 0.2731], \lambda_2 = 3.6761$$

Bước 06: Xây dưng ma trận hình chiếu $W$ từ $k=2$ vector riêng 

$$W = 
\begin{bmatrix}
-0.1376 & 0.699\\
-0.2505 & 0.6609\\
0.9583 & 0.2731\\
\end{bmatrix}
$$

Bước 7: Giảm chiều dữ liệu

$$
\begin{bmatrix}
b_1 \\
b_2 \\
... \\
b_k \\
\end{bmatrix} = 
\begin{bmatrix}
w_1^T \\
w_2^T \\
... \\
w_k^T \\
\end{bmatrix}(x- \bar{x}) = W^T(X- \bar{X})
$$

$$
\begin{bmatrix}
-0.1376 & -0.2505 & 0.9583\\
0.699 & 0.6609 & 0.2731
\end{bmatrix} 
\begin{bmatrix}
0.1 & -2.9 & -0.9 & 1.1 & 1.1 & 0.1 & -1.9 & 2.1 & 0.1 & 1.1\\
0.5 & -2.5 & -0.5 & 2.5 & 1.5 & -1.5 & -0.5 & 1.5 & 0.5 & -1.5\\
-2.1 & 2.9 & -0.1 & -4.1 & -1.9 & -3.9 & -2.1 & 2.9 & -0.1 & -3.1\\
\end{bmatrix} =
\begin{bmatrix}
-2.15 & 3.80 & 0.15 & -4.7 & 1.29 & 4.09 & -1.63 & 2.11 & -0.23 & -2.75\\
-0.17 & -2.89 & -0.999 & 1.30 & 2.28 & 0.14 & -2.23 & 3.25 & 0.37 & -1.07\\
\end{bmatrix}
$$

Tính $\hat{X}$

$$\hat{X} = W^T.Y + \bar{X}
=\begin{bmatrix}
7.075 & 4.3582 & 6.1891 & 8.4573 & 8.3152 & 6.4364 & 5.5633 & 8.8818 & 7.1931 & 6.5306\\
3.9244 & 0.6388 & 2.8094 & 5.5389 & 4.6822 & 2.5682 & 2.4320 & 5.1129 & 3.8054 & 3.4814 \\
2.9910 & 7.9570 & 4.9773& 0.9451& 6.9622& 9.6076& 2.9324& 8.0142 &4.9768 &7.1762\\
\end{bmatrix}
$$

$$MSE = \frac{1}{10}\sum_{i=1}^{10}(X_i - \hat{X}_i)^2 = 0.67493$$
## Câu 03: Support Vector Machines

Trình bày các bước thực hiện của thuật toán phân lớp dùng vector hỗ trợ SVM? Phân tích cụ thể từng bước của thuật toán?

**Hướng đi**

Phát biểu bài toán:

Đầu vào: Tập huấn luyện $$(\mathbf{x}_i, y_i)_{i=1, ..., N} \in \mathbb{R}^D \times \{-1, 1\}$$

Đầu ra: $(\mathbf{w}, b)$

Tác vụ: Tìm một siêu phẳng (hyperplane) 

$$\mathbf{w}^T\mathbf{x}+b = 0, (\mathbf{w} \in \mathbb{R}^D, b \in \mathbb{R})$$

phân tách hai lớp

# Đề 02

## Câu 01: Local Binary Pattern

a) Trình bày các kiến thức về đặc trưng mẫu nhị phân cục bộ $LBP_{(P, R)}$
- (1) Phương pháp xác định giá trị
- (2) Ưu điểm
- (3) Nhược điểm
- (4) Các vị dụ minh hoạ cho các kiến thức đẵ trình bày

b) Sự khác biệt giữa đặc trưng  $LBP_{(P, R)}$ so với đặc trưng $LBP_{(P, R)^{ri}}$. Trình bày ví dụ cho cách tính đặc trưng $LBP_{(P, R)^{ri}}$

**Hướng đi**
a) Trình bày các kiến thức về đặc trưng mẫu nhị phân cục bộ $LBP_{(P, R)}$

**(1) Phương pháp xác định giá trị**

Phương pháp xác định giá trị LBP sơ khai: Được tính toán bằng cách tại mỗi điểm ảnh, xét 8 điểm xung quanh nó
- Bước 01: Các điểm xung quanh có giá trị nhỏ hơn điểm đang xét sẽ được đánh dấu là 0, lớn hơn điểm đang xét sẽ được đánh dấu là 1
- Bước 02: Sau đó, các giá trị sau khi tính toán phân ngưỡng ở trên sẽ được nhân với ma trận trọng số và được sử dụng để tính giá trị LBP của điểm đang xét

$$\begin{bmatrix}1 & 2 & 4 \\128 & & 8\\64 & 32 & 16\end{bmatrix}$$

![](/assets/images_posts/orginal_lbp.png)

Phương pháp xác định giá trị LBP cải tiến: Xét các điểm thuộc đường tròn với tâm là điểm đang xét. Ta gọi $(P,R)$ là vùng lân cận gồm $P$ điểm trên một đường tròn bán kinh $R$

$$T = t(g_c, g_0, ..., g_{p-1})$$

Trong đó:
- $g_c$ và $(g_0, ..., g_{p-1})$ là giá trị trên ảnh xám của điểm trung tâm và các điểm trên đường tròn bán kính $R$

Ta có thể xấp xỉ công thức trên bằng cách lấy từng điểm trên đường tròn bán kinh $R$ trừ đi giá trị trung tâm $g_c$ do các giá trị chỉ thể hiện cường độ sáng của điểm ảnh.

$$T \approx t(g_0 - g_c, g_1 - g_c, ..., g_{p-1} - g_c)$$

Để không bị ảnh hưởng bởi các giá trị của các điểm ảnh, chuẩn hoá công thức trên bằng việc xét dấu

$$T \approx t(sign(g_0 - g_c), sign(g_1 - g_c), ..., sign(g_{p-1} - g_c))$$

Trong đó:

$$sign(x) = \begin{cases}1, x \geq 0\\ 0, x < 0\end{cases}$$

Giá trị LBP được tính như sau:

$$LBP_{P, R} = \sum_{i=0}^{p-1} sign(g_i-g_c) \times 2^i$$

**(2) Ưu điểm**
- Bất biến đối với bất kỳ phép biến đổi đơn điệu nào trên ảnh xám (grayscale) nên hiệu quả cho việc làm giảm tỷ lệ từ chối sai (FRR) cho mỗi đối tượng, khi đối tuợng đó được chụp tại những điều kiện sáng/ tối khác nhau
- Lượng tử hoá vector là không cần thiết
- Tính toán một cách đơn giản

![](/assets/images_posts/lbp_pr.png)

**(3) Nhược điểm**
- Chưa quan tâm đến phép quay, đặc trưng LBP áp dụng trên hình ảnh xoay của cùng một đối tượng sẽ tạo ra các giá trị khác nhau

![](/assets/images_posts/stuck_lbp_pr.png)

- Có vấn đề khi số điểm lân cận tăng quá cao, giá trị LBP sẽ tăng lên rất cao

b) Sự khác biệt giữa đặc trưng $LBP_{(P, R)}$ so với đặc trưng $LBP_{(P, R)^{ri}}$. Trình bày ví dụ cho cách tính đặc trưng $LBP_{(P, R)^{ri}}$

Đặc trưng $LBP_{(P, R)^{ri}}$ khắc phục nhược điểm của đặc trưng $LBP_{(P, R)}$: bất biến với phép xoay, chọ giá trị nhỏ nhất để đại diện cho tất cả các ảnh xoay (mỗi ảnh được xoay theo hướng của vòng tròn luợng giác một góc xác định cho trước)

Để kiểm tra một mẫu kết cấu có phải là "Uniform Patterns" hay không, ta áp dụng công thức

$$U(LBP_{P,R}) = |sign(g_{p-1} - g_c) - sign(g_0 - g_c)| + \sum_{i=1}^{p-1}|sign(g_i - g_c) - sign(g_{i-1} - g_c)|$$

Khi $U \leq 2$, mẫu kết cấu gọi là "Uniform Patterns", ngược lại là "Non-Uniform Patterns"

Công thức tính "Uniform Patterns"

$$LBP_{P, R}^{riu2} = \begin{cases}\sum_{i=0}^{p-1}(g_i - g_c), \text{ if } U(LBP_{P, R}) \leq 2\\P+1, \text{otherwise}\end{cases}$$

Để đạt được bất biến với phép quay, một hàm bất biến xoay của LBP được định nghĩa bằng 

$$LBP_{(P, R)^{ri}} = min(ROR(LBP_{(P, R)^{ri}}, i)i = 0, 1, ..., P-1)$$

## Câu 02: Linear Discriminant Analysis

Trình bày các bước chính của thuật toán tách lớp tuyến tính LDA? Điểm khác biệt giữa PCA và LDA? Cho ví dụ minh hoạ?

**Hướng đi**

Thuật toán LDA:

Input: Data labeled $$\mathcal{D} = \{(x_1, y_1), (x_2, y_2), ..., (x_n, y_n)\}, x_i \in \mathcal{R}^D, y_i \in \{c_1, ..., c_k\}$$

Output: ma trận chiếu $W$

Giả sử có $k$ lớp

Dặt $\mu_i$ là vector trung bình của mỗi lớp $i$, với $i = 1, 2, ..., k$

Đặt $N_i$ là số lượng mẫu trong mỗi lớp thứ  $i$, với $i = 1, 2, ..., k$

Đặt $N = \sum_{i=1}^kN_i$ là tổng số lượng mẫu

Bước 01: Với mỗi lớp, tính toán vector trung bình $D$

Bước 02: Xây dựng ma trận phân tán between-class $S_B$ và ma trận phân tán within-class $S_W$

Within-class scatter matrix

$$S_W = \sum_{i=1}^k\sum_{j=1}^{N_i}(x_j - \mu_i)(x_j - \mu_j)^T$$

Between-class scatter matrix

$$S_B = \sum_{i=1}^k(\mu_i - \mu)(\mu_i - \mu)^T$$

$$\mu = \frac{1}{k}\sum_{i=1}^k\mu_i$$

Bước 03: Tính toán vector riêng và giá trị tương ứng của ma trận $S_W^{-1}S_B$

$$S_bw_k = \lambda_kS_Ww_k$$

Bước 04: Sắp xếp những giá trị riêng tương ứng với những vector riêng theo chiều giảm dần 

Bước 05: Chọn $n$ vector riêng tương ứng với $n$ giá trị riêng lớn nhất để xây dựng ma trận biến đổi $D \times D$ chiều $W$, những vector riêng là những cột của ma trận này

$$
\begin{bmatrix}
b_1 \\
b_2 \\
... \\
b_k \\
\end{bmatrix} = 
\begin{bmatrix}
w_1^T \\
w_2^T \\
... \\
w_k^T \\
\end{bmatrix}(x- \bar{x}) = W^T(x- \bar{x})
$$

Bởi vì $S_B$ có hạng lớn nhất là $k-1$, số vector riêng lớn nhất khác 0 là $k-1$

Trường hợp $S_W^{-1}$ không tồn tại
- Nếu $S_W$ không là ma trận đơn (non-singular matrix)
$$S_W^{-1}S_Bw_k = \lambda_kw_k$$
 Trường hợp $S_W^{-1}$ luôn luôn không tồn tại
 - Dùng PCA trước

$$
\begin{bmatrix}
x_1 \\
x_2 \\
... \\
x_N \\
\end{bmatrix} \longrightarrow PCA \longrightarrow \begin{bmatrix}
y_1 \\
y_2 \\
... \\
y_K \\
\end{bmatrix}
$$

- Sau đó áp dụng LDA

$$
\begin{bmatrix}
x_1 \\
x_2 \\
... \\
x_N \\
\end{bmatrix} \longrightarrow LDA \longrightarrow \begin{bmatrix}
y_1 \\
y_2 \\
... \\
y_{k-1} \\
\end{bmatrix}
$$

Điểm khác biệt giữa PCA và LDA

![](/assets/images_posts/pca_vs_lda.png)

|  PCA 	|  LDA 	|
|---	|---	|
|  Kỹ thuật biến đổi tuyến tính, một kỹ thuật giảm chiều dữ liệu không giám sát 	|  Kỹ thuật biến đổi tuyến tính, một kỹ thuật giảm chiều dữ liệu có giám sát 	|
|  Không quan tâm về nhãn dữ liệu | Quan tâm về nhãn dữ liệu, dữ liệu đầu vào phải được gán nhãn trước khi chạy thuật toán LDA 	|
|  PCA tập trung cực đại phương sai của dữ liệu 	|  LDA tập trung cực tiểu khoảng cách các phần tử trong cùng một lớp (within classes distance) và cực đại khoảng cách giữa các lớp với nhau (between classes distance) 	|

## Câu 03: Neural Networks

Cho trước một mạng neural với hai tầng: 
- Tầng 1: nhận vào một ảnh nhập
- Tầng 2 là tầng xuất ứng với 4 bộ phân lớp được xuất ra (Bộ phân lớp 1 đại diện cho 'học sinh nữ có tóc dài'; Bộ phân lớp 2 đại diện cho 'học sinh nam có tóc dài'; Bộ phân lớp 3 đại diện cho 'học sinh nữ có tóc ngẵn' và Bộ phân lớp 4 đại diện cho 'học sinh nam có tóc ngắn').

Tuy nhiên, dữ liệu huấn luyện mạng phân bố không đều:
- 4 mẫu cho bộ phân lớp 1
- 4 mẫu cho bộ phân lớp 3
- 4 mẫu cho bộ phân lớp 4
- Riêng bộ phân lớp 2 chỉ có 1 mẫu huấn luyện

Như vậy, bộ phân lớp 2 là yếu. Học viên hãy đề nghị một cấu trúc mạng nơ ron mới sao cho nâng hiệu quả bộ lớp 2 mà vẫn sử dụng bộ mẫu huấn luyện nói trên? Nêu lên tính hiệu quả của cấu trúc mạng nơ ron đề nghị?

**Hướng đi**

Đề nghị một cấu trúc mạng Neural tăng thêm một tầng cơ sở
- Tầng 01: Tầng nhận ảnh đầu vào
- Tầng 02 là tầng ẩn: Trong tầng này gồm hai bộ phân lớp
  - Bộ phân lớp 1 dùng để phân loại học sinh nam hay học sinh nữ gồm 8 mẫu huấn luyện cho nữ và 8 mẫu nam
  - Bộ phân lớp 2 dùng để phân loại tóc ngắn hay tóc dài gồm 5 mẫu huấn luyện tóc dài và 8 mẫu tóc ngắn
- Tầng 03 là tập đầu ra với 4 bộ phân lớp: học sinh nữ có tóc dài, học sinh nam có tóc dài, học sinh nữ có tóc ngẵn, học sinh nam có tóc ngắn

![](/assets/images_posts/file.txt.png)

Tính hiệu quả của cấu trúc mạng Neural:
- Xây dựng thêm một tầng cơ sở với 2 bộ phân lớp, tăng cường dữ liệu cho mỗi bộ phân lớp, nên sẽ trả về kết quả tốt
- Tầng phân lớp (tầng xuất) chỉ có nhiệm vụ tổng hợp các kết quả nhận được từ tầng cơ sở (không cần phải huấn luyện)