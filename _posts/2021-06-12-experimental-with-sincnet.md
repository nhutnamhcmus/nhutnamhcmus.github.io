---
title: "Thực nghiệm với SincNet"
categories:
  - Speaker Recognition
  - Speech Processing
  - Convolutional Neural Networks
  - Raw waveform
tags:
  - Machine Learning
  - Speaker Recognition
  - Speaker Verfication
  - Speaker Identification
toc: true
comment: true
---

Trong những năm gần đây, Machine Learning (Học máy/ Máy học) đạt được rất nhiều thành tựu nổi bật, thúc đẩy Cách mạng Công nghệ 4.0 phát triển nhanh chóng. Một trong những tác nhân chính mạnh mẽ nhất đến từ Deep Learning (Deep Neural Networks - Các mạng Học sâu), các mô hình học dựa trên những phương pháp này đã rất thành công, đạt được một hiệu năng đầy triển vọng, trong nhiều tác vụ khác nhau như Thị giác Máy tính (Computer Vision), Nhận dạng Nhân trắc học (Biometrics): Giọng nói (Speech), Khuôn mặt (Face), Vân tay (Fingerprint), ..., Xử lý ngôn ngữ tự nhiên (Natural Language Processing), ...
	
Trong tác vụ nhận dạng, đặc biệt là nhận dạng giọng nói (Voice/ Speech Recognition) gần như là một bài toán khó trong nhiều năm, cần những phương pháp rất phức tạp để có thể giải quyết được. Nhờ vào Học sâu, phương giải phải khả thi, mà cụ thể là với Mạng Neural Tích chập (Convolutional Neural Networks - CNNs) đem lại những kết quả đầy hứa hẹn khi chỉ cần đầu vào trực tiếp là mẫu giọng nói thô. Thay vì sử dụng các tính năng thủ công tiêu chuẩn, CNNs sau này học cách biểu diễn giọng nói cấp thấp từ các dạng sóng, có khả năng cho phép mạng nắm bắt tốt hơn các đặc điểm giọng nói ở dải tần hẹp quan trọng như cao độ và hình dạng

# Chuẩn bị dữ liệu

## Tiếng Anh

Với TIMIT, ta có một kho ngữ liệu với 462 người nói, các khoảng không phải lời nói ở đầu và cuối mỗi câu đã bị xóa, những tập tin về nội dung câu nói của TIMIT cũng được loại bỏ. Sau khi tinh chỉnh toàn bộ dữ liệu, tác giả dùng 5 câu nói của mỗi người nói để huấn luyện, 3 câu nói của mỗi người nói dùng để kiểm tra.

Với tập ngữ liệu LibriSpeech, những phần với độ im lặng bên trong kéo dài hơn 125 ms được chia thành nhiều phần nhỏ. Việc chia tập huấn luyện (training set), tập kiểm tra (testing set) là ngẫu nhiên bằng cách chọn 12-15 giây dữ liệu huấn luyện của mỗi người nói và các câu kiểm tra kéo dài từ 2-6 giây. 

## Tiếng Việt

Sử dụng tập dữ liệu Son et al. Dataset. Nguồn dữ liệu từ bài báo Vietnamese Speaker Authentication Using Deep Models
- Dung lượng của tập dữ liệu: 535 MB
- Số mẫu trong tập dữ liệu: 400 mẫu
- Bộ dữ liệu gồm: hai tập  Men và Women, mỗi tập con chứa 10 thư mục người nói. Mỗi thư mục người nói chứa 20 đoạn ghi âm, chia ra Long và Short (mỗi loại 10 đoạn) 

# Xây dựng thực nghiệm

## Phân chia tập train, test

Bộ dữ liệu train (huấn luyện) và test (kiểm tra) được giữ nguyên theo cài đặt của nhóm tác giả

Với TIMIT
- Tập train: 2310 file âm thanh
- Tập test: gồm 1386 file âm thanh

Với Librispeech
- Tập train: 14481 file âm thanh
- Tập test: gồm 7452 file âm thanh

Với Son et al. Dataset

Do bị lỗi file âm thanh nên mình bỏ 1 số file bị lỗi từ tập dữ liệu và bổ
sung thêm 20 (Long & Short) file do mình tự ghi âm sẵn bằng
microphone. 

Tập dữ liệu sao khi kiểm tra gồm có 19 người nói, mỗi người thu hai loại
file, 10 file âm ngắn, 10 file âm dài.

Mình thực hiện chạy thử và đánh giá mô hình dựa vào độ lỗi phân lớp
FER trong tác vụ định danh người nói. Giả định những có hai loại người
vị trí số chẵn và lẻ để thực hiện đánh giá xác minh người nói.

- Tập train: mỗi người chọn ra 5 file đầu tiên làm file huấn luyện
- Tập test: mỗi người chọn ra 2 file tiếp theo làm file kiểm tra
- Tập validation: phần còn lại, dùng để tính toán d-vector

## Thông số mô hình với tập dữ liệu

### TIMIT

#### Thông số mô hình
- Các cửa sổ có $\text{fs} = 16000$, tín hiệu được cắt thành những chunks với $\text{cw\_len}=200$, overlap $\text{cw\_shift}=10$s
- Lớp Input: sử dụng 80 bộ lọc SincNet có kích thước $L=251$, max pool - 3, sử dụng Layer Norm cho cả input và output, không dùng Batch Norm, hàm kích hoạt activation leaky-ReLU, dropout = 0
- Hai lớp CNN: sử dụng 2 lớp CNN, với mỗi lớp dùng 60 bộ lọc có kích thuốc $L=5$, sử dụng Layer Norm cho cả input và output, không dùng Batch Norm, hàm kích hoạt activation leaky-ReLU, dropout = 0
- Ba lớp DNN: sử dụng 3 lớp DNN (Multi Layer Perceptron) fully-connected với 2048 neurons, Layer Norm cho input, Batch Norm cho output, các lớp ẩn (hidden layers) dùng leaky-ReLU
- Lớp Output: Multi Layer Perceptron, 462 nodes, không dropout, không LayerNorm, không BatchNorm cho cả input và output, hàm activation function dùng softmax
-  Hàm mất mát: Negative Log Likelihood Loss

#### Các siêu tham số
- Tốc độ học - learning rate $\text{lr} = 0.001$
- $\alpha = 0.95$
- $\epsilon = 10^{-7}$
- Kích thước mỗi batch $\text{batch\_size}=128$
- Số lượng epochs training $\text{N\_epochs}=100$
- Số lượng batch $\text{N\_batches}=800$
- Số đánh giá mỗi epoch $\text{N\_eval\_epoch}=8$
- $\text{seed}=1234$

### Librispeech

#### Thông số mô hình

- Các cửa sổ có $\text{fs} = 8000$, tín hiệu được cắt thành những chunks với $\text{cw\_len}=375$, overlap $\text{cw\_shift}=10$s
- Lớp Input: sử dụng 80 bộ lọc SincNet có kích thước $L=251$, max pool - 3, sử dụng Layer Norm cho cả input và output, không dùng Batch Norm, hàm kích hoạt activation leaky-ReLU, dropout = 0
- Hai lớp CNN: sử dụng 2 lớp CNN, với mỗi lớp dùng 60 bộ lọc có kích thuốc $L=5$, sử dụng Layer Norm cho cả input và output, không dùng Batch Norm, hàm kích hoạt activation leaky-ReLU, dropout = 0
- Hai lớp DNN: sử dụng 2 lớp DNN (Multi Layer Perceptron) fully-connected với 2048 neurons, Layer Norm cho input, Batch Norm cho output, các lớp ẩn (hidden layers) dùng leaky-ReLU làm activation cho lớp DNN thứ nhất, lớp kia dùng linear
- Lớp Output: 2 lớp Multi Layer Perceptron, 2048 nodes cho mỗi lớp, không dropout, Layer Norm cho input, Batch Norm cho output, hàm activation function lớp thứ nhất dùng leaky-ReLU, lớp thứ hai dùng softmax
- Hàm mất mát: Negative Log Likelihood Loss

#### Các siêu tham số
- Tốc độ học - learning rate $\text{lr} = 0.001$
- $\alpha = 0.95$
- $\epsilon = 10^{-7}$
- Kích thước mỗi batch $\text{batch\_size}=128$
- Số lượng epochs training $\text{N\_epochs}=100$
- Số lượng batch $\text{N\_batches}=100$
- Số đánh giá mỗi epoch $\text{N\_eval\_epoch}=10$
- $\text{reg\_factor}=1000$
- $\text{fact\_amp=0.2}$
- $\text{seed}=1234$

### Son et al.

#### Thông số mô hình
- Các cửa sổ có $\text{fs} = 16000$, tín hiệu được cắt thành những chunks với $\text{cw\_len}=200$, overlap $\text{cw\_shift}=10$s
- Lớp Input: sử dụng 80 bộ lọc SincNet có kích thước $L=251$, max pool - 3, sử dụng Layer Norm cho cả input và output, không dùng Batch Norm, hàm kích hoạt activation leaky-ReLU, dropout = 0
- Hai lớp CNN: sử dụng 2 lớp CNN, với mỗi lớp dùng 60 bộ lọc có kích thuốc $L=5$, sử dụng Layer Norm cho cả input và output, không dùng Batch Norm, hàm kích hoạt activation leaky-ReLU, dropout = 0
- Ba lớp DNN: sử dụng 3 lớp DNN (Multi Layer Perceptron) fully-connected với 2048 neurons, Layer Norm cho input, Batch Norm cho output, các lớp ẩn (hidden layers) dùng leaky-ReLU
- Lớp Output: Multi Layer Perceptron, 18 nodes, không dropout, không LayerNorm, không BatchNorm cho cả input và output, hàm activation function dùng softmax
- Hàm mất mát: Negative Log Likelihood Loss

#### Các siêu tham số
- Tốc độ học - learning rate $\text{lr} = 0.001$
- $\alpha = 0.95$
- $\epsilon = 10^{-7}$
- Kích thước mỗi batch $\text{batch\_size}=128$
- Số lượng epochs training $\text{N\_epochs}=300$
- Số lượng batch $\text{N\_batches}=100$
- Số đánh giá mỗi epoch $\text{N\_eval\_epoch}=1$
- $\text{seed}=1234$ 

# Đánh giá mô hình
- FER (Frame Error Rate): độ đo tính toán sự sai khác giữa kết quả dự đoán (predicted value) của mô hình so với kết quả thật sự (actual value) của một đoạn tín hiệu.
-  CER (Classification Error Rate): thể hiện độ lỗi phân loại của mô hình. CER càng thấp chứng tỏ khả năng phân loại của mô hình càng cao.
- EER (Equal Error Rate): thể hiện khả năng xác minh danh tính của một mô hình nhận dạng (giọng nói). EER càng thấp chứng tỏ khả năng xác minh danh tính càng cao.

# Các kết quả đạt được

## Tiếng Anh

### TIMIT

|![](/assets/images_posts/intro2sincnet/evaluate_result_timit.png)|
|:--:| 
| Kết quả thực nghiệm trên TIMIT Database |

- Độ lỗi huấn luyện trung bình mỗi frame $loss\_tr=4.217127$
- Giá trị phân lớp sai mức frame $err\_te=0.513561$
- Giá trị phân lớp sai mức câu $err\_te\_snt =0.018038$

|![](/assets/images_posts/intro2sincnet/sincnet_timit_plot.png)|
|:--:| 
| Kết quả thực nghiệm trên TIMIT Database |

|![](/assets/images_posts/intro2sincnet/roc_curve_timit.png)|
|:--:| 
| Kết quả thực nghiệm trên TIMIT Database |

|![](/assets/images_posts/intro2sincnet/pr_curve_timit.png)|
|:--:| 
| Kết quả thực nghiệm trên TIMIT Database |

## Librispeech

|![](/assets/images_posts/intro2sincnet/evaluate_result_libris.png)|
|:--:| 
| Kết quả thực nghiệm trên Librispeech Database |

- Độ lỗi huấn luyện trung bình mỗi frame $loss\_tr=5.448840$
- Giá trị phân lớp sai mức frame $err\_te=0.907977$
- Giá trị phân lớp sai mức câu $err\_te\_snt =0.456924$

|![](/assets/images_posts/intro2sincnet/sincnet_librispeech_plot.png)|
|:--:| 
| Kết quả thực nghiệm trên Librispeech Database |

|![](/assets/images_posts/intro2sincnet/roc_curve_librispeech.png)|
|:--:| 
| Kết quả thực nghiệm trên Librispeech Database |

|![](/assets/images_posts/intro2sincnet/pr_curve_librispeech.png)|
|:--:| 
| Kết quả thực nghiệm trên Librispeech Database |

## Tiếng Việt

|![](/assets/images_posts/intro2sincnet/evaluate_result_vn_speaker.png)|
|:--:| 
| Kết quả thực nghiệm trên Son et al. Database |

- Độ lỗi huấn luyện trung bình mỗi frame$loss\_tr=0.113859$
- Giá trị phân lớp sai mức frame $err\_te=0.031011$
- Giá trị phân lớp sai mức câu$err\_te\_snt =0.000000$

|![](/assets/images_posts/intro2sincnet/sincnet_vietnamese_plot.png)|
|:--:| 
| Kết quả thực nghiệm trên Son et al. Database |

|![](/assets/images_posts/intro2sincnet/roc_curve_vietnamese.png)|
|:--:| 
| Kết quả thực nghiệm trên Son et al. Database |

|![](/assets/images_posts/intro2sincnet/pr_curve_vietnamese.png)|
|:--:| 
| Kết quả thực nghiệm trên Son et al. Database |

# Kết luận
- Sinh trắc học giọng nói là một lĩnh lớn có nhiều ứng dụng, với các phương pháp truyền thống giải quyết tương đối bài toán này
- Các phương pháp hiện đại sử dụng các phương pháp rút trích đặc trưng dựa vào mạng Học Sâu d-vectors đem lại nhiều kết quả khả quan, đầy mong đợi
- Đặc biệt, với bài toán phân lớp giọng nói việc kết hợp đa nhiệm trở nên một hướng giải quyết tốt cho bài toán này. Ngoài ra, SincNet đem lại một làn gió mới, khi tận dụng và tối ưu lại CNN truyền thống bằng các bộ lọc hình Sinc, giúp tác vụ nhận dạng người nói có nhiều triễn vọng khi kết hợp với d-vectors, DNN-class.

# Tài liệu tham khảo

- \[1\] Mirco Ravanelli. A brief introduction to sincnet, 2018. 
- \[2\] Mirco Ravanelli and Yoshua Bengio. Speaker recognition from raw waveform with sincnet, 2019.
- \[3\] Dávid Sztahó, György Szaszák, and András Beke. Deep learning methods in speaker recognition: a review, 2019.