---
layout: single
permalink: /intro2ml/topic00-intro-ml
title: "[Intro2ML] Máy Học là gì?"
toc: true
comments: True
---
Trong những năm gần đây, có lẽ mọi người không lạ gì khi nghe đến cụm từ "AI", từ viết tắt của "Artificial Intelligence" - Trí tuệ nhân tạo mà ta được biết đến. AI nổi lên mạnh mẽ thúc đẩy nhiều lĩnh vực trong xã hội đi lên, đặc biệt là trong tự động hóa, nhận dạng, dự đoán xu hướng, ... Để đạt được những thành công vượt trội, tạo nên cuộc cách mạng công nghiệp 4.0, chính là nhờ vào lĩnh vực Machine Learning hay Máy học (Học máy) và ẩn sau nó, thứ làm nên bước nhảy vọt đó chính là Học Sâu (Deep Learning).

|![From Winning Go to Making Dough: What Can Deep Learning Do for Your Business?](https://blogs.nvidia.com/wp-content/uploads/2016/10/Deep_Learning_for_Business_Defined_History.jpg)|
|:--:| 
| Nividia Blogs: From Winning Go to Making Dough: What Can Deep Learning Do for Your Business?|

## Các khái niệm

**AI - Trí tuệ Nhân tạo là gì?**

Trí tuệ nhân tạo (AI) là sự mô phỏng các quá trình thông minh của con người bằng máy móc, đặc biệt là hệ thống máy tính. Các ứng dụng cụ thể của AI bao gồm hệ thống chuyên gia, xử lý ngôn ngữ tự nhiên (NLP), nhận dạng giọng nói và thị giác máy. Điều này khác với sự thông minh tự nhiên như ở con người hay động vật, thường liên quan đến ý thức và cảm xúc.

Trong việc nghiên cứu trí tuệ nhân tạo, con người ta cố gắng trong việc mô phỏng lại trí thông minh tự nhiên ấy, để tạo ra Trí tuệ nhân tạo tổng quát (Artificial General Intelligence), dể có thể tạo ra được Artificial Biological Intelligence - Trí tuệ sinh học nhân tạo.

Tuy nhiên, theo các định nghĩa của sách giáo khoa thì AI có thể hiểu là lĩnh vực nghiên cứu về các "tác tử thông minh - intelligent agents", giúp bất kỳ thuết bị nào nhận thức được môi trường của nó và thực hiện các hành động nhằm tối đa hóa khả năng hoàn thành mục tiêu của nó.

Chúng ta có thể thấy, nghe nói hoặc đọc qua nhiều ứng dụng của Trí tuệ Nhân tạo vào cuộc sống như hệ thống xe tự hành, hệ thống nhận diện khuôn mặt, nhận dạng giọng nói, hệ thống chuyên gia gợi ý, các máy chơi cờ của Google Deepmind như AlphaGo, ...

**Machine Learning - Máy học là gì?**

Machine Learning, hay hiểu nôm na là Máy học (hoặc Học máy), là một nhánh con của AI, là một trong những cách tiếp cận Trí tuệ Nhân tạo được các nhà Khoa học máy tính, Toán học, Vật lý, ... tập trung nghiên cứu rất nhiều và trong những năm gần đây đạt được rất nhiều thành công. Nó cung cấp cho hệ thống khả năng tự động học hỏi và cải thiện từ kinh nghiệm mà không cần được lập trình rõ ràng, máy tính truy cập vào dữ liệu và sử dụng, học hỏi những tri thức từ dữ liệu mà chúng ta cung cấp cho nó. Vì thế chúng ta mới nghe cụm từ "learning from DATA", dữ liệu rất quan trọng, không có dữ liệu, bạn sẽ không gặp man mắn.

**Deep Learning - Học Sâu là gì?**

Từ khoảng năm 2012, Deep Learning trở nên phát triển mạnh mẽ hơn bao giờ hết, giúp Machine Learning tiến lên một tầm cao mới trong ước mơ đạt dược trí tuệ nhân tạo sinh học của con người. Phần lớp sự thành công này đến từ khả năng tính toán của các máy tính, siêu máy tính, dữ liệu chúng ta ngày càng lớn, được thu thập, tiền xử lý bởi rất nhiều chuyên gia. Deep Learning đã giúp chúng ta giải quyết hàng loạt những bài toán khó, tưởng chừng như không thể giải quyết nổi hoặc rất khó khăn và tốn nhiều thời gian. Tuy vậy, chúng ta cũng không thể quên những mô hình học thống kê đơn giản, nhỏ gọn như LDA, Naive Bayes, ... mà ngày càng được chú ý hơn vì xu hướng IoT trong thời gian tới (Internet of Things - Kết nối vạn vật)

## Tác tử học - Learning Agent


## Học từ "dữ liệu" - Learning from DATA

Muốn áp dụng Machine Learning thì chúng ta cần xem xét bài toán đang giải quyết có thõa mãn những tính chất sau đây hay không:
- Tồn tại một mẫu (A pattern exists)
- Không thể biểu diễn nó chính xác bằng công thức Toán học (Cannot pin it down mathematically)
- Chúng ta có DỮ LIỆU

|![](/intro2ml/img/topic0/Learning_from_Data.png)|
|:--:| 
| Image source: Yaser S. Abu-Mostafa, Learning from Data| 

## Phân loại theo hình thức học

### Học có giám sát - Supervised Learning

|![Supervised Machine Learning](https://7qkiy1yofpnz20qc4wdcb9t6-wpengine.netdna-ssl.com/wp-content/uploads/2020/07/machine-learning-infographics-2-scaled.jpg)|
|:--:| 
| Supervised Machine Learning |

Với phương pháp học có giám sát, chúng ta có tập dataset $D$ chứa (input, correct output hay label). Tùy vào đầu ra của bài toán mà nó có thể chia thành hai loại chính:
- Khi đầu ra bài toán là một tập hợp hữu hạn các giá trị, thì bài toán mà chúng ta đang giải quyết là bài toán phân loại (classification).
- Khi đầu ra bài toán là một con số, thì bài toán mà chúng ta đang giải quyết là bài toán hồi quy (regression).

### Semi-supervised Learning

|![Semi-supervised Learningg](https://www.researchgate.net/profile/Fabien-Lotte/publication/277605013/figure/fig4/AS:281234816159750@1444063013959/Principle-of-semi-supervised-learning-1-a-model-eg-CSP-LDA-classifier-is-first.png)|
|:--:| 
| Semi-supervised Learning |

Với phương pháp học Semi-Supervised Learning, bài toán của chúng ta có một phần được dán nhãn (label) và cũng tùy vào output mà chúng ta có thể xử lý nó với kỹ thuật của bài toán phân lớp hay bài toán hồi quy.

### Học không có giám sát - Unsupvised Learning

|![Unsupervised Machine Learning](https://www.diegocalvo.es/wp-content/uploads/2020/02/unsupervised-learning.png)|
|:--:| 
| Unsupervised Machine Learning |

Với phương pháp học không có giám sát, dữ liệu chúng ta không có nhãn, các thuật toán học không giám sát sẽ tự động giải quyết vấn đề này thông qua việc rút trích đặc trưng dữ liệu từ đó thực hiện một số tác vụ cần thiết như gom nhóm (clustering), giảm chiều dữ liệu (dimesional-reduction) để có thể tính toán và học từ dữ liệu.

### Học tăng cường - Reinforcement Learning

|![Reinforcement Learning](https://kitrum.com/wp-content/uploads/2020/10/EHR-2-1024x683.png)|
|:--:| 
| Reinforcement Learning |

Với phương pháp học tăng cường, chúng ta có tập dữ liệu $D$ bao gồm (observation - quan sát, action - hàng động, reward - phần thưởng), với mục dich là tự động xác định hành động dựa trên hoàn cảnh, những quan sát được để đạt được lợi ích, phần thưởng cao nhất (maximizing reward, exactly the system performance). Các bạn có thể tìm hiểu AlphaGo của Google Deepmind.

## Các bước chính để giải quyết bài toán máy học

Bước 01: Chuẩn bị dữ liệu

Bước 02: Chuẩn hóa dữ liệu

Bước 03: Phân chia tập dữ liệu
    - Tập huấn luyện (training set)
    - Tập xác nhận trong quá trình huấn luyện (Validation set)
    - Tập kiểm tra (Testing set)

Bước 04: Deploy mô hình

Bước 05: Tiếp tục thu thập dữ liệu, củng cố mô hình

Và đây chính là một machine learning pipeline

|![Machine Learning Pipeline](https://www.kindpng.com/picc/m/346-3460588_machine-learning-pipeline-diagram-hd-png-download.png)|
|:--:|
| Machine Learning Pipeline |


