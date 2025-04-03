# Các bước cải tiến
1. Tối ưu tốc độ xử lý PDF
    - Thêm xử lý đa luồng cho file PDF lớn

    - Cache kết quả OCR để tái sử dụng

    - Giảm chunk size nếu gặp lỗi token limit

2. Xử lý PDF phức tạp

    - File scan chất lượng thấp

    - Bảng biểu/công thức toán học

    - PDF nhiều layer/text ẩn

3. Nâng cao chức năng

    - Tích hợp web crawler để tự động thu thập dữ liệu

    - Thêm metadata filtering khi query

    - Triển khai versioning cho vector database
