import torch
import os
import psutil


def test_chunks(chunks, num_samples=3, full_chunk_index=None, verbose=False):
    """
    Kiểm tra và in thông tin về chunks

    Args:
        chunks: Danh sách các chunks cần kiểm tra
        num_samples: Số lượng chunks mẫu để in ra
        full_chunk_index: Chỉ số của chunk muốn in toàn bộ nội dung
        verbose: Nếu True, in thêm thông tin chi tiết về cấu trúc của chunk
    """
    print(f"Tổng số chunks: {len(chunks)}")

    # Kiểm tra cấu trúc của chunk (chỉ khi verbose=True)
    if verbose and len(chunks) > 0:
        first_chunk = chunks[0]
        print("\n=== THÔNG TIN CẤU TRÚC CHUNK ===")
        print(f"Kiểu dữ liệu: {type(first_chunk)}")
        print(f"Các thuộc tính sẵn có: {dir(first_chunk)[:5]}...")
        print(f"Các khóa trong metadata: {list(first_chunk.metadata.keys())}")

    # In mẫu một số chunks đầu tiên
    print("\n=== MẪU CHUNKS ===")
    for i in range(min(num_samples, len(chunks))):
        print(f"\n--- Chunk {i+1} ---")
        print(f"Metadata: {chunks[i].metadata}")
        print(f"Nội dung: {chunks[i].page_content[:200]}...")  # In 200 ký tự đầu tiên

    # In toàn bộ nội dung của một chunk cụ thể nếu được chỉ định
    if full_chunk_index is not None and 0 <= full_chunk_index < len(chunks):
        print(f"\n=== NỘI DUNG ĐẦY ĐỦ CỦA CHUNK {full_chunk_index+1} ===")
        print(chunks[full_chunk_index].page_content)


def inspect_chunk_structure(chunks):
    """
    Kiểm tra và hiển thị cấu trúc chi tiết của một chunk

    Args:
        chunks: Danh sách các chunks cần kiểm tra
    """
    # Kiểm tra xem danh sách chunks có phần tử không
    if len(chunks) > 0:
        # Lấy chunk đầu tiên để kiểm tra
        first_chunk = chunks[0]

        # 1. In ra toàn bộ thông tin về chunk
        print("1. Toàn bộ thông tin về chunk:")
        print(first_chunk)

        # 2. In ra kiểu dữ liệu (type) của chunk
        print("\n2. Kiểu dữ liệu của chunk:")
        print(type(first_chunk))

        # 3. In ra các thuộc tính và phương thức của chunk
        print("\n3. Các thuộc tính và phương thức của chunk:")
        print(dir(first_chunk))

        # 4. In ra nội dung chính của chunk
        print("\n4. Nội dung của chunk:")
        print(first_chunk.page_content)

        # 5. In ra metadata của chunk
        print("\n5. Metadata của chunk:")
        print(first_chunk.metadata)

        # 6. In ra các khóa trong metadata
        print("\n6. Các khóa trong metadata:")
        if hasattr(first_chunk, "metadata") and isinstance(first_chunk.metadata, dict):
            print(list(first_chunk.metadata.keys()))
    else:
        print("Không có chunk nào được tạo ra")


def get_system_info():
    """
    Get System Configuration
    Return:
        dict:[cpu_cores, gpu_vram_gb]
    """
    try:
        cpu_cores = psutil.cpu_count(logical=False)  # Physical cores
        if cpu_cores is None:
            cpu_cores = psutil.cpu_count(logical=True)  # fallback to logical cores
    except:
        cpu_cores = os.cpu_count()

    # check gpu and vram
    gpu_available = torch.cuda.is_available()
    gpu_vram = 0
    if gpu_available:
        try:
            gpu_index = 0
            gpu_vram = torch.cuda.get_device_properties(gpu_index).total_memory / (
                1024**3
            )  # convert to GB
        except:
            gpu_available = False
    return {
        "cpu_cores": cpu_cores,
        "gpu_available": gpu_available,
        "gpu_vram_gb": gpu_vram,
    }


def determine_optimal_configuration():
    """
    Xác định cấu hình tối ưu cho embedding model dựa trên tài nguyên hệ thống.

    Returns:
        dict: Một dictionary chứa các thông số cấu hình (device, batch_size, num_workers, dtype)
    """
    # Lấy thông tin về hệ thống
    system_info = get_system_info()
    cpu_cores = system_info["cpu_cores"]
    gpu_available = system_info["gpu_available"]
    gpu_vram_gb = system_info["gpu_vram_gb"]

    print(
        f"System detected: {cpu_cores} CPU cores, GPU {'available' if gpu_available else 'not available'}"
    )
    if gpu_available:
        print(f"GPU VRAM: {gpu_vram_gb:.2f} GB")

    # Xác định thiết bị để chạy mô hình
    if gpu_available and gpu_vram_gb > 2:
        device = "cuda"
        dtype = torch.float16
        print("Using GPU for embeddings")
    else:
        device = "cpu"
        dtype = torch.float32
        print("Using CPU for embeddings")

    # Tự động cấu hình batch_size và num_workers
    if device == "cuda":
        if gpu_vram_gb > 8:
            batch_size = 32
        elif gpu_vram_gb > 4:
            batch_size = 16
        elif gpu_vram_gb > 2:
            batch_size = 8
        else:
            batch_size = 4
    else:
        batch_size = min(32, max(8, cpu_cores * 2))

    if device == "cpu":
        num_workers = min(4, max(1, cpu_cores // 2))
    else:
        num_workers = 0

    print(f"Configuration: batch_size={batch_size}, num_workers={num_workers}")

    return {
        "device": device,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "dtype": dtype,
    }
