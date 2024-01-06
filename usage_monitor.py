# usage_monitor.py

import psutil
import GPUtil
import time
import threading

def get_gpu_memory_usage():
    gpus = GPUtil.getGPUs()
    return [gpu.memoryUsed / 1024 for gpu in gpus]

def get_gpu_usage():
    gpus = GPUtil.getGPUs()
    return [gpu.load for gpu in gpus]

def start_monitoring(interval=1, filename='system_usage.txt'):
    with open(filename, 'w') as file:
        file.write('Time,CPU_Usage,GPU_Usage,Memory_Usage,GPU_Memory_Usage\n')

        while True:
            # 获取当前时间
            current_time = time.time()

            # 获取 CPU 和 GPU 的使用率
            cpu_usage = psutil.cpu_percent()

            # 获取内存占用大小
            memory_usage = psutil.virtual_memory().used / 1024 / 1024 / 1024

            # GPU 数量和显存使用量
            gpu_memory_usage = get_gpu_memory_usage()
            gpu_usage = get_gpu_usage()
            gpu_count = len(gpu_memory_usage)

            # 写入文件
            file.write(f"{current_time},{cpu_usage},{memory_usage},{gpu_count},{' '.join(map(str, gpu_memory_usage))}, {' '.join(map(str, gpu_usage))}\n")
            file.flush()
            time.sleep(interval)

def start_monitor_thread(interval=0.1, filename='system_usage.txt'):
    thread = threading.Thread(target=start_monitoring, args=(interval, filename))
    thread.start()
    
if __name__ == '__main__':
    start_monitoring(interval=0.1, filename="usage_4.txt")
    pass
