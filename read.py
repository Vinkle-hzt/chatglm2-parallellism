import matplotlib.pyplot as plt

def main():
    file = 'usage_4.txt' # change it
    cnt = 0
    with open(file) as f:
        lines = f.readlines()[300:]
        x = []
        mem = []
        cpu = []
        gpus_usage = [[],[],[],[]]
        gpus_mem = [[],[],[],[]]
        for line in lines:
            data = line.split(',')
            x.append(float(data[0]))
            cpu.append(float(data[1]))
            mem.append(float(data[2]))
            for i in range(4):
                gpus_mem[i].append(float(data[4].strip().split(' ')[i]))
                gpus_usage[i].append(float(data[5].strip().split(' ')[i]))
            cnt += 1
            if cnt > 200:
                break
        plt.figure(figsize=(12, 8))
        plt.subplot(4, 1, 1)
        plt.plot(x, cpu, 'r-', label='cpu')
        plt.legend()
        
        plt.subplot(4, 1, 2)
        plt.plot(x, mem, 'r-', label='mem')
        plt.legend()
        
        plt.subplot(4, 1, 3)
        plt.plot(x, gpus_mem[0], 'r-', label='gpu_mem_0')
        plt.plot(x, gpus_mem[1], 'b-', label='gpu_mem_1')
        plt.plot(x, gpus_mem[2], 'g-', label='gpu_mem_2')
        plt.plot(x, gpus_mem[3], 'y-', label='gpu_mem_3')
        plt.legend()
        
        plt.subplot(4, 1, 4)
        plt.plot(x, gpus_usage[0], 'r-', label='gpu_usage_0')
        plt.plot(x, gpus_usage[1], 'b-', label='gpu_usage_1')
        plt.plot(x, gpus_usage[2], 'g-', label='gpu_usage_2')
        plt.plot(x, gpus_usage[3], 'y-', label='gpu_usage_3')
        plt.legend()
        plt.show()           
    pass

if __name__ == '__main__':
    main()