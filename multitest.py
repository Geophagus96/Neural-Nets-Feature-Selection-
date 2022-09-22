from multiprocessing import Process, Pool
import time
import numpy as np
import sys
def cube(a=1):
    print(a**3)

if __name__ == '__main__':
    numbers = [float(sys.argv[4]), float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])]

    t0 = time.time()
    for i in range(len(numbers)):
        cube(numbers[i])
    print(time.time()-t0)
    
    # t0 = time.time()
    # procs = []
    # for number in numbers:
    #     proc = Process(target=cube, args=(number,))
    #     procs.append(proc)
    #     proc.start()
    #     proc.join()
    # print(time.time()-t0)
    
    t0 = time.time()
    num_threads = 4
    with Pool(num_threads) as pool: 
        results = pool.map(cube, numbers)
    print(time.time()-t0)
    
    
# def summation(array):
#     print(array[0]+array[1])
    
# a = np.array([[1,2],[1,3],[2,3],[7,9]])

# if __name__ == '__main__':
#     t0 = time.time()
#     procs = []
#     for element in a:
#         proc = Process(target=summation, args=(element,))
#         procs.append(proc)
#         proc.start()
#         proc.join()
#     print(time.time()-t0)


# def square(a):
#     if (a%10000==1):
#         print(a)

# a = np.arange(100000)

# if __name__ == '__main__':
#     t0 = time.time()
#     procs = []
#     for element in a:
#         proc = Process(target=square, args=(element,))
#         procs.append(proc)
#         proc.start()
#         proc.join()
#     print(time.time()-t0)
    
#     t0 = time.time()
#     for element in a:
#         square(element)
#     print(time.time()-t0)