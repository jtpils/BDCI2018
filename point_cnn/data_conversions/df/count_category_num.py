import os
import numpy as np
from multiprocessing import Process


def count_cate_num(dir_input, framenames):
    cate_nums = np.zeros(8, int)
    for framename in framenames:
        print(framename)
        path_cate = os.path.join(dir_input, 'category', framename)
        categories = np.loadtxt(path_cate).astype(np.int32)
        for cate in categories:
            cate_nums[cate] += 1
        print(cate_nums)
    return cate_nums


if __name__ == '__main__':
    num_prccess = 6
    dir_input = "/home/leon/Disk/dataset/Downloads/DataFountain/dataset/training"

    framenames = sorted(os.listdir(os.path.join(dir_input, 'pts')))

    chunk_size = len(framenames) // num_prccess
    chunks_framenames = [framenames[i:i + chunk_size] for i in range(0, len(framenames), chunk_size)]
    if len(chunks_framenames) > num_prccess:
        chunks_framenames[len(chunks_framenames) - 2].extend(chunks_framenames[len(chunks_framenames) - 1])
        del chunks_framenames[len(chunks_framenames) - 1]

    cate_nums_sum = np.zeros(8, int)
    cate_nums = count_cate_num(dir_input, framenames)

    # tasks = []
    # for i, chunk_filepaths in enumerate(chunks_framenames):
    #     task = Process(target=count_cate_num, args=(dir_input, chunk_filepaths))
    #     tasks.append(task)
    #     task.start()
    #
    # for task in tasks:
    #     task.join()

    cate_nums_sum += cate_nums
    print(cate_nums_sum)