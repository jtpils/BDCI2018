import os
import numpy as np
from multiprocessing import Process
from utils import df_utils


def clear_data(dir_df, framenames, dir_out):
    test_num = -1
    for i, framename in enumerate(framenames):
        print(framename)
        if i == test_num:
            break

        pts_ins, categories = df_utils.load_frame(dir_df, framename)
        pts_ins_cleared = []
        categories_cleared = []
        for p, c in zip(pts_ins, categories):
            if not(abs(p[0]) < 0.5 and abs(p[1]) < 0.5 and abs(p[2]) < 0.5):
                pts_ins_cleared.append(p)
                categories_cleared.append(c)

        pts_ins_cleared = np.array(pts_ins_cleared)
        categories_cleared = np.array(categories_cleared)
        df_utils.save_frame_to_bin(dir_out, framename, pts_ins_cleared, categories_cleared, False)


if __name__ == '__main__':
    num_prccess = 3
    dir_input = "/home/leon/Disk/dataset/Downloads/DataFountain/dataset/training"
    dir_output = "/home/leon/Disk/dataset/DataFountain/training_cleared"

    dir_cleared = os.path.join(dir_output, 'data_bin')
    if not os.path.exists(dir_cleared):
        os.makedirs(dir_cleared)
    dir_cleared = os.path.join(dir_output, 'ply_colored')
    if not os.path.exists(dir_cleared):
        os.makedirs(dir_cleared)

    framenames = sorted(os.listdir(os.path.join(dir_input, 'pts')))

    chunk_size = len(framenames) // num_prccess
    chunks_framenames = [framenames[i:i + chunk_size] for i in range(0, len(framenames), chunk_size)]
    if len(chunks_framenames) > num_prccess:
        chunks_framenames[len(chunks_framenames) - 2].extend(chunks_framenames[len(chunks_framenames) - 1])
        del chunks_framenames[len(chunks_framenames) - 1]

    # prepare_df_box(dir_input, framenames, dir_output)
    tasks = []
    for i, chunk_filepaths in enumerate(chunks_framenames):
        task = Process(target=clear_data, args=(dir_input, chunk_filepaths, dir_output))
        tasks.append(task)
        task.start()

    for task in tasks:
        task.join()