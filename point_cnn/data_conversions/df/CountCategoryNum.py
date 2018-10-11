

if __name__ == '__main__':
    num_prccess = 6
    dir_input = "/home/leon/Disk/dataset/Downloads/DataFountain/dataset/training"
    dir_output = "/home/leon/Disk/dataset/Downloads/DataFountain/dataset/training"
    max_distance = 1
    padding = 0.3

    dir_bbox = os.path.join(dir_output, 'bbox')
    dir_vis_bbox = os.path.join(dir_output, 'vis_bbox')
    if not os.path.exists(dir_bbox):
        os.makedirs(dir_bbox)
    if not os.path.exists(dir_vis_bbox):
        os.makedirs(dir_vis_bbox)

    framenames = sorted(os.listdir(os.path.join(dir_input, 'pts')))

    chunk_size = len(framenames) // num_prccess
    chunks_framenames = [framenames[i:i + chunk_size] for i in range(0, len(framenames), chunk_size)]
    if len(chunks_framenames) > num_prccess:
        chunks_framenames[len(chunks_framenames) - 2].extend(chunks_framenames[len(chunks_framenames) - 1])
        del chunks_framenames[len(chunks_framenames) - 1]

    # prepare_df_box(dir_input, framenames, dir_output)
    tasks = []
    for i, chunk_filepaths in enumerate(chunks_framenames):
        task = Process(target=prepare_df_box, args=(dir_input, chunk_filepaths, dir_output))
        tasks.append(task)
        task.start()

    for task in tasks:
        task.join()