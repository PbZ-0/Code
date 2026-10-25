# """ Experimental settings """
class Config():

    hid_dim = 256

    seed = 42

    all_index = 29

    query_set_size = 50

    batch_iter = 100

    freeze_epoch = 20

    mini_task_size = 100

    meta_lr = 0.0001

    task_lr = 0.0001

    inner_batch_size = 128#

    test_batch_size = 128#

    meta_batch_size = 5



    n_classes = 3

    reinit_num = 4

    input_dim = 155


config = Config()
