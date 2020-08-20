#-*- Encoding: UTF-8 -*-

from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types

from os.path import join

"""
Following this example:
https://towardsdatascience.com/fast-data-augmentation-in-pytorch-using-nvidia-dali-68f5432e1f5f
"""

# dir_root = "/gpfs/wolf/proj-shared/gen141/disruptCNN/ecei_d3d/"
# dir_root_data = join(dir_root, "data")


# d3d_clear_ecei.final.txt:
# 1600 lines, 5713 sequences
# d3d_disrupt_ecei.final.txt:
# 1289 lines, 3373 sequences
# Total: 9086 sequences, 1289 disruptive: ~14%disruptive discharges


# We need an ExternalSource operator. That allows us to use external data sources
# as input to the pipeline


# class ECEIInputIterator():
#     def __init__(self, batch_size, data_dir):
#         self.data_dir = data_dir
#
#         self.clear_file = join(root_dir, "d3d_clear_ecei.final.txt")
#         self.disrupt_file = join(root_dir, "d3d_disrupt_ecei.final.txt")
#
#
#     def __iter__(self):
#         self.i = 0
#         self.n = 20
#
#         return self
#
#
#     def __next__(self):
#         batch = []
#         labels = []
#
#         for _ in range(self.batch_size):
#             # Randomly pick either a disruptive or a clear sequence
#             choice = int(np.round(np.random.random()))
#
#             if choice == 0: # Pick a clear sequence
#                 continue
#             elif choice == 1: # Pick a disruptive sequence
#                 continue
#             else:
#                 raise ValueError("choice should be either 0 or 1")
#
#
# train_loader, val_loader, test_loader = data_generator(dataset, args.batch_size,
#                                                        distributed=args.distributed,
#                                                        num_workers=args.workers,
#                                                        undersample=args.undersample)


class EceiPipeline_v01(Pipeline):
    """Following https://docs.nvidia.com/deeplearning/dali/archives/dali_0180_beta/dali-developer-guide/docs/examples/pytorch/pytorch-external_input.html"""
    def __init__(self, data_iterator, batch_size, num_threads, device_id=0):
        super(EceiPipeline_v01, self).__init__(batch_size, num_threads, device_id=0, seed=12)

        self.input = ops.ExternalSource()
        self.input_label = ops.ExternalSource()
        self.input_global_index = ops.ExternalSource()
        self.input_weight = ops.ExternalSource()

        self.data_iterator = data_iterator
        self.iterator = iter(self.data_iterator)


    def define_graph(self):
        self.sequence = self.input()
        self.labels = self.input_label()
        self.global_index = self.input_global_index()
        self.weight = self.input_weight()

        return (self.sequence, self.labels, self.global_index, self.weight)

    def iter_setup(self):
        try:
            data, target, global_index, weight = self.iterator.next()
            print("data", type(data), data.shape)
            print("target", type(target), target.shape)
            print("global_index", type(global_index), global_index.shape)
            print("weight", type(weight), weight.shape)

            self.feed_input(self.sequence, data.numpy())
            self.feed_input(self.labels, target.numpy())
            self.feed_input(self.global_index, global_index.numpy())
            self.feed_input(self.weight, weight.numpy())
        except StopIteration:
            self.iterator = iter(self.data_iterator)
            raise StopIteration





# End of file loader_dali.py
