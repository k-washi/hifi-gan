from dataclasses import dataclass

@dataclass
class ComBDConfig:
    combd_h_u = [
        [16, 64, 256, 1024, 1024, 1024],
        [16, 64, 256, 1024, 1024, 1024],
        [16, 64, 256, 1024, 1024, 1024]
    ]
    combd_d_k = [
        [7, 11, 11, 11, 11, 5],
        [11, 21, 21, 21, 21, 5],
        [15, 41, 41, 41, 41, 5]
    ]
    combd_d_s = [
        [1, 1, 4, 4, 4, 1],
        [1, 1, 4, 4, 4, 1],
        [1, 1, 4, 4, 4, 1]
    ]
    combd_d_d = [
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1]
    ]
    combd_d_g = [
        [1, 4, 16, 64, 256, 1],
        [1, 4, 16, 64, 256, 1],
        [1, 4, 16, 64, 256, 1]
    ]
    combd_d_p = [
        [3, 5, 5, 5, 5, 2],
        [5, 10, 10, 10, 10, 2],
        [7, 20, 20, 20, 20, 2]
    ]
    combd_op_f = [1, 1, 1]
    combd_op_k = [3, 3, 3]
    combd_op_g = [1, 1, 1]

@dataclass
class SBDConfig:
    use_sbd = True
    sbd_filters = [[64, 128, 256, 256, 256],[64, 128, 256, 256, 256],[64, 128, 256, 256, 256],[32, 64, 128, 128, 128]]
    sbd_strides = [[1, 1, 3, 3, 1], [1, 1, 3, 3, 1], [1, 1, 3, 3, 1], [1, 1, 3, 3, 1]]
    sbd_kernel_sizes = [
        [[7, 7, 7],[7, 7, 7],[7, 7, 7],[7, 7, 7],[7, 7, 7]],
        [[5, 5, 5],[5, 5, 5],[5, 5, 5],[5, 5, 5],[5, 5, 5]],
        [[3, 3, 3],[3, 3, 3],[3, 3, 3],[3, 3, 3],[3, 3, 3]],
        [[5, 5, 5],[5, 5, 5],[5, 5, 5],[5, 5, 5],[5, 5, 5]]
    ]
    sbd_dilations = [
        [[5, 7, 11], [5, 7, 11], [5, 7, 11], [5, 7, 11], [5, 7, 11]],
        [[3, 5, 7], [3, 5, 7], [3, 5, 7], [3, 5, 7], [3, 5, 7]],
        [[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]],
        [[1, 2, 3], [1, 2, 3], [1, 2, 3], [2, 3, 5], [2, 3, 5]]
    ]
    sbd_band_ranges =[[0, 6], [0, 11], [0, 16], [0, 64]]
    sbd_transpose = [False, False, False, True]
    pqmf_sbd = [16, 256, 0.03, 10.0]
    pqmf_fsbd = [64, 256, 0.1, 9.0]
    segment_size = 8192

@dataclass
class AvocodoConfig:
    resblock: int = 1
    upsample_rates = [[8], [8], [2], [2]]
    upsample_kernel_sizes = [[16], [16], [4], [4]]
    upsample_initial_channel: int = 512
    resblock_kernel_sizes = [3,7,11]
    resblock_dilation_sizes =  [[1,3,5], [1,3,5], [1,3,5]]
    projection_filters =  [0, 1, 1, 1]
    projection_kernels = [0, 5, 7, 11]
    pqmf_lv1 = [2, 256, 0.25, 10.0]
    pqmf_lv2 = [4, 192, 0.13, 10.0]
    combd: ComBDConfig = ComBDConfig()
    sbd: SBDConfig = SBDConfig()
    
    def update_config(self, config):
        self.sbd.segment_size = config.dataset.train.waveform_length
        self.sbd.use_sbd = config.ml.loss.use_sbd