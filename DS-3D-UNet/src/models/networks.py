
from typing import Tuple, List, Union

from common.exceptionmanager import catch_error_exception
from imageoperators.boundingboxes import BoundBox3DType, BoundBox2DType


class NeuralNetwork(object):

    def __init__(self,
                 shape_input: Tuple[int, ...],
                 shape_output: Tuple[int, ...]
                 ) -> None:
        self._shape_input = shape_input
        self._shape_output = shape_output

    def get_shape_input(self) -> Tuple[int, ...]:
        return self._shape_input

    def get_shape_output(self) -> Tuple[int, ...]:
        return self._shape_output

    def count_model_params(self) -> int:
        raise NotImplementedError

    def _build_model(self) -> None:
        raise NotImplementedError


class ConvNetBase(NeuralNetwork):

    def __init__(self,
                 size_image_in: Union[Tuple[int, int, int], Tuple[int, int]],
                 num_featmaps_in: int,
                 num_channels_in: int,
                 num_classes_out: int,
                 is_use_valid_convols: bool = False
                 ) -> None:
        self._size_image_in = size_image_in
        self._num_featmaps_in = num_featmaps_in
        self._num_channels_in = num_channels_in
        self._num_classes_out = num_classes_out
        self._is_use_valid_convols = is_use_valid_convols

        if self._is_use_valid_convols:
            self._build_auxiliar_data_valid_convols()

        shape_input = self._size_image_in + (self._num_channels_in,)
        shape_output = self.get_size_output_last_layer() + (self._num_classes_out,)

        super(ConvNetBase, self).__init__(shape_input, shape_output)

    def _build_auxiliar_data_valid_convols(self):
        self._names_operations_layers_all = []
        self._build_names_operation_layers()
        self._build_sizes_output_layers_all()

    def _build_names_operation_layers(self) -> None:
        raise NotImplementedError

    def get_size_output_last_layer(self) -> Union[Tuple[int, int, int], Tuple[int, int]]:
        if self._is_use_valid_convols:
            return self._get_size_output_group_layers(level_begin=0, level_end=len(self._names_operations_layers_all))
        else:
            return self._size_image_in

    def _get_size_output_layer(self, size_input: Union[Tuple[int, int, int], Tuple[int, int]], name_operation: str
                               ) -> Union[Tuple[int, int, int], Tuple[int, int]]:
        if name_operation == 'convols':
            if self._is_use_valid_convols:
                return self._get_size_output_valid_convolution(size_input)
            else:
                return size_input
        elif name_operation == 'convols_padded':
            return size_input
        elif name_operation == 'pooling':
            return self._get_size_output_pooling(size_input)
        elif name_operation == 'upsample':
            return self._get_size_output_upsample(size_input)
        elif name_operation == 'classify':
            return size_input
        else:
            raise NotImplementedError

    def _get_size_output_group_layers(self, level_begin: int = 0, level_end: int = None
                                      ) -> Union[Tuple[int, int, int], Tuple[int, int]]:
        if not level_end:
            level_end = len(self._names_operations_layers_all)

        if level_end < level_begin or \
                level_begin >= len(self._names_operations_layers_all) or \
                level_end > len(self._names_operations_layers_all):
            message = 'ConvNetBase: wrong input \'level_begin\' (%s) or \'level_end\' (%s)' % (level_begin, level_end)
            catch_error_exception(message)

        in_names_operation_layers = self._names_operations_layers_all[level_begin:level_end]

        if level_begin == 0:
            size_input = self._size_image_in
        else:
            size_input = self._get_size_output_group_layers(level_begin=0, level_end=level_begin)

        size_nxt = size_input
        for name_operation in in_names_operation_layers:
            size_nxt = self._get_size_output_layer(size_nxt, name_operation)

        return size_nxt

    def _build_sizes_output_layers_all(self) -> List[Union[Tuple[int, int, int], Tuple[int, int]]]:
        self._sizes_output_all_layers = []
        size_nxt = self._size_image_in
        for name_operation in self._names_operations_layers_all:
            size_nxt = self._get_size_output_layer(size_nxt, name_operation)
            self._sizes_output_all_layers.append(size_nxt)

        return self._sizes_output_all_layers

    @staticmethod
    def _get_size_output_valid_convolution(size_input: Union[Tuple[int, int, int], Tuple[int, int]],
                                           size_kernel: int = 3
                                           ) -> Union[Tuple[int, int, int], Tuple[int, int]]:
        ndims = len(size_input)
        size_output = []
        for i in range(ndims):
            size_out_1d = size_input[i] - size_kernel + 1
            size_output.append(size_out_1d)

        if ndims == 3:
            return (size_output[0], size_output[1], size_output[2])
        else:
            return (size_output[0], size_output[1])

    @staticmethod
    def _get_size_output_pooling(size_input: Union[Tuple[int, int, int], Tuple[int, int]],
                                 size_pool: int = 2
                                 ) -> Union[Tuple[int, int, int], Tuple[int, int]]:
        ndims = len(size_input)
        size_output = []
        for i in range(ndims):
            size_out_1d = size_input[i] // size_pool
            size_output.append(size_out_1d)

        if ndims == 3:
            return (size_output[0], size_output[1], size_output[2])
        else:
            return (size_output[0], size_output[1])

    @staticmethod
    def _get_size_output_upsample(size_input: Union[Tuple[int, int, int], Tuple[int, int]],
                                  size_upsample: int = 2
                                  ) -> Union[Tuple[int, int, int], Tuple[int, int]]:
        ndims = len(size_input)
        size_output = []
        for i in range(ndims):
            size_out_1d = size_input[i] * size_upsample
            size_output.append(size_out_1d)

        if ndims == 3:
            return (size_output[0], size_output[1], size_output[2])
        else:
            return (size_output[0], size_output[1])


class UNetBase(ConvNetBase):
    _num_levels_valid_convols_default = 3

    def __init__(self,
                 size_image_in: Union[Tuple[int, int, int], Tuple[int, int]],
                 num_levels: int,
                 num_featmaps_in: int,
                 num_channels_in: int,
                 num_classes_out: int,
                 is_use_valid_convols: bool = False,
                 num_levels_valid_convols: int = _num_levels_valid_convols_default,
                 ) -> None:
        self._num_levels = num_levels

        if is_use_valid_convols:
            # option to enable zero-padding in the deeper conv. layers of the UNet, to relax the reduction
            # of size of network output due to valid convolutions
            self._num_levels_valid_convols = num_levels_valid_convols
        else:
            self._num_levels_valid_convols = None

        super(UNetBase, self).__init__(size_image_in, num_featmaps_in, num_channels_in, num_classes_out,
                                       is_use_valid_convols=is_use_valid_convols)

    def _build_auxiliar_data_valid_convols(self):
        super(UNetBase, self)._build_auxiliar_data_valid_convols()
        self._build_info_crop_where_merge()

    def _build_names_operation_layers(self) -> None:
        if self._num_levels == 1:
            self._names_operations_layers_all = ['convols'] * 4 + ['classify']

        elif self._is_use_valid_convols \
                and self._num_levels > self._num_levels_valid_convols:
            num_levels_nonpadded = self._num_levels_valid_convols
            num_levels_padded_exclast = self._num_levels - num_levels_nonpadded - 1
            self._names_operations_layers_all = \
                num_levels_nonpadded * (['convols'] * 2 + ['pooling']) \
                + num_levels_padded_exclast * (['convols_padded'] * 2 + ['pooling']) \
                + ['convols_padded'] * 2 \
                + num_levels_padded_exclast * (['upsample'] + ['convols_padded'] * 2) \
                + num_levels_nonpadded * (['upsample'] + ['convols'] * 2) \
                + ['classify']
        else:
            self._names_operations_layers_all = \
                (self._num_levels - 1) * (['convols'] * 2 + ['pooling']) \
                + ['convols'] * 2 \
                + (self._num_levels - 1) * (['upsample'] + ['convols'] * 2) \
                + ['classify']

    def _build_info_crop_where_merge(self) -> None:
        raise NotImplementedError

    @staticmethod
    def _get_limits_output_crop_1d(size_input_1d: int, size_crop_1d: int) -> Tuple[int, int]:
        coord_begin = int((size_input_1d - size_crop_1d) / 2)
        coord_end = coord_begin + size_crop_1d
        return (coord_begin, coord_end)

    @classmethod
    def _get_limits_output_crop(cls,
                                size_input: Union[Tuple[int, int, int], Tuple[int, int]],
                                size_crop: Union[Tuple[int, int, int], Tuple[int, int]]
                                ) -> Union[BoundBox3DType, BoundBox2DType]:
        ndims = len(size_input)
        limits_output_crop = []
        for i in range(ndims):
            (limit_left, limit_right) = cls._get_limits_output_crop_1d(size_input[i], size_crop[i])
            limits_output_crop.append((limit_left, limit_right))

        if ndims == 3:
            return (limits_output_crop[0], limits_output_crop[1], limits_output_crop[2])
        else:
            return (limits_output_crop[0], limits_output_crop[1])

    @staticmethod
    def _get_size_borders_output_crop(size_input: Union[Tuple[int, int, int], Tuple[int, int]],
                                      size_crop: Union[Tuple[int, int, int], Tuple[int, int]]
                                      ) -> Union[Tuple[int, int, int], Tuple[int, int]]:
        ndims = len(size_input)
        size_borders_output_crop = []
        for i in range(ndims):
            size_borders_crop_1d = int((size_input[i] - size_crop[i]) / 2)
            size_borders_output_crop.append(size_borders_crop_1d)

        if ndims == 3:
            return (size_borders_output_crop[0], size_borders_output_crop[1], size_borders_output_crop[2])
        else:
            return (size_borders_output_crop[0], size_borders_output_crop[1])
