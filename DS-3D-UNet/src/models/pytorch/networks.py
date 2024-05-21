
from typing import Tuple, List, Dict, Union, Any

from torch.nn import Conv3d, MaxPool3d, Upsample, BatchNorm3d, Dropout3d, ReLU, LeakyReLU, Sigmoid
import torch.nn as nn
import torch

from common.exceptionmanager import catch_error_exception
from common.functionutil import ImagesUtil
from imageoperators.imageoperator import CropImage
from models.networks import UNetBase

LIST_AVAIL_NETWORKS = ['UNet3DOriginal',
                       'UNet3DGeneral',
                       'UNet3DPlugin',
                       ]


class UNet(UNetBase, nn.Module):

    def __init__(self,
                 size_image_in: Union[Tuple[int, int, int], Tuple[int, int]],
                 num_levels: int,
                 num_featmaps_in: int,
                 num_channels_in: int,
                 num_classes_out: int,
                 is_use_valid_convols: bool = False,
                 num_levels_valid_convols: int = UNetBase._num_levels_valid_convols_default,
                 ) -> None:
        super(UNet, self).__init__(size_image_in,
                                   num_levels,
                                   num_featmaps_in,
                                   num_channels_in,
                                   num_classes_out,
                                   is_use_valid_convols=is_use_valid_convols,
                                   num_levels_valid_convols=num_levels_valid_convols)
        nn.Module.__init__(self)

        self._shape_input = ImagesUtil.get_shape_channels_first(self._shape_input)
        self._shape_output = ImagesUtil.get_shape_channels_first(self._shape_output)

    def get_network_input_args(self) -> Dict[str, Any]:
        raise NotImplementedError

    def _build_info_crop_where_merge(self) -> None:
        indexes_output_where_merge = [i for i, elem in enumerate(self._names_operations_layers_all)
                                      if elem == 'upsample']
        self._sizes_crop_where_merge = [self._sizes_output_all_layers[ind] for ind in indexes_output_where_merge][::-1]

    def _crop_image_2d(self, input: torch.Tensor, size_crop: Tuple[int, int]) -> torch.Tensor:
        size_input_image = input.shape[-2:]
        limits_out_image = self._get_limits_output_crop(size_input_image, size_crop)
        return CropImage._compute2d_channels_first(input, limits_out_image)

    def _crop_image_3d(self, input: torch.Tensor, size_crop: Tuple[int, int, int]) -> torch.Tensor:
        size_input_image = input.shape[-3:]
        limits_out_image = self._get_limits_output_crop(size_input_image, size_crop)
        return CropImage._compute3d_channels_first(input, limits_out_image)


class UNet3DOriginal(UNet):
    _num_levels_fixed = 5

    def __init__(self,
                 size_image_in: Tuple[int, int, int],
                 num_featmaps_in: int = 16,
                 num_channels_in: int = 1,
                 num_classes_out: int = 1
                 ) -> None:
        super(UNet3DOriginal, self).__init__(size_image_in,
                                             self._num_levels_fixed,
                                             num_featmaps_in,
                                             num_channels_in,
                                             num_classes_out,
                                             is_use_valid_convols=False)
        self._build_model()

    def get_network_input_args(self) -> Dict[str, Any]:
        return {'size_image': self._size_image_in,
                'num_featmaps_in': self._num_featmaps_in,
                'num_channels_in': self._num_channels_in,
                'num_classes_out': self._num_classes_out}

    def _build_model(self) -> None:

        num_featmaps_lev1 = self._num_featmaps_in
        self._convolution_down_lev1_1 = Conv3d(self._num_channels_in, num_featmaps_lev1, kernel_size=3, padding=1)
        self._convolution_down_lev1_2 = Conv3d(num_featmaps_lev1, num_featmaps_lev1, kernel_size=3, padding=1)
        self._pooling_down_lev1 = MaxPool3d(kernel_size=2, padding=0)

        num_featmaps_lev2 = 2 * num_featmaps_lev1
        self._convolution_down_lev2_1 = Conv3d(num_featmaps_lev1, num_featmaps_lev2, kernel_size=3, padding=1)
        self._convolution_down_lev2_2 = Conv3d(num_featmaps_lev2, num_featmaps_lev2, kernel_size=3, padding=1)
        self._pooling_down_lev2 = MaxPool3d(kernel_size=2, padding=0)

        num_featmaps_lev3 = 2 * num_featmaps_lev2
        self._convolution_down_lev3_1 = Conv3d(num_featmaps_lev2, num_featmaps_lev3, kernel_size=3, padding=1)
        self._convolution_down_lev3_2 = Conv3d(num_featmaps_lev3, num_featmaps_lev3, kernel_size=3, padding=1)
        self._pooling_down_lev3 = MaxPool3d(kernel_size=2, padding=0)

        num_featmaps_lev4 = 2 * num_featmaps_lev3
        self._convolution_down_lev4_1 = Conv3d(num_featmaps_lev3, num_featmaps_lev4, kernel_size=3, padding=1)
        self._convolution_down_lev4_2 = Conv3d(num_featmaps_lev4, num_featmaps_lev4, kernel_size=3, padding=1)
        self._pooling_down_lev4 = MaxPool3d(kernel_size=2, padding=0)

        num_featmaps_lev5 = 2 * num_featmaps_lev4
        self._convolution_down_lev5_1 = Conv3d(num_featmaps_lev4, num_featmaps_lev5, kernel_size=3, padding=1)
        self._convolution_down_lev5_2 = Conv3d(num_featmaps_lev5, num_featmaps_lev5, kernel_size=3, padding=1)
        self._upsample_up_lev5 = Upsample(scale_factor=2, mode='nearest')

        num_feats_lev4pl5 = num_featmaps_lev4 + num_featmaps_lev5
        self._convolution_up_lev4_1 = Conv3d(num_feats_lev4pl5, num_featmaps_lev4, kernel_size=3, padding=1)
        self._convolution_up_lev4_2 = Conv3d(num_featmaps_lev4, num_featmaps_lev4, kernel_size=3, padding=1)
        self._upsample_up_lev4 = Upsample(scale_factor=2, mode='nearest')

        num_feats_lev3pl4 = num_featmaps_lev3 + num_featmaps_lev4
        self._convolution_up_lev3_1 = Conv3d(num_feats_lev3pl4, num_featmaps_lev3, kernel_size=3, padding=1)
        self._convolution_up_lev3_2 = Conv3d(num_featmaps_lev3, num_featmaps_lev3, kernel_size=3, padding=1)
        self._upsample_up_lev3 = Upsample(scale_factor=2, mode='nearest')

        num_feats_lev2pl3 = num_featmaps_lev2 + num_featmaps_lev3
        self._convolution_up_lev2_1 = Conv3d(num_feats_lev2pl3, num_featmaps_lev2, kernel_size=3, padding=1)
        self._convolution_up_lev2_2 = Conv3d(num_featmaps_lev2, num_featmaps_lev2, kernel_size=3, padding=1)
        self._upsample_up_lev2 = Upsample(scale_factor=2, mode='nearest')

        num_feats_lev1pl2 = num_featmaps_lev1 + num_featmaps_lev2
        self._convolution_up_lev1_1 = Conv3d(num_feats_lev1pl2, num_featmaps_lev1, kernel_size=3, padding=1)
        self._convolution_up_lev1_2 = Conv3d(num_featmaps_lev1, num_featmaps_lev1, kernel_size=3, padding=1)

        self._classification_last = Conv3d(num_featmaps_lev1, self._num_classes_out, kernel_size=1, padding=0)
        self._activation_last = Sigmoid()

    def forward(self, input: torch.Tensor) -> torch.Tensor:

        hidden_nxt = self._convolution_down_lev1_1(input)
        hidden_nxt = self._convolution_down_lev1_2(hidden_nxt)
        hidden_skip_lev1 = hidden_nxt
        hidden_nxt = self._pooling_down_lev1(hidden_nxt)

        hidden_nxt = self._convolution_down_lev2_1(hidden_nxt)
        hidden_nxt = self._convolution_down_lev2_2(hidden_nxt)
        hidden_skip_lev2 = hidden_nxt
        hidden_nxt = self._pooling_down_lev2(hidden_nxt)

        hidden_nxt = self._convolution_down_lev3_1(hidden_nxt)
        hidden_nxt = self._convolution_down_lev3_2(hidden_nxt)
        hidden_skip_lev3 = hidden_nxt
        hidden_nxt = self._pooling_down_lev3(hidden_nxt)

        hidden_nxt = self._convolution_down_lev4_1(hidden_nxt)
        hidden_nxt = self._convolution_down_lev4_2(hidden_nxt)
        hidden_skip_lev4 = hidden_nxt
        hidden_nxt = self._pooling_down_lev4(hidden_nxt)

        hidden_nxt = self._convolution_down_lev5_1(hidden_nxt)
        hidden_nxt = self._convolution_down_lev5_2(hidden_nxt)
        hidden_nxt = self._upsample_up_lev5(hidden_nxt)

        hidden_nxt = torch.cat([hidden_nxt, hidden_skip_lev4], dim=1)
        hidden_nxt = self._convolution_up_lev4_1(hidden_nxt)
        hidden_nxt = self._convolution_up_lev4_2(hidden_nxt)
        hidden_nxt = self._upsample_up_lev4(hidden_nxt)

        hidden_nxt = torch.cat([hidden_nxt, hidden_skip_lev3], dim=1)
        hidden_nxt = self._convolution_up_lev3_1(hidden_nxt)
        hidden_nxt = self._convolution_up_lev3_2(hidden_nxt)
        hidden_nxt = self._upsample_up_lev3(hidden_nxt)

        hidden_nxt = torch.cat([hidden_nxt, hidden_skip_lev2], dim=1)
        hidden_nxt = self._convolution_up_lev2_1(hidden_nxt)
        hidden_nxt = self._convolution_up_lev2_2(hidden_nxt)
        hidden_nxt = self._upsample_up_lev2(hidden_nxt)

        hidden_nxt = torch.cat([hidden_nxt, hidden_skip_lev1], dim=1)
        hidden_nxt = self._convolution_up_lev1_1(hidden_nxt)
        hidden_nxt = self._convolution_up_lev1_2(hidden_nxt)

        output = self._activation_last(self._classification_last(hidden_nxt))
        return output


class UNet3DGeneral(UNet):
    _num_levels_default = 5
    _num_featmaps_in_default = 16
    _num_channels_in_default = 1
    _num_classes_out_default = 1
    _dropout_rate_default = 0.2
    _type_activate_hidden_default = 'relu'
    _type_activate_output_default = 'sigmoid'
    _num_convols_levels_down_default = 2
    _num_convols_levels_up_default = 2
    _sizes_kernel_convols_levels_down_default = (3, 3, 3)
    _sizes_kernel_convols_levels_up_default = (3, 3, 3)
    _sizes_pooling_levels_default = (2, 2, 2)

    def __init__(self,
                 size_image_in: Tuple[int, int, int],
                 num_levels: int = _num_levels_default,
                 num_featmaps_in: int = _num_featmaps_in_default,
                 num_channels_in: int = _num_channels_in_default,
                 num_classes_out: int = _num_classes_out_default,
                 is_use_valid_convols: bool = False,
                 type_activate_hidden: str = _type_activate_hidden_default,
                 type_activate_output: str = _type_activate_output_default,
                 num_featmaps_levels: List[int] = None,
                 num_convols_levels_down: Union[int, Tuple[int, ...]] = _num_convols_levels_down_default,
                 num_convols_levels_up: Union[int, Tuple[int, ...]] = _num_convols_levels_up_default,
                 sizes_kernel_convols_levels_down: Union[Tuple[int, int, int], List[Tuple[int, int, int]]] =
                 _sizes_kernel_convols_levels_down_default,
                 sizes_kernel_convols_levels_up: Union[Tuple[int, int, int], List[Tuple[int, int, int]]] =
                 _sizes_kernel_convols_levels_up_default,
                 sizes_pooling_levels: Union[Tuple[int, int, int], List[Tuple[int, int, int]]] =
                 _sizes_pooling_levels_default,
                 is_disable_convol_pooling_axialdim_lastlevel: bool = False,
                 is_use_dropout: bool = False,
                 dropout_rate: float = _dropout_rate_default,
                 is_use_dropout_levels_down: Union[bool, List[bool]] = True,
                 is_use_dropout_levels_up: Union[bool, List[bool]] = True,
                 is_use_batchnormalize=False,
                 is_use_batchnormalize_levels_down: Union[bool, List[bool]] = True,
                 is_use_batchnormalize_levels_up: Union[bool, List[bool]] = True
                 ) -> None:
        super(UNet, self).__init__(size_image_in,
                                   num_levels,
                                   num_featmaps_in,
                                   num_channels_in,
                                   num_classes_out,
                                   is_use_valid_convols=is_use_valid_convols)
        self._type_activate_hidden = type_activate_hidden
        self._type_activate_output = type_activate_output

        if num_featmaps_levels:
            self._num_featmaps_levels = num_featmaps_levels
        else:
            # default: double featmaps after every pooling
            self._num_featmaps_levels = [self._num_featmaps_in]
            for i in range(1, self._num_levels):
                self._num_featmaps_levels[i] = 2 * self._num_featmaps_levels[i - 1]

        if type(num_convols_levels_down) == int:
            self._num_convols_levels_down = [num_convols_levels_down] * self._num_levels
        else:
            self._num_convols_levels_down = num_convols_levels_down
        if type(num_convols_levels_up) == int:
            self._num_convols_levels_up = [num_convols_levels_up] * (self._num_levels - 1)
        else:
            self._num_convols_levels_up = num_convols_levels_up

        if type(sizes_kernel_convols_levels_down) == tuple:
            self._sizes_kernel_convols_levels_down = [sizes_kernel_convols_levels_down] * self._num_levels
        else:
            self._sizes_kernel_convols_levels_down = sizes_kernel_convols_levels_down
        if type(sizes_kernel_convols_levels_up) == tuple:
            self._sizes_kernel_convols_levels_up = [sizes_kernel_convols_levels_up] * (self._num_levels - 1)
        else:
            self._sizes_kernel_convols_levels_up = sizes_kernel_convols_levels_up

        if type(sizes_pooling_levels) == tuple:
            self._sizes_pooling_levels = [sizes_pooling_levels] * self._num_levels
        else:
            self._sizes_pooling_levels = sizes_pooling_levels
        self._sizes_upsample_levels = self._sizes_pooling_levels[:-1]

        if is_disable_convol_pooling_axialdim_lastlevel:
            size_kernel_convol_lastlevel = self._sizes_kernel_convols_levels_down[-1]
            self._sizes_kernel_convols_levels_down[-1] = (1, size_kernel_convol_lastlevel[1],
                                                          size_kernel_convol_lastlevel[2])
            size_pooling_lastlevel = self._sizes_pooling_levels[-1]
            self._sizes_pooling_levels[-1] = (1, size_pooling_lastlevel[1], size_pooling_lastlevel[2])

        self._is_use_dropout = is_use_dropout
        if is_use_dropout:
            self._dropout_rate = dropout_rate

            if type(is_use_dropout_levels_down) == bool:
                self._is_use_dropout_levels_down = [is_use_dropout_levels_down] * self._num_levels
            else:
                self._is_use_dropout_levels_down = is_use_dropout_levels_down
            if type(is_use_dropout_levels_up) == bool:
                self._is_use_dropout_levels_up = [is_use_dropout_levels_up] * (self._num_levels - 1)
            else:
                self._is_use_dropout_levels_up = is_use_dropout_levels_up

        self._is_use_batchnormalize = is_use_batchnormalize
        if is_use_batchnormalize:
            if type(is_use_batchnormalize_levels_down) == bool:
                self._is_use_batchnormalize_levels_down = [is_use_batchnormalize_levels_down] * self._num_levels
            else:
                self._is_use_batchnormalize_levels_down = is_use_batchnormalize_levels_down
            if type(is_use_batchnormalize_levels_up) == bool:
                self._is_use_batchnormalize_levels_up = [is_use_batchnormalize_levels_up] * (self._num_levels - 1)
            else:
                self._is_use_batchnormalize_levels_up = is_use_batchnormalize_levels_up

        self._build_model()

    def get_network_input_args(self) -> Dict[str, Any]:
        return {'size_image_in': self._size_image_in,
                'num_levels': self._num_levels,
                'num_featmaps_in': self._num_featmaps_in,
                'num_channels_in': self._num_channels_in,
                'num_classes_out': self._num_classes_out,
                'is_use_valid_convols': self._is_use_valid_convols}

    def _build_model(self) -> None:
        value_padding_convols = 0 if self._is_use_valid_convols else 1

        self._convolutions_levels_down = [[] for i in range(self._num_levels)]
        self._convolutions_levels_up = [[] for i in range(self._num_levels - 1)]
        self._poolings_levels_down = []
        self._upsamples_levels_up = []
        self._batchnormalize_levels_down = [[] for i in range(self._num_levels)]
        self._batchnormalize_levels_up = [[] for i in range(self._num_levels - 1)]

        # ENCODING LAYERS
        for i_lev in range(self._num_levels):
            num_featmaps_in_level = self._num_channels_in if i_lev == 0 else self._num_featmaps_levels[i_lev - 1]
            num_featmaps_out_level = self._num_featmaps_levels[i_lev]

            for i_con in range(self._num_convols_levels_down[i_lev]):
                num_featmaps_in_convol = num_featmaps_in_level if i_con else num_featmaps_in_level
                num_featmaps_out_convol = num_featmaps_out_level

                new_convolution = Conv3d(num_featmaps_in_convol, num_featmaps_out_convol,
                                         kernel_size=self._sizes_kernel_convols_levels_down[i_lev],
                                         padding=value_padding_convols)
                self._convolutions_levels_down[i_lev].append(new_convolution)

                if self._is_use_batchnormalize and self._is_use_batchnormalize_levels_down[i_lev]:
                    new_batchnormalize = BatchNorm3d(num_featmaps_out_convol)
                    self._batchnormalize_levels_down[i_lev].append(new_batchnormalize)

            if (i_lev != self._num_levels - 1):
                new_pooling = MaxPool3d(kernel_size=self._sizes_pooling_levels[i_lev], padding=0)
                self._poolings_levels_down.append(new_pooling)

        # DECODING LAYERS
        for i_lev in range(self._num_levels - 2, -1, -1):
            num_featmaps_in_level = self._num_featmaps_levels[i_lev - 1] + self._num_featmaps_levels[i_lev]
            num_featmaps_out_level = self._num_featmaps_levels[i_lev]

            new_upsample = Upsample(scale_factor=self._sizes_upsample_levels[i_lev], mode='nearest')
            self._upsamples_levels_up.append(new_upsample)

            for i_con in range(self._num_convols_levels_up[i_lev]):
                num_featmaps_in_convol = num_featmaps_in_level if i_con else num_featmaps_in_level
                num_featmaps_out_convol = num_featmaps_out_level

                new_convolution = Conv3d(num_featmaps_in_convol, num_featmaps_out_convol,
                                         kernel_size=self._sizes_kernel_convols_levels_up[i_lev],
                                         padding=value_padding_convols)
                self._convolutions_levels_up[i_lev].append(new_convolution)

                if self._is_use_batchnormalize and self._is_use_batchnormalize_levels_up[i_lev]:
                    new_batchnormalize = BatchNorm3d(num_featmaps_out_convol)
                    self._batchnormalize_levels_up[i_lev].append(new_batchnormalize)

        self._classification_last = Conv3d(self._num_featmaps_in, self._num_classes_out, kernel_size=1, padding=0)

        if self._is_use_dropout:
            self._dropout_all_levels = Dropout3d(self._dropout_rate, inplace=True)

        if self._type_activate_hidden == 'relu':
            self._activation_hidden = ReLU(inplace=True)
        elif self._type_activate_hidden == 'leaky_relu':
            self._activation_hidden = LeakyReLU(inplace=True)
        elif self._type_activate_hidden == 'none':
            def func_activation_none(input: torch.Tensor) -> torch.Tensor:
                return input
            self._activation_hidden = func_activation_none
        else:
            message = 'Type activation hidden not existing: \'%s\'' % (self._type_activate_hidden)
            catch_error_exception(message)

        if self._type_activate_output == 'sigmoid':
            self._activation_last = Sigmoid()
        elif self._type_activate_output == 'linear':
            def func_activation_linear(input: torch.Tensor) -> torch.Tensor:
                return input
            self._activation_last = func_activation_linear
        else:
            message = 'Type activation output not existing: \'%s\' ' % (self._type_activate_output)
            catch_error_exception(message)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        hidden_nxt = input
        hidden_skips_levels = []

        # ENCODING LAYERS
        for i_lev in range(self._num_levels):
            for i_con in range(self._num_convols_levels_down[i_lev]):
                hidden_nxt = self._activation_hidden(self._convolutions_levels_down[i_lev][i_con](hidden_nxt))

                if self._is_use_batchnormalize and self._is_use_batchnormalize_levels_down[i_lev]:
                    hidden_nxt = self._batchnormalize_levels_down[i_lev][i_con](hidden_nxt)

            if self._is_use_dropout and self._is_use_dropout_levels_down[i_lev]:
                hidden_nxt = self._dropout_all_levels(hidden_nxt)

            if (i_lev != self._num_levels - 1):
                hidden_skips_levels.append(hidden_nxt)
                hidden_nxt = self._poolings_levels_down[i_lev](hidden_nxt)

        # DECODING LAYERS
        for i_lev in range(self._num_levels - 2, -1, -1):
            hidden_nxt = self._upsamples_levels_up[i_lev](hidden_nxt)

            hidden_skip_this = hidden_skips_levels[i_lev]
            if self._is_use_valid_convols:
                hidden_skip_this = self._crop_image_3d(hidden_skip_this, self._sizes_crop_where_merge[3])
            hidden_nxt = torch.cat([hidden_nxt, hidden_skip_this], dim=1)

            for i_con in range(self._num_convols_levels_up[i_lev]):
                hidden_nxt = self._activation_hidden(self._convolutions_levels_up[i_lev][i_con](hidden_nxt))

                if self._is_use_batchnormalize and self._is_use_batchnormalize_levels_up[i_lev]:
                    hidden_nxt = self._batchnormalize_levels_up[i_lev][i_con](hidden_nxt)

            if self._is_use_dropout and self._is_use_dropout_levels_up[i_lev]:
                hidden_nxt = self._dropout_all_levels(hidden_nxt)

        output = self._activation_last(self._classification_last(hidden_nxt))
        return output


    

class bunchconv(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(bunchconv,self).__init__()
        self.bunconv=nn.Sequential(
            nn.Conv3d(in_channels,in_channels//4,kernel_size=1,padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels//4,in_channels//4,kernel_size=3,padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels//4,in_channels,kernel_size=1,padding='same'),
            
        )
        self.Relu=nn.ReLU(inplace=True)
        
    def forward(self,x):
        x1=self.bunconv(x)
        x=x+x1
        x=self.Relu(x)
        x1=self.bunconv(x)
        x=x+x1
        x=self.Relu(x)
        x1=self.bunconv(x)
        x=x+x1
        x=self.Relu(x)
        x1=self.bunconv(x)
        x=x+x1
        x=self.Relu(x)
        return x   
class SE_Block(nn.Module):
    def __init__(self, ch_in, reduction=16):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)  # 全局自适应池化
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )
 
    def forward(self, x):
        b, c, _, _,_ = x.size()
        y = self.avg_pool(x).view(b, c) # squeeze操作
        y = self.fc(y).view(b, c, 1, 1,1) # FC获取通道注意力权重，是具有全局信息的
        return x * y.expand_as(x) # 注意力作用每一个通道上
class Upsam2(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(Upsam2,self).__init__()
        self.upsam=nn.Sequential(
            nn.Conv3d(in_channels,out_channels,kernel_size=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(out_channels,out_channels,kernel_size=2,stride=2),
            nn.ReLU(inplace=True),
        )
    def forward(self,x):
        return self.upsam(x)
    
class Upsam4(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(Upsam4,self).__init__()
        self.upsam=nn.Sequential(
            nn.Conv3d(in_channels,out_channels,kernel_size=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(out_channels,out_channels,kernel_size=4,stride=4),
            nn.ReLU(inplace=True),

        )
    def forward(self,x):
        return self.upsam(x)
    
class Upsam8(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(Upsam8,self).__init__()
        self.upsam=nn.Sequential(
            nn.Conv3d(in_channels,out_channels,kernel_size=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(out_channels,out_channels,kernel_size=8,stride=8),
            nn.ReLU(inplace=True),
        )
    def forward(self,x):
        return self.upsam(x)
    
    
# class Upsam2_ds(nn.Module):
#     def __init__(self,in_channels,out_channels):
#         super(Upsam2_ds,self).__init__()
#         self.upsam=nn.Sequential(
#             nn.Conv3d(in_channels,out_channels,kernel_size=1),
#             nn.ReLU(inplace=True),
#             nn.ConvTranspose3d(out_channels,out_channels,kernel_size=2,stride=2),
#             nn.ReLU(inplace=True),
#         )
#     def forward(self,x):
#         return self.upsam(x)
    
# class Upsam4_ds(nn.Module):
#     def __init__(self,in_channels,out_channels):
#         super(Upsam4_ds,self).__init__()
#         self.upsam=nn.Sequential(
#             nn.Conv3d(in_channels,out_channels,kernel_size=1),
#             nn.ReLU(inplace=True),
#             nn.ConvTranspose3d(out_channels,out_channels,kernel_size=4,stride=4),
#             nn.ReLU(inplace=True),
#         )
#     def forward(self,x):
#         return self.upsam(x)
    
# class Upsam8_ds(nn.Module):
#     def __init__(self,in_channels,out_channels):
#         super(Upsam8_ds,self).__init__()
#         self.upsam=nn.Sequential(
#             nn.Conv3d(in_channels,out_channels,kernel_size=1),
#             nn.ReLU(inplace=True),
#             nn.ConvTranspose3d(out_channels,out_channels,kernel_size=8,stride=8),
#             nn.ReLU(inplace=True),
#         )
#     def forward(self,x):
#         return self.upsam(x)

class Upsam2_ds(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(Upsam2_ds,self).__init__()
        self.upsam=nn.Sequential(
            nn.Conv3d(in_channels,out_channels,kernel_size=1),
            nn.Sigmoid(),
        )
    def forward(self,x):
        return self.upsam(x)
    
class MultiFuse(nn.Module):
    def __init__(self,in_channels):
        super(MultiFuse,self).__init__()
        self.conv1=nn.Conv3d(in_channels,in_channels,kernel_size=3,dilation=1,padding='same')
        self.conv2=nn.Conv3d(in_channels,in_channels,kernel_size=3,dilation=3,padding='same')
        self.conv3=nn.Conv3d(in_channels,in_channels,kernel_size=5,dilation=1,padding='same')
        self.conv4=nn.Conv3d(in_channels,in_channels,kernel_size=3,dilation=5,padding='same')
        self.conv5=nn.Conv3d(in_channels,in_channels,kernel_size=7,dilation=1,padding='same')
        self.activation=nn.Sequential(
            nn.Conv3d(in_channels,in_channels,kernel_size=1,padding='same'),
            nn.Sigmoid(),
        )
        self.SE=CBAM(in_channels*6)
        self.conv6=nn.Conv3d(in_channels*6,in_channels,kernel_size=7,dilation=1,padding='same')
        
        
    def forward(self,x):
        x1=self.conv1(x)
        x2=self.conv2(x1)
        x3=self.conv3(x)
        x4=self.conv4(x3)
        x5=self.conv5(x)
        
        x1=self.activation(x1)
        x2=self.activation(x2)
        x3=self.activation(x3)
        x4=self.activation(x4)
        x5=self.activation(x5)
        
        output=torch.cat([x1,x2],dim=1)
        output=torch.cat([output,x3],dim=1)
        output=torch.cat([output,x4],dim=1)
        output=torch.cat([output,x5],dim=1)
        output=torch.cat([output,x],dim=1)
        output=self.SE(output)
        output=self.conv6(output)
        
        
        
        return output
    
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.fc1   = nn.Conv3d(in_planes, in_planes // 4, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv3d(in_planes // 4, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


# 空间注意力机制
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv3d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

# CBAM注意力模块：轻量+即插即用
class CBAM(nn.Module):
    def __init__(self, c1):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out
class ProjectExciteLayer(nn.Module):
    def __init__(self, num_channels,D, H, W, reduction_ratio=2):

        super(ProjectExciteLayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.convModule = nn.Sequential(
            nn.Conv3d(in_channels=num_channels, out_channels=num_channels_reduced, kernel_size=1, stride=1),\
            nn.ReLU(inplace=True),\
            nn.Conv3d(in_channels=num_channels_reduced, out_channels=num_channels, kernel_size=1, stride=1),\
            nn.Sigmoid())
        self.spatialdim = [D, H, W]
        self.D_squeeze = nn.Conv3d(in_channels=D, out_channels=1, kernel_size=1, stride=1)
        self.H_squeeze = nn.Conv3d(in_channels=H, out_channels=1, kernel_size=1, stride=1)
        self.W_squeeze = nn.Conv3d(in_channels=W, out_channels=1, kernel_size=1, stride=1)
        self.C_squeeze = nn.Conv3d(in_channels=num_channels, out_channels=1, kernel_size=1, stride=1)
        
        self.sigmoid=nn.Sigmoid()

    def forward(self, input_tensor):

        squared_tensor = torch.pow(input_tensor, exponent=2)

        # Project:
        # Weight along channels and different axes
        D, H, W = self.spatialdim[0], self.spatialdim[1], self.spatialdim[2]
        D_channel = input_tensor.permute(0, 2, 1, 3, 4)  # B, D, C, H, W
        H_channel = input_tensor.permute(0, 3, 2, 1, 4)  # B, H, D, C, W

        squeeze_tensor_1D = self.D_squeeze(D_channel)  # B, 1, C, H, W

        squeeze_tensor_W = squeeze_tensor_1D.permute(0, 3, 1, 2, 4)  # B, H, 1, C, W
        squeeze_tensor_W = self.H_squeeze(squeeze_tensor_W).permute(0, 3, 2, 1, 4)  # B, C, 1, 1, W
        squeeze_tensor_W = self.C_squeeze(squeeze_tensor_W)# B, 1, 1, 1, W

        squeeze_tensor_H = squeeze_tensor_1D.permute(0, 4, 1, 3, 2)  # B, W, 1, H, C
        squeeze_tensor_H = self.W_squeeze(squeeze_tensor_H).permute(0, 4, 2, 3, 1)  # B, C, 1, H, 1
        squeeze_tensor_H = self.C_squeeze(squeeze_tensor_H) # B, 1, 1, H, 1

        squeeze_tensor_D = self.H_squeeze(H_channel).permute(0, 4, 2, 1, 3)  # B, W, D, 1, C
        squeeze_tensor_D = self.W_squeeze(squeeze_tensor_D).permute(0, 4, 2, 3, 1)  # B, C, D, 1, 1
        squeeze_tensor_D = self.C_squeeze(squeeze_tensor_D) # B, 1, D, 1, 1
        
        squeeze_tensor_C = squeeze_tensor_1D.permute(0, 3, 1, 2, 4) # B, H, 1, C, W
        squeeze_tensor_C = self.H_squeeze(squeeze_tensor_C).permute(0, 4, 2, 1, 3) # B, W, 1, 1, C
        squeeze_tensor_C = self.W_squeeze(squeeze_tensor_C).permute(0, 4, 2, 3, 1)# B, C, 1, 1, 1

        final_squeeze_tensor = squeeze_tensor_W + squeeze_tensor_H + squeeze_tensor_D + squeeze_tensor_C
        # Excitation:
        final_squeeze_tensor = self.convModule(final_squeeze_tensor)

        output_tensor = torch.mul(input_tensor, final_squeeze_tensor)
        
        feature_mapping = torch.sum(squared_tensor, dim=1, keepdim=True)
        output_tensor = output_tensor

        return output_tensor, feature_mapping

def sad_loss(pred, target, encoder_flag=True):
    """
    AD: atention distillation loss
    : param pred: input prediction
    : param target: input target
    : param encoder_flag: boolean, True=encoder-side AD, False=decoder-side AD
    """
    target = target.detach()
    if (target.size(-1) == pred.size(-1)) and (target.size(-2) == pred.size(-2)):
        # target and prediction have the same spatial resolution
        pass
    else:
        if encoder_flag == True:
            # target is smaller than prediction
            # use consecutive layers with scale factor = 2
            target = F.interpolate(target, scale_factor=2, mode='trilinear')
        else:
            # prediction is smaller than target
            # use consecutive layers with scale factor = 2
            pred = F.interpolate(pred, scale_factor=2, mode='trilinear')

    num_batch = pred.size(0)
    pred = pred.view(num_batch, -1)
    target = target.view(num_batch, -1)
    pred = F.softmax(pred, dim=1)
    target = F.softmax(target, dim=1)
    return F.mse_loss(pred, target)
    
class UNet3DPlugin(UNet):
    _num_levels_fixed = 5
    _num_levels_valid_convols_fixed = 3
    _num_featmaps_in_default = 16
    _num_channels_in_default = 1
    _num_classes_out_default = 1
    _dropout_rate_default = 0.2
    _type_activate_hidden_default = 'relu'
    _type_activate_output_default = 'sigmoid'

    def __init__(self,
                 size_image_in: Tuple[int, int, int],
                 num_featmaps_in: int = _num_featmaps_in_default,
                 num_channels_in: int = _num_channels_in_default,
                 num_classes_out: int = _num_classes_out_default,
                 is_use_valid_convols: bool = False,
                 is_valid_convols_deep_levels: bool = False
                 ) -> None:
        super(UNet3DPlugin, self).__init__(size_image_in,
                                           self._num_levels_fixed,
                                           num_featmaps_in,
                                           num_channels_in,
                                           num_classes_out,
                                           is_use_valid_convols=is_use_valid_convols,
                                           num_levels_valid_convols=self._num_levels_valid_convols_fixed)
        self._type_activate_hidden = self._type_activate_hidden_default
        self._type_activate_output = self._type_activate_output_default
        self._is_valid_convols_deep_levels = is_valid_convols_deep_levels

        self._build_model()

    def get_network_input_args(self) -> Dict[str, Any]:
        return {'size_image_in': self._size_image_in,
                'num_featmaps_in': self._num_featmaps_in,
                'num_channels_in': self._num_channels_in,
                'num_classes_out': self._num_classes_out,
                'is_use_valid_convols': self._is_use_valid_convols}

    def _build_model(self) -> None:
        value_padding = 0 if self._is_use_valid_convols else 1
        value_padding_deep_levels = 0 if self._is_valid_convols_deep_levels else 1
        
        self.pe1 = ProjectExciteLayer(16,128,176,176)
        self.pe2 = ProjectExciteLayer(32,64,88,88)
        self.pe3 = ProjectExciteLayer(64,32,44,44)
        self.pe4 = ProjectExciteLayer(128,16,22,22)
        self.pe5 = ProjectExciteLayer(256,8,11,11)
        self.pe6 = ProjectExciteLayer(128,16,22,22)
        self.pe7 = ProjectExciteLayer(64,32,44,44)
        self.pe8 = ProjectExciteLayer(32,64,88,88)
        self.pe9 = ProjectExciteLayer(16,128,176,176)
        self.relu=nn.ReLU(inplace=True)

        
        
        num_featmaps_lev1 = self._num_featmaps_in
        self.bunchconv_1=bunchconv(num_featmaps_lev1,num_featmaps_lev1)
        self.bunchconv_1_1=bunchconv(num_featmaps_lev1,num_featmaps_lev1)
        self._convolution_down_lev1_1 = Conv3d(self._num_channels_in, num_featmaps_lev1, kernel_size=3,
                                               padding=value_padding)
        self._convolution_down_lev1_2 = Conv3d(num_featmaps_lev1, num_featmaps_lev1, kernel_size=3,
                                               padding=value_padding)
        self._pooling_down_lev1 = MaxPool3d(kernel_size=2, padding=0)

        num_featmaps_lev2 = 2 * num_featmaps_lev1
        self.bunchconv_2=bunchconv(num_featmaps_lev2,num_featmaps_lev1)
        self.bunchconv_2_1=bunchconv(num_featmaps_lev2,num_featmaps_lev1)
        self.x2SE=CBAM(num_featmaps_lev2)
        self.x2Up=Upsam2(num_featmaps_lev2,num_featmaps_lev1)
        self.x2Up_1=Upsam2(num_featmaps_lev2,num_featmaps_lev1)
        self.ds1=Upsam2_ds(num_featmaps_lev1,1)
        self._convolution_down_lev2_1 = Conv3d(num_featmaps_lev1, num_featmaps_lev2, kernel_size=3,
                                               padding=value_padding)
        self._convolution_down_lev2_2 = Conv3d(num_featmaps_lev2, num_featmaps_lev2, kernel_size=3,
                                               padding=value_padding)
        self._pooling_down_lev2 = MaxPool3d(kernel_size=2, padding=0)

        num_featmaps_lev3 = 2 * num_featmaps_lev2
        self.bunchconv_3=bunchconv(num_featmaps_lev3,num_featmaps_lev1)
        self.bunchconv_3_1=bunchconv(num_featmaps_lev3,num_featmaps_lev1)
        self.ds2=Upsam2_ds(num_featmaps_lev1,1)
        self._convolution_down_lev3_1 = Conv3d(num_featmaps_lev2, num_featmaps_lev3, kernel_size=3,
                                               padding=value_padding)
        self._convolution_down_lev3_2 = Conv3d(num_featmaps_lev3, num_featmaps_lev3, kernel_size=3,
                                               padding=value_padding)
        self.x3Up=Upsam4(num_featmaps_lev3,num_featmaps_lev1)
        self.x3Up_1=Upsam4(num_featmaps_lev3,num_featmaps_lev1)
        self.ds3=Upsam2_ds(num_featmaps_lev1,1)
        self._pooling_down_lev3 = MaxPool3d(kernel_size=2, padding=0)

        num_featmaps_lev4 = 2 * num_featmaps_lev3
        self.bunchconv_4=bunchconv(num_featmaps_lev4,num_featmaps_lev1)
        self.bunchconv_4_1=bunchconv(num_featmaps_lev4,num_featmaps_lev1)
        self._convolution_down_lev4_1 = Conv3d(num_featmaps_lev3, num_featmaps_lev4, kernel_size=3,
                                               padding=value_padding_deep_levels)
        self._convolution_down_lev4_2 = Conv3d(num_featmaps_lev4, num_featmaps_lev4, kernel_size=3,
                                               padding=value_padding_deep_levels)
        self.x4Up=Upsam8(num_featmaps_lev4,num_featmaps_lev1)
        self.x4Up_1=Upsam8(num_featmaps_lev4,num_featmaps_lev1)
        self._pooling_down_lev4 = MaxPool3d(kernel_size=2, padding=0)

        num_featmaps_lev5 = 2 * num_featmaps_lev4
        self._convolution_down_lev5_1 = Conv3d(num_featmaps_lev4, num_featmaps_lev5, kernel_size=3,
                                               padding=value_padding_deep_levels)
        self._convolution_down_lev5_2 = Conv3d(num_featmaps_lev5, num_featmaps_lev5, kernel_size=3,
                                               padding=value_padding_deep_levels)
        self._upsample_up_lev5 = Upsample(scale_factor=2, mode='nearest')

        num_feats_lev4pl5 = num_featmaps_lev4 + num_featmaps_lev5
        self._convolution_up_lev4_1 = Conv3d(num_feats_lev4pl5, num_featmaps_lev4, kernel_size=3,
                                             padding=value_padding_deep_levels)
        self._convolution_up_lev4_2 = Conv3d(num_featmaps_lev4, num_featmaps_lev4, kernel_size=3,
                                             padding=value_padding_deep_levels)
#         self.x4Up_ds=Upsam8_ds(num_featmaps_lev4,1)
        self._upsample_up_lev4 = Upsample(scale_factor=2, mode='nearest')

        num_feats_lev3pl4 = num_featmaps_lev3 + num_featmaps_lev4
        self._convolution_up_lev3_1 = Conv3d(num_feats_lev3pl4, num_featmaps_lev3, kernel_size=3,
                                             padding=value_padding)
        self._convolution_up_lev3_2 = Conv3d(num_featmaps_lev3, num_featmaps_lev3, kernel_size=3,
                                             padding=value_padding)
#         self.x3Up_ds=Upsam4_ds(num_featmaps_lev3,1)
        self._upsample_up_lev3 = Upsample(scale_factor=2, mode='nearest')

        num_feats_lev2pl3 = num_featmaps_lev2 + num_featmaps_lev3
        self._convolution_up_lev2_1 = Conv3d(num_feats_lev2pl3, num_featmaps_lev2, kernel_size=3,
                                             padding=value_padding)
        self._convolution_up_lev2_2 = Conv3d(num_featmaps_lev2, num_featmaps_lev2, kernel_size=3,
                                             padding=value_padding)
#         self.x2Up_ds=Upsam2_ds(num_featmaps_lev2,1)
        self._upsample_up_lev2 = Upsample(scale_factor=2, mode='nearest')

        num_feats_lay1pl2 = num_featmaps_lev1 + num_featmaps_lev2
        self._convolution_up_lev1_1 = Conv3d(num_feats_lay1pl2, num_featmaps_lev1, kernel_size=3,
                                             padding=value_padding)
        self._convolution_up_lev1_2 = Conv3d(num_featmaps_lev1, num_featmaps_lev1, kernel_size=3,
                                             padding=value_padding)


        self.outputSE=CBAM(num_featmaps_lev1)
        self.MultiFuse=MultiFuse(num_featmaps_lev1)
        self._classification_last = Conv3d(num_featmaps_lev1, self._num_classes_out, kernel_size=1, padding=0)#要改

        if self._type_activate_hidden == 'relu':
            self._activation_hidden = ReLU(inplace=True)
        elif self._type_activate_hidden == 'leaky_relu':
            self._activation_hidden = LeakyReLU(inplace=True)
        elif self._type_activate_hidden == 'linear':
            def func_activation_linear(input: torch.Tensor) -> torch.Tensor:
                return input
            self._activation_hidden = func_activation_linear
        else:
            message = 'Type activation hidden not existing: \'%s\'' % (self._type_activate_hidden)
            catch_error_exception(message)

        if self._type_activate_output == 'sigmoid':
            self._activation_last = Sigmoid()
        elif self._type_activate_output == 'linear':
            def func_activation_linear(input: torch.Tensor) -> torch.Tensor:
                return input
            self._activation_last = func_activation_linear
        else:
            message = 'Type activation output not existing: \'%s\' ' % (self._type_activate_output)
            catch_error_exception(message)

    def forward(self, input: torch.Tensor) -> torch.Tensor:

        hidden_nxt = self._activation_hidden(self._convolution_down_lev1_1(input))
        hidden_nxt = self._activation_hidden(self._convolution_down_lev1_2(hidden_nxt))
        hidden_skip_lev1 = hidden_nxt
        hidden_skip_lev1_1=self.bunchconv_1(hidden_skip_lev1)
        hidden_nxt,_=self.pe1(hidden_nxt)
        hidden_nxt = self._pooling_down_lev1(hidden_nxt)

        hidden_nxt = self._activation_hidden(self._convolution_down_lev2_1(hidden_nxt))
        hidden_nxt = self._activation_hidden(self._convolution_down_lev2_2(hidden_nxt))
        x2_sub=self.bunchconv_2(hidden_nxt)
        hidden_nxt,_=self.pe2(hidden_nxt)
        hidden_skip_lev2 = hidden_nxt
#         x2_sub=self.x2SE(hidden_skip_lev2)
        x2_sub_up1=self.x2Up(x2_sub)
        hidden_nxt = self._pooling_down_lev2(hidden_nxt)

        hidden_nxt = self._activation_hidden(self._convolution_down_lev3_1(hidden_nxt))
        hidden_nxt = self._activation_hidden(self._convolution_down_lev3_2(hidden_nxt))
        x3_sub=self.bunchconv_3(hidden_nxt)
        hidden_nxt,_=self.pe3(hidden_nxt)
        hidden_skip_lev3 = hidden_nxt
        x3_sub_up1=self.x3Up(x3_sub)
        hidden_nxt = self._pooling_down_lev3(hidden_nxt)

        hidden_nxt = self._activation_hidden(self._convolution_down_lev4_1(hidden_nxt))
        hidden_nxt = self._activation_hidden(self._convolution_down_lev4_2(hidden_nxt))
        x4_sub=self.bunchconv_4(hidden_nxt)
        hidden_nxt,_=self.pe4(hidden_nxt)
        hidden_skip_lev4 = hidden_nxt
        x4_sub_up1=self.x4Up(x4_sub)
        hidden_nxt = self._pooling_down_lev4(hidden_nxt)

        hidden_nxt = self._activation_hidden(self._convolution_down_lev5_1(hidden_nxt))
        hidden_nxt = self._activation_hidden(self._convolution_down_lev5_2(hidden_nxt))
        hidden_nxt,mapping5=self.pe5(hidden_nxt)
        hidden_nxt = self._upsample_up_lev5(hidden_nxt)

        if self._is_use_valid_convols:
            hidden_skip_lev4 = self._crop_image_3d(hidden_skip_lev4, self._sizes_crop_where_merge[3])
#         print(hidden_nxt.shape)
#         print(hidden_skip_lev4.shape)
        hidden_nxt = torch.cat([hidden_nxt, hidden_skip_lev4], dim=1)
        hidden_nxt = self._activation_hidden(self._convolution_up_lev4_1(hidden_nxt))
        hidden_nxt = self._activation_hidden(self._convolution_up_lev4_2(hidden_nxt))
#         ds_4=self._activation_last(self.x4Up_ds(hidden_nxt))
        hidden_nxt,mapping6=self.pe6(hidden_nxt)
        
        hidden_nxt = self._upsample_up_lev4(hidden_nxt)

        if self._is_use_valid_convols:
            hidden_skip_lev3 = self._crop_image_3d(hidden_skip_lev3, self._sizes_crop_where_merge[2])
        hidden_nxt = torch.cat([hidden_nxt, hidden_skip_lev3], dim=1)
        hidden_nxt = self._activation_hidden(self._convolution_up_lev3_1(hidden_nxt))
        hidden_nxt = self._activation_hidden(self._convolution_up_lev3_2(hidden_nxt))
#         ds_3=self._activation_last(self.x3Up_ds(hidden_nxt))
        hidden_nxt,mapping7=self.pe7(hidden_nxt)
        
        hidden_nxt = self._upsample_up_lev3(hidden_nxt)

        if self._is_use_valid_convols:
            hidden_skip_lev2 = self._crop_image_3d(hidden_skip_lev2, self._sizes_crop_where_merge[1])
        hidden_nxt = torch.cat([hidden_nxt, hidden_skip_lev2], dim=1)
        hidden_nxt = self._activation_hidden(self._convolution_up_lev2_1(hidden_nxt))
        hidden_nxt = self._activation_hidden(self._convolution_up_lev2_2(hidden_nxt))
#         ds_2=self._activation_last(self.x2Up_ds(hidden_nxt))
        hidden_nxt,mapping8=self.pe8(hidden_nxt)
        
        hidden_nxt = self._upsample_up_lev2(hidden_nxt)

        if self._is_use_valid_convols:
            hidden_skip_lev1 = self._crop_image_3d(hidden_skip_lev1, self._sizes_crop_where_merge[0])
        hidden_nxt = torch.cat([hidden_nxt, hidden_skip_lev1], dim=1)
        hidden_nxt = self._activation_hidden(self._convolution_up_lev1_1(hidden_nxt))
        hidden_nxt = self._activation_hidden(self._convolution_up_lev1_2(hidden_nxt))
        hidden_nxt,mapping9=self.pe9(hidden_nxt)
        
        x1_sub=x3_sub_up1+x4_sub_up1+x2_sub_up1+hidden_skip_lev1_1
        x1_sub=self.bunchconv_1_1(x1_sub)
        x2_sub=self.bunchconv_2_1(x2_sub)
        x3_sub=self.bunchconv_3_1(x3_sub)
        x4_sub=self.bunchconv_4_1(x4_sub)
        x2_sub_up2=self.x2Up_1(x2_sub)
        ds_1=self.ds1(x2_sub_up2)
        x3_sub_up2=self.x3Up_1(x3_sub)
        ds_2=self.ds2(x3_sub_up2)
        x4_sub_up2=self.x4Up_1(x4_sub)
        ds_3=self.ds3(x4_sub_up2)
        
        

        output=hidden_nxt+x3_sub_up2+x4_sub_up2+x2_sub_up2+x1_sub
        output=self.relu(output)
 
         #output=self.outputSE(output)
#         output=self.MultiFuse(output)
        output = self._activation_last(self._classification_last(output))
        return output
