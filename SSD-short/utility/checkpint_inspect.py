from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tensorflow.python import pywrap_tensorflow


def print_tensors_in_checkpoint_file(file_name, tensor_name, all_tensors):
    # 输出检查点文件内容
    try:
        reader = pywrap_tensorflow.NewCheckpointReader(file_name)
        if all_tensors:
            var_to_shape_map = reader.get_variable_to_shape_map()
            for key in var_to_shape_map:
                print("tensor_name: ", key)
                print(reader.get_tensor(key))
        elif not tensor_name:
            print(reader.debug_string().decode("utf-8"))
        else:
            print("tensor_name: ", tensor_name)
            print(reader.get_tensor(tensor_name))
    except Exception as e:
        print(str(e))
        if "corrupted compressed block contents" in str(e):
            print("It's likely that your checkpoint file has been compressed "
                  "with SNAPPY.")


def print_all_tensors_name(file_name):
    # 输出所有tensor名字
    try:
        reader = pywrap_tensorflow.NewCheckpointReader(file_name)
        var_to_shape_map = reader.get_variable_to_shape_map()
        for key in var_to_shape_map:
            print(key)
    except Exception as e:
        print(str(e))
        if "corrupted compressed block contents" in str(e):
            print("It's likely that your checkpoint file has been compressed "
                  "with SNAPPY.")

if __name__ == "__main__":
    print_all_tensors_name('/home/yhq/Desktop/SSD-short/model/vgg16_reducedfc.ckpt')