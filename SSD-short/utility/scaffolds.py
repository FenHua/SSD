from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import tensorflow as tf


# 有关模型恢复的一系列操作
def get_init_fn_for_scaffold(model_dir, checkpoint_path, model_scope, checkpoint_model_scope, checkpoint_exclude_scopes,ignore_missing_vars, name_remap=None):
    if tf.train.latest_checkpoint(model_dir):
        tf.logging.info('Ignoring --checkpoint_path because a checkpoint already exists in %s.' % model_dir)
        return None
    exclusion_scopes = []
    if checkpoint_exclude_scopes:
        exclusion_scopes = [scope.strip() for scope in checkpoint_exclude_scopes.split(',')]  # 获取不需要恢复的层
    variables_to_restore = []
    for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
        excluded = False
        for exclusion in exclusion_scopes:
            if exclusion in var.op.name:
                # 不操作需要排除的层
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)
    if checkpoint_model_scope is not None:
        if checkpoint_model_scope.strip() == '':
            variables_to_restore = {var.op.name.replace(model_scope + '/', ''): var for var in variables_to_restore}
        else:
            variables_to_restore = {var.op.name.replace(model_scope, checkpoint_model_scope.strip()): var for var in variables_to_restore}
        if name_remap is not None:
            renamed_variables_to_restore = dict()
            for var_name, var in variables_to_restore.items():
                found = False
                for k, v in name_remap.items():
                    if k in var_name:
                        renamed_variables_to_restore[var_name.replace(k, v)] = var
                        found = True
                        break
                if not found:
                    renamed_variables_to_restore[var_name] = var
            variables_to_restore = renamed_variables_to_restore
    # 从最新的检查点文件恢复
    checkpoint_path = tf.train.latest_checkpoint(checkpoint_path) if tf.gfile.IsDirectory(checkpoint_path) else checkpoint_path
    tf.logging.info('Fine-tuning from %s. Ignoring missing vars: %s.' % (checkpoint_path, ignore_missing_vars))
    if not variables_to_restore:
        raise ValueError('variables_to_restore cannot be empty')
    if ignore_missing_vars:
        reader = tf.train.NewCheckpointReader(checkpoint_path)
        if isinstance(variables_to_restore, dict):
            var_dict = variables_to_restore
        else:
            var_dict = {var.op.name: var for var in variables_to_restore}
        available_vars = {}
        for var in var_dict:
            if reader.has_tensor(var):
                available_vars[var] = var_dict[var]
            else:
                tf.logging.warning('Variable %s missing in checkpoint %s.', var, checkpoint_path)
        variables_to_restore = available_vars
    if variables_to_restore:
        saver = tf.train.Saver(variables_to_restore, reshape=False)
        saver.build()
        def callback(scaffold, session):
            saver.restore(session, checkpoint_path)
        return callback
    else:
        tf.logging.warning('No Variables to restore.')
        return None