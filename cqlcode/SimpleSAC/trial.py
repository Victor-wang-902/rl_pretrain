import numpy as np
import absl.app
import absl.flags

def define_flags_with_default(**kwargs):
    for key, val in kwargs.items():
        if isinstance(val, bool):
            # Note that True and False are instances of int.
            absl.flags.DEFINE_bool(key, val, 'automatically defined flag')
        elif isinstance(val, int):
            absl.flags.DEFINE_integer(key, val, 'automatically defined flag')
        elif isinstance(val, float):
            absl.flags.DEFINE_float(key, val, 'automatically defined flag')
        elif isinstance(val, str):
            absl.flags.DEFINE_string(key, val, 'automatically defined flag')
        else:
            raise ValueError('Incorrect value type')
    return kwargs

def flatten_config_dict(config, prefix=None):
    output = {}
    for key, val in config.items():
        if prefix is not None:
            next_prefix = '{}.{}'.format(prefix, key)
        else:
            next_prefix = key
        output[next_prefix] = val
    return output

def get_user_flags(flags, flags_def):
    output = {}
    for key in flags_def:
        val = getattr(flags, key)
        output[key] = val

    return output


actual_setting = {
    'env':'zzz',
    'seed':22,
}

parameter_dict = {
    'env': 'hopper-medium-v2',
    'max_traj_length': 1000,
    'seed': 42,
    'device': 'cuda',
}

FLAGS_DEF = define_flags_with_default(
    **parameter_dict
)


def main(argv):
    FLAGS = absl.flags.FLAGS
    variant = get_user_flags(FLAGS, FLAGS_DEF)
    print(variant)

if __name__ == '__main__':
    absl.app.run(main)






