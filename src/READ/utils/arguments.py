import argparse 
import pydoc
from pathlib import Path
import munch


class ActionNoYes(argparse.Action):
    def __init__(self, 
                option_strings,
                dest,
                nargs=0,
                const=None,
                default=None,
                type=None,
                choices=None,
                required=False,
                help="",
                metavar=None):

        assert len(option_strings) == 1
        assert option_strings[0][:2] == '--'
        
        name= option_strings[0][2:]
        help += f'Use "--{name}" for True, "--no-{name}" for False'
        super(ActionNoYes, self).__init__(['--' + name, '--no-' + name], 
                                          dest=dest,
                                          nargs=nargs,
                                          const=const,
                                          default=default,
                                          type=type,
                                          choices=choices, 
                                          required=required, 
                                          help=help,
                                          metavar=metavar)
        
    def __call__(self, parser, namespace, values, option_string=None):
        if option_string.startswith('--no-'):
            setattr(namespace, self.dest, False)
        else:
            setattr(namespace, self.dest, True)


class SplitStr(argparse.Action):
    
    def split(self, x):
        if x == '':
            return []
        else:
            return [self.elem_type(y) for y in x.split(self.delimiter)]

    def __init__(self, 
                option_strings,
                dest,
                nargs=None,
                const=None,
                default=None,
                type=None,
                choices=None,
                required=False,
                help="",
                metavar=None,
                delimiter=',',
                elem_type=str):

        self.delimiter = delimiter
        self.elem_type = elem_type
        
        default = self.split(default)
        super(SplitStr, self).__init__(option_strings, 
                                          dest=dest,
                                          nargs=nargs,
                                          const=const,
                                          default=default,
                                          type=type,
                                          choices=choices, 
                                          required=required, 
                                          help=help,
                                          metavar=metavar)
        
    def __call__(self, parser, namespace, values, option_string=None):
        print(values)
        setattr(namespace, self.dest, self.split(values))


class MyArgumentParser(argparse.ArgumentParser):
    def __init__(self, **kwargs):
        super(MyArgumentParser, self).__init__(**kwargs)
        self.register('action', 'store_bool', ActionNoYes)
        self.register('action', 'split_str',  SplitStr) 

    def add(self, *args, **kwargs):
        return self.add_argument(*args, **kwargs)


def parse_image_size(string):
    error_msg = 'size must have format WxH'
    tokens = string.split('x')
    if len(tokens) != 2:
        raise argparse.ArgumentTypeError(error_msg)
    try:
        w = int(tokens[0])
        h = int(tokens[1])
        return w, h
    except ValueError:
        raise argparse.ArgumentTypeError(error_msg)


PREFIX = '___'


def eval_modules(config):
    _upd = {}
    for arg in config:
        if isinstance(arg, str) and arg.endswith('_module'):
            # print(arg, config[arg])
            m = pydoc.locate(config[arg])
            if not m:
                raise ValueError(f"module {arg}={config[arg]} not found")
            else:
                config[arg], _upd[PREFIX + arg] = m, config[arg]
        elif isinstance(config[arg], dict):
            eval_modules(config[arg])
    config.update(_upd)
            

def eval_paths(config):
    _upd = {}
    for arg in config:
        if isinstance(arg, str) and arg.endswith('_path'):
            config[arg], _upd[PREFIX + arg] = Path(config[arg]), config[arg]
        elif isinstance(config[arg], dict):
            eval_paths(config[arg])
    config.update(_upd)


def eval_functions(config):
    _upd = {}
    for arg in config:
        if isinstance(arg, str) and arg.endswith('_func'):
            config[arg], _upd[PREFIX + arg] = eval(config[arg]), config[arg]
        elif isinstance(config[arg], dict):
            eval_functions(config[arg])
    config.update(_upd)


def eval_args(args):
    args = vars(args)
    
    eval_modules(args)
    eval_paths(args)
    eval_functions(args)

    return munch.munchify(args)


def deval_args(args):
    args = vars(args)

    for key in list(args):
        if key.startswith(PREFIX):
            args[key.replace(PREFIX, '')] = args[key]
            del args[key]

    return args
