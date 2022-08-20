def print_cmdline_args(args: dict):
    for k, v in args.__dict__.items():
        if isinstance(v, list):
            print('{0}'.format(k))
            for vv in v:
                print('{0:20}{1}'.format('', vv))
        elif isinstance(v, dict):
            print('{0}'.format(k))
            shift = max([len(x) for x in v.keys()]) + 1
            for kk, vv in v.items():
                print('{0:{shift}}{1}'.format(kk, vv, shift=shift))
        else:
            print('{0:20}{1}'.format(k, v))
