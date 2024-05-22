from .hoi import build
from .hoi_sep import build as build_sep

def build_model(args):
    if args.uncertainty:
        return build_sep(args)
    else:
        return build(args)
