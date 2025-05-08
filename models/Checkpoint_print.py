from rich import print

DEFAULT_WIDTH  = 80
DEFAULT_RANGE  = 35
TITLE_COLOR    = "yellow"
DEFAULT_SYMBOL = '='

def print_header(title: str, args: dict = None, symbol: str = DEFAULT_SYMBOL, width: int = DEFAULT_WIDTH, param_range: int = DEFAULT_RANGE, color: str = TITLE_COLOR):
    """Prints a header for the specified section with a title."""
    print('\n')
    print(symbol * width)
    print(f"[{color}]{title.center(width)}[{color}]")
    print(symbol * width + "\n")

    # print aditional arguments
    if args is not None: print_arguments(args, param_range)

def print_arguments(args: dict, param_range: int = DEFAULT_RANGE):
    # print aditional arguments
    for key, value in args.items():
        print(f"> {key:<{param_range}}: {value}")

def print_small_header(title: str, symbol: str = DEFAULT_SYMBOL, width: int = DEFAULT_WIDTH, color: str = TITLE_COLOR):
    print(symbol * width)
    print(f"[{color}]{title.center(width)}[{color}]\n")

def print_separator(symbol: str = DEFAULT_SYMBOL, width: int = DEFAULT_WIDTH):
    print("\n" + symbol * width + "\n")

def print_update(s: str):
    print(f"{s}")
