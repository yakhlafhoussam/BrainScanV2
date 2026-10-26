def warningMsg(msg: str):
    """
    Display a warning message in red text in the terminal using ANSI escape codes.

    Args:
        msg (str): The warning message to display.
    """
    print(f"\033[31m{msg}\033[0m")


def infoMsg(msg: str):
    """
    Display a info message in yellow text in the terminal using ANSI escape codes.

    Args:
        msg (str): The info message to display.
    """
    print(f"\033[33m{msg}\033[0m")


def successMsg(msg: str):
    """
    Display a success message in green text in the terminal using ANSI escape codes.

    Args:
        msg (str): The success message to display.
    """
    print(f"\033[32m{msg}\033[0m")
