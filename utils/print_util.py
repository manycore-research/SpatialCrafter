
import warnings
from functools import wraps
from colorama import Fore

def cyan(x: str) -> str:
    return f"{Fore.CYAN}{x}{Fore.RESET}"

class suppress_warnings:
    """
    A context manager to suppress warnings. This can be used as a decorator as well.
    Example:
    ```
    @suppress_warnings()
    def my_function():
        pass

    with suppress_warnings():
        another_function()
    ```
    """

    def __init__(self, category=Warning):
        self.category = category

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return wrapper

    def __enter__(self):
        self.warnings_manager = warnings.catch_warnings()
        self.warnings_manager.__enter__()
        warnings.simplefilter("ignore", self.category)

    def __exit__(self, exc_type, exc_value, traceback):
        self.warnings_manager.__exit__(exc_type, exc_value, traceback)