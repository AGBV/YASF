import logging

NUMERICS = 15
SCATTER = 25


def _log_for_numerics(self, message, *args, **kwargs):
    """
    Log a message with the NUMERICS level if it is enabled.

    Args:
        message (str): The log message.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
    """
    if self.isEnabledFor(NUMERICS):
        self._log(NUMERICS, message, args, **kwargs)


def _log_for_scatter(self, message, *args, **kwargs):
    """
    Log a message for the SCATTER level.

    Args:
        message (str): The log message.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
    """
    if self.isEnabledFor(SCATTER):
        self._log(SCATTER, message, args, **kwargs)


# def _log_to_root(message, *args, **kwargs):
#     logging.log(level_value, message, *args, **kwargs)


# Sources:
# https://stackoverflow.com/questions/7621897/python-logging-module-globally
# https://stackoverflow.com/questions/2183233/how-to-add-a-custom-loglevel-to-pythons-logging-facility
def scattering_logger(name):
    """
    Create a logger with custom log levels for scattering-related logging.

    Args:
        name (str): The name of the logger.

    Returns:
        (logging.Logger): The logger object.

    """
    levels = ["NUMERICS", "SCATTER"]
    methods = [_log_for_numerics, _log_for_scatter]
    for i, level in enumerate(levels):
        level_name = level
        level_value = globals()[level]
        method_name = level_name.lower()
        if hasattr(logging, level_name):
            # print('{} already defined in logging module'.format(level_name))
            continue
        if hasattr(logging, method_name):
            # print('{} already defined in logging module'.format(method_name))
            continue
        if hasattr(logging.getLoggerClass(), method_name):
            # print('{} already defined in logger class'.format(method_name))
            continue

        logging.addLevelName(level_value, level_name)
        setattr(logging, level_name, level_value)
        setattr(logging.getLoggerClass(), method_name, methods[i])
        # setattr(logging, method_name, _log_to_root)

    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        formatter = logging.Formatter(fmt="%(levelname)s (%(name)s): %(message)s")
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        # logger.handlers.clear()
    logger.propagate = False

    return logger
