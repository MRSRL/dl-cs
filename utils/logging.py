"""Common functions and params."""
import logging

# Setup of main logger
logger = logging.getLogger('dl-cs')
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(logging.BASIC_FORMAT, None))
logger.addHandler(handler)
