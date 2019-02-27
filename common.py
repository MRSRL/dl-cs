import logging

BIN_BART = 'bart'
FILENAME_PARAMS = 'params.txt'

logger = logging.getLogger('dl-cs')
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(logging.BASIC_FORMAT, None))
logger.addHandler(handler)
