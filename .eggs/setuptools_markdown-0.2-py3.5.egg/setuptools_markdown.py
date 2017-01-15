import inspect
import os
import logging

import pypandoc


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def long_description_markdown_filename(dist, attr, value):
    logger.debug(
        'long_description_markdown_filename: '
        'dist = %r; attr = %r; value = %r',
        dist, attr, value)
    frame = _get_code_object()
    setup_py_path = inspect.getsourcefile(frame)
    markdown_filename = os.path.join(os.path.dirname(setup_py_path), value)
    logger.debug('markdown_filename = %r', markdown_filename)
    try:
        output = pypandoc.convert(markdown_filename, 'rst')
    except OSError:
        output = open(markdown_filename).read()
    dist.metadata.long_description = output


def _get_code_object():
    frame = inspect.currentframe()

    while frame:
        code = frame.f_back.f_code
        if code.co_filename.endswith('setup.py'):
            return code
        frame = frame.f_back
