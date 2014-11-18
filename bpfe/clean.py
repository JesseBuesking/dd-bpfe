

import re


def clean_value(value):
    value = remove_extra_ws(value)
    value = lower(value)
    return value


def remove_extra_ws(value):
    return re.sub('\s+', ' ', value).strip()


def lower(value):
    return value.lower()
