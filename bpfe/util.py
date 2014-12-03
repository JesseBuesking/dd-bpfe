

def isfloat(value):
    # noinspection PyBroadException
    try:
        float(value)
        return True
    except:
        return False
