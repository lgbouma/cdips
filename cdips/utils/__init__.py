from datetime import datetime

def today_YYYYMMDD():
    txt = '{}{}{}'.format(str(datetime.today().year),
                          str(datetime.today().month).zfill(2),
                          str(datetime.today().day).zfill(2))
    return txt


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")
