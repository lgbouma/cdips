from datetime import datetime
import tarfile, os, subprocess

def today_YYYYMMDD():
    txt = '{}{}{}'.format(str(datetime.today().year),
                          str(datetime.today().month).zfill(2),
                          str(datetime.today().day).zfill(2))
    return txt


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def make_gztarfile_directory(output_filename, source_dir):
    # to make .tar.gz files of a directory
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))


def make_tarfile_directory(output_filename, source_dir):
    # to make .tar files of a directory
    with tarfile.open(output_filename, "w:") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))


def make_tarfile_from_fpaths(output_filename, paths_to_ball_together):
    # given many paths, put them all in the same archive

    tar = tarfile.open(output_filename, "w")

    for name in paths_to_ball_together:
        tar.add(name)

    tar.close()


def bash_grep(pattern, filename):
    """
    Grep a text file for a pattern. Literally executes grep, but just from
    python's subprocess module.

    Returns a list of matching lines if any are found, else returns None.
    """

    cmd = f'grep "{pattern}" {filename}'

    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)

    grep_stdout = process.communicate()[0]

    output = grep_stdout.decode('utf-8').split()

    if len(output) == 0:
        return None
    else:
        return output
