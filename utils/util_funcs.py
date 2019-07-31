import sys


def print_progress(count, is_total: bool = False):
    if is_total:
        # static variable in function
        print_progress.total = getattr(print_progress, 'last_hour', count)
    else:
        # Status-message.
        # Note the \r which means the line should overwrite itself.
        msg = "\r- Progress: {}/{}".format(count, print_progress.total)
        # Print it.
        sys.stdout.write(msg)
        sys.stdout.flush()

    if count == print_progress.total:
        print("")
