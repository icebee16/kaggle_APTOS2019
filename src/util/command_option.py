from argparse import ArgumentParser


def get_option():
    """
    """

    argparser = ArgumentParser()
    argparser.add_argument("version",
                           type=str,
                           help="Version ID")
    return argparser.parse_args()


def get_version():
    return get_option().version
