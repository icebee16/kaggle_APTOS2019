import gc

from processor_factory import ProcessorFactory
from util.log_module import create_main_logger, get_main_logger
from util.command_option import get_option, get_version
from util.log_module import stop_watch


@stop_watch("main()")
def main(args):
    p_list = ProcessorFactory.make_process()

    for p in p_list:
        get_main_logger(get_version()).info(f"{p} fold start ======>>")
        p.data_preprocess()
        p.load_condition()
        p.training()
        get_main_logger(get_version()).info(f">>===== {p} fold finish")


if __name__ == "__main__":
    gc.enable()
    version = get_version()
    create_main_logger(version)
    try:
        main(get_option())
    except NotImplementedError:
        get_main_logger(version).info("Not Implemented Exception Occured.")
