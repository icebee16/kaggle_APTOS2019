import gc

from processor_factory import ProcessorFactory
from util.log_module import create_main_logger, get_main_logger
from util.command_option import get_option, get_version
from util.log_module import stop_watch


@stop_watch("main()")
def main(args):
    p_list = ProcessorFactory.make_process()

    for i, p in enumerate(p_list):
        get_main_logger(get_version()).info(f"<< {i} fold start  >>")
        p.data_preprocess()
        p.load_condition()
        best_score = p.training()
        get_main_logger(get_version()).info(__fold_log(best_score, get_version(), i))
        get_main_logger(get_version()).info(f"<< {i} fold finish >>")


def __fold_log(result, version, fold):
    text = "\n\t== [{}] {} fold best ==\n\tepoch\t\t: {}\n\ttrain_loss\t: {}\n\tvalid_loss\t: {}\n\ttrain_qwk\t: {}\n\tvalid_qwk\t: {}".format(
        str(version),
        fold,
        result["epoch"],
        result["train_loss"],
        result["valid_loss"],
        result["train_qwk"],
        result["valid_qwk"]
    )
    return text


if __name__ == "__main__":
    gc.enable()
    version = get_version()
    create_main_logger(version)
    try:
        main(get_option())
    except NotImplementedError:
        get_main_logger(version).info("Not Implemented Exception Occured.")
