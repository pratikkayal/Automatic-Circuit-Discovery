# Based on ACDC_Main_Demo notebook

import os
import argparse
import torch

# our own (ACDC) imports
from acdc.acdc_utils import (
    reset_network,
)  # these introduce several important classes !!!
from acdc.TLACDCExperiment import TLACDCExperiment
from acdc.multilingual.utils import (
    get_all_multilingual_things,
)
from acdc.acdc_graphics import (
    show,
)


def parse_options():
    parser = argparse.ArgumentParser(description="Used to launch ACDC runs. Only task and threshold are required")

    task_choices = ['ioi', 'docstring', 'induction', 'tracr-reverse', 'tracr-proportion', 'greaterthan']
    parser.add_argument('--task', type=str, required=True, choices=task_choices,
                        help=f'Choose a task from the available options: {task_choices}')
    parser.add_argument('--threshold', type=float, required=True, help='Value for THRESHOLD')
    parser.add_argument('--zero-ablation', action='store_true', help='Use zero ablation')
    parser.add_argument('--indices-mode', type=str, default="normal")
    parser.add_argument('--names-mode', type=str, default="normal")
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--reset-network', type=int, default=0,
                        help="Whether to reset the network we're operating on before running interp on it")
    parser.add_argument('--metric', type=str, default="kl_div", help="Which metric to use for the experiment")
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument("--max-num-epochs", type=int, default=100_000)
    parser.add_argument('--single-step', action='store_true', help='Use single step, mostly for testing')

    args = parser.parse_args(
        [line.strip() for line in r"""--task=multilingual\
    --zero-ablation\
    --threshold=0.71\
    --indices-mode=reverse\
    --first-cache-cpu=False\
    --second-cache-cpu=False\
    --max-num-epochs=100000""".split("\\\n")]
    )

    return args


if __name__ == '__main__':
    # initial settings
    torch.autograd.set_grad_enabled(False)

    # plotly.io.renderers.default = "colab"  # added by Arthur so running as a .py notebook with #%% generates .ipynb notebooks that display in colab
    # disable this option when developing rather than generating notebook outputs

    if not os.path.exists("ims/"):  # make images folder
        os.mkdir("ims/")

    args = parse_options()
    torch.manual_seed(args.seed)

    TASK = args.task
    ONLINE_CACHE_CPU = True
    CORRUPTED_CACHE_CPU = True
    THRESHOLD = args.threshold  # only used if >= 0.0
    ZERO_ABLATION = True if args.zero_ablation else False

    INDICES_MODE = args.indices_mode
    NAMES_MODE = args.names_mode
    DEVICE = args.device

    num_examples = 100
    things = get_all_multilingual_things(
        num_examples=num_examples, device=DEVICE, metric_name=args.metric
    )

    tl_model = things.tl_model  # transformerlens model
    toks_int_values = things.validation_data  # clean data x_i
    toks_int_values_other = things.validation_patch_data  # corrupted data x_i'
    validation_metric = things.validation_metric  # metric we use (e.g KL divergence)

    if args.reset_network:
        reset_network(TASK, DEVICE, tl_model)

    tl_model.reset_hooks()

    custom_args = {
        'zero_ablation': ZERO_ABLATION,
        'verbose': True,
        'indices_mode': INDICES_MODE,
        'names_mode': NAMES_MODE,
    }

    exp = TLACDCExperiment(
        tl_model,
        toks_int_values,
        toks_int_values_other,
        THRESHOLD,
        validation_metric,
        **custom_args
    )

    for i in range(args.max_num_epochs):
        exp.step(testing=False)

        show(
            exp.corr,
            f"ims/img_new_{i + 1}.png",
            show_full_index=False,
        )

        print(i, "-" * 50)
        print(exp.count_no_edges())

        if i == 0:
            exp.save_edges("edges.pkl")

        if exp.current_node is None or args.single_step:
            break

    exp.save_edges("another_final_edges.pkl")
    exp.save_subgraph(
        return_it=True,
    )



