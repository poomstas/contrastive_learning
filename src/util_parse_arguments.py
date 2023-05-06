# %%
''' Parse arguments from command line and from W&B sweep '''

# %%
def parse_arguments(parser):
    parser.add_argument('--NUM_EPOCHS',                     type=int,           default=10000,              help='')
    parser.add_argument('--BATCH_SIZE',                     type=int,           default=128,                help='')
    parser.add_argument('--LR',                             type=float,         default=0.0001,             help='')
    parser.add_argument('--DATAFRAME_MULTIPLIER',           type=int,           default=1,                  help='Used to increase the size of the dataset by replicating the dataframe. This enables larger batch size')
    parser.add_argument('--FIRST_TCN_MODEL',                type=str,           default='/home/brian/github/visemenet/model_checkpoint/20230310_080832_brian-a100/epoch=185-val_loss=0.00142.ckpt',           help='Full path reference to the model .cpkt file')
    parser.add_argument('--TCN_NUM_CHANNELS_ACTIVATION',    type=list,          default=[512, 256, 256],    help='Number of channels for each layer in the TCN model - Activation')
    parser.add_argument('--TCN_NUM_CHANNELS_RIGPARAMS',     type=list,          default=[512, 256, 256],    help='Number of channels for each layer in the TCN model - Rig Params')
    parser.add_argument('--TCN_NUM_CHANNELS_JALI',          type=list,          default=[512, 256, 256],    help='Number of channels for each layer in the TCN model - JALI')
    parser.add_argument('--SINGLE_OUT_SPEAKER',             type=str,           default='F1',               help='')
    parser.add_argument('--FEATURE_TIMESTEPS',              type=int,           default=200,                help='Number of audio feature timesteps to use for one instance in training')
    parser.add_argument('--SHIFT_JALI_BY',                  type=int,           default=4,                  help='')
    parser.add_argument('--LOSS_WA',                        type=float,         default=0.1,                help='')
    parser.add_argument('--LOSS_WV',                        type=float,         default=0.2,                help='')
    parser.add_argument('--LOSS_WJ',                        type=float,         default=0.2,                help='')
    parser.add_argument('--LOSS_WV_PRIME',                  type=float,         default=0.15,               help='')
    parser.add_argument('--LOSS_WJ_PRIME',                  type=float,         default=0.15,               help='')
    parser.add_argument('--ONECYCLELR_MAX_LR',              type=float,         default=0.000005,           help='')
    parser.add_argument('--ONECYCLELR_PCT_START',           type=float,         default=0.4,                help='')
    parser.add_argument('--ONECYCLELR_DIV_FACTOR',          type=int,           default=25,                 help='')
    parser.add_argument('--ONECYCLELR_FINAL_DIV_FACTOR',    type=float,         default=0.6,                help='')
    parser.add_argument('--NOTE',                           type=str,           default='',                 help='')

    args = parser.parse_args()

    return args


# %%
def get_dict_from_args(args):
    params = [item for item in dir(args) if not item.startswith('_')]
    config = {}
    for item in params:
        config[item] = getattr(args, item)

    return config


# %%
def parse_list_from_wandb(wandb_input):
    '''
        From W&B sweep, 
        - 0 123 456 678 turns in to below:
        ['0', ' ', '1', '2', '3', ' ', '4', '5', '6', ' ', '6', '7', '8'] 
        Would like to retrieve the original form: [0, 123, 456, 678]
    '''
    if type(wandb_input[0]) != str:
        return wandb_input
    else:
        strings = ''.join(wandb_input).split()
        parsed = [int(item) for item in strings]
    return parsed
