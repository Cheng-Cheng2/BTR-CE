import argparse

def set_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", default=None, type=str, required=True)
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--eval_per_epoch", default=10, type=int,
                        help="How many times it evaluates on dev set per epoch")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--negative_label", default="no_relation", type=str)
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--train_file", default=None, type=str, help="The path of the training data.")
    parser.add_argument("--train_mode", type=str, default='random_sorted', choices=['random', 'sorted', 'random_sorted'])
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument("--eval_test", action="store_true", help="Whether to evaluate on final test set.")
    parser.add_argument("--eval_with_gold", action="store_true", help="Whether to evaluate the relation model with gold entities provided.")
    parser.add_argument("--train_batch_size", default=16, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--eval_metric", default="f1", type=str)
    parser.add_argument("--learning_rate", default=1e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed', type=int, default=0,
                        help="random seed for initialization")
    parser.add_argument("--bertadam", action="store_true", help="If bertadam, then set correct_bias = False")

    # parser.add_argument("--entity_output_dir", type=str, default=None, help="The directory of the prediction files of the entity model")
    # parser.add_argument("--entity_predictions_dev", type=str, default="ent_pred_dev.json", help="The entity prediction file of the dev set")
    # parser.add_argument("--entity_predictions_test", type=str, default="ent_pred_test.json", help="The entity prediction file of the test set")

    parser.add_argument("--prediction_file", type=str, default="predictions.json", help="The prediction filename for the relation model")

    parser.add_argument('--task', type=str, default=None, required=True, choices=['ace04', 'ace05', 'scierc', 'i2b2', 'tbd'])
    
    parser.add_argument('--marker', type=str, default=None, required=True, choices=['ner', 'ner_plus_time', 'etype','pos', 'dep', 'pos_tense', 'pos_tense_polarity', "ner_dep", "ner_pos", "ner_dep_pos", "ner_time_dep", "ner_time_dep_pos", "ner_time_pos", "bert"])

    parser.add_argument('--context_window', type=int, default=0)

    parser.add_argument('--add_new_tokens', action='store_true', 
                        help="Whether to add new tokens as marker tokens instead of using [unusedX] tokens.")
    # NOTE: ADDED
    parser.add_argument('--data_dir', type=str, default=None, required=True, 
                help="path to the preprocessed dataset")
    parser.add_argument('--use_cached', action='store_true', default=False,
                    help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--test_order_swap', action='store_true', default=False,
                help="Overwrite the cached training and evaluation sets")
    # args.test_order_swap    
    args = parser.parse_args()

    return args
# end def