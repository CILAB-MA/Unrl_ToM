from experiments.only_lstm import exp_main as ONLY_LSTM
from experiments.tomnet import exp_main as TOMNET
from experiments.oracle import exp_main as ORACLE
from experiments.attention import exp_main as ATTENTION

MODULES = dict(only_lstm=ONLY_LSTM, tomnet=TOMNET, oracle=ORACLE, attention=ATTENTION)
