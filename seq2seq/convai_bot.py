#!/usr/bin/env python3

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Basic example which allows local human keyboard input to talk to a trained model.
Examples
--------
.. code-block:: shell
  python examples/interactive.py -m drqa -mf "models:drqa/squad/model"
When prompted, enter something like: ``Bob is Blue.\\nWhat is Bob?``
Input is often model or task specific, but in drqa, it is always
``context '\\n' question``.
"""
from parlai.core.params import ParlaiParser
from parlai.core.agents import create_agent
from parlai.core.worlds import create_task
from convai_world import ConvAIWorld
from parlai.agents.local_human.local_human import LocalHumanAgent

import random


def setup_args(parser=None):
    if parser is None:
        parser = ParlaiParser(True, True)
    parser.add_argument('-d', '--display-examples', type='bool', default=False)
    parser.add_argument('-d', '--display-examples', type='bool', default=False)
    parser.add_argument('--display-prettify', type='bool', default=False,
                        help='Set to use a prettytable when displaying '
                             'examples with text candidates')
    parser.add_argument('--display-ignore-fields', type=str,
                        default='label_candidates,text_candidates',
                        help='Do not display these fields')
    LocalHumanAgent.add_cmdline_args(parser)
    
    ConvAIWorld.add_cmdline_args(parser)
    print("New parser",parser)
    return parser


def interactive(opt, print_parser=None):
    if print_parser is not None:
        if print_parser is True and isinstance(opt, ParlaiParser):
            print_parser = opt
        elif print_parser is False:
            print_parser = None
    if isinstance(opt, ParlaiParser):
        print('[ Deprecated Warning: interactive should be passed opt not Parser ]')
        opt = opt.parse_args()
    opt['task'] = 'parlai.agents.local_human.local_human:LocalHumanAgent'

    # Create model and assign it to the specified task
    agent = create_agent(opt, requireModelExists=True)
    #world = create_task(opt, agent)
    world = ConvAIWorld(opt, [agent])
    if print_parser:
        # Show arguments after loading model
        print_parser.opt = agent.opt
        print_parser.print_args()

    # Show some example dialogs:
    while True:
        #world.parley() 
        try:
            world.parley()
        except Exception as e:
            agent = create_agent(opt, requireModelExists=True)
            world = ConvAIWorld(opt, [agent])
            print('Exception: {}'.format(e))


if __name__ == '__main__':
    random.seed(42)
    parser = setup_args()
    parser.set_params(
        model='parlai.agents.msa_agent.seq2seq.seq2seq_v0:PerplexityEvaluatorAgent',
        #model_file='models:convai2/seq2seq/convai2_self_seq2seq_model',
        #dict_file='models:convai2/seq2seq/convai2_self_seq2seq_model.dict',
        model_file='../model/convai2_self_seq2seq_model',
        dict_file='../model/dict_convai2_self',
        dict_lower=True,
        batchsize=1,
    )
    #parser = setup_args()
    opt = parser.parse_args()
    interactive(opt)

