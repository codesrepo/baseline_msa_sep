# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Evaluate pre-trained model trained for hits@1 metric.
This seq2seq model was trained on convai2:self.
"""

from parlai.core.build_data import download_models
from projects.convai2.eval_hits import setup_args, eval_hits


if __name__ == '__main__':
    parser = setup_args()
    parser.set_params(
        model='parlai.agents.msa_agent.seq2seq.seq2seq_v0:Seq2seqAgent',
        model_file='../model/convai2_self_seq2seq_model',
        dict_file='../model/dict_convai2_self',
        dict_lower=True,
        rank_candidates=True,
        batchsize=32,
    )
    opt = parser.parse_args(print_args=False)
    if (opt.get('model_file', '')
            .find('convai2/seq2seq/convai2_self_seq2seq_model') != -1) or 1==1:
        opt['model_type'] = 'seq2seq'
    eval_hits(opt, print_parser=parser)
