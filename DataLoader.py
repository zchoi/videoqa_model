# DISTRIBUTION STATEMENT A. Approved for public release: distribution unlimited.
#
# This material is based upon work supported by the Assistant Secretary of Defense for Research and
# Engineering under Air Force Contract No. FA8721-05-C-0002 and/or FA8702-15-D-0001. Any opinions,
# findings, conclusions or recommendations expressed in this material are those of the author(s) and
# do not necessarily reflect the views of the Assistant Secretary of Defense for Research and
# Engineering.
#
# © 2017 Massachusetts Institute of Technology.
#
# MIT Proprietary, Subject to FAR52.227-11 Patent Rights - Ownership by the contractor (May 2014)
#
# The software/firmware is provided to you on an As-Is basis
#
# Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or
# 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are
# defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than
# as specifically authorized by the U.S. Government may violate any copyrights that exist in this
# work.

import numpy as np
import json
import random
import pickle
import torch
import math
import copy
import h5py
from torch.utils.data import Dataset, DataLoader
from DataBase import ClipBertBaseDataset
from src.utils.load_save import LOGGER

def invert_dict(d):
    return {v: k for k, v in d.items()}


def load_vocab(path):
    with open(path, 'r') as f:
        vocab = json.load(f)
        vocab['question_idx_to_token'] = invert_dict(vocab['question_token_to_idx'])
        vocab['answer_idx_to_token'] = invert_dict(vocab['answer_token_to_idx'])
        vocab['question_answer_idx_to_token'] = invert_dict(vocab['question_answer_token_to_idx'])
    return vocab


class ClipBertVideoQADataset(ClipBertBaseDataset):
    """ This should work for both train and test (where labels are not available).
    task_type: str, one of [action, frameqa, transition]
        where action and transition are multiple-choice QA,
            frameqa is opened QA similar to VQA.
    datalist: list(tuples)  each tuple is (img_id, list(dicts)),
        each dict
    tokenizer:
    max_img_size: int,
    max_txt_len: int, max text sequence length, including special tokens.
    itm_neg_prob: float [0, 1] set to 0 will disable itm.
    return_label: bool, whether return label in __getitem__
    random_sample_clips:
    """
    open_ended_qa_names = ["frameqa", "msrvtt_qa"]

    def __init__(self, task_type, datalist, tokenizer, img_lmdb_dir,
                 fps=3, num_frm=3, frm_sampling_strategy="rand",
                 max_img_size=1000, max_txt_len=20, ans2label=None,
                 ensemble_n_clips=1, return_label=False, is_train=False, random_sample_clips=False):
        super(ClipBertVideoQADataset, self).__init__(
            datalist, tokenizer, img_lmdb_dir,
            fps=fps, num_frm=num_frm,
            frm_sampling_strategy=frm_sampling_strategy,
            max_img_size=max_img_size, max_txt_len=max_txt_len)
        self.ensemble_n_clips = ensemble_n_clips
        self.return_label = return_label
        self.is_train = is_train
        self.random_sample_clips = random_sample_clips

    def __len__(self):
        return len(self.datalist)

    def _load_video_multi_clips_uniform(self, vid_id):
        """take multiple clips at fixed position"""
        vid_frm_array_list = []
        prev_clip = None
        video_max_pts = None
        for clip_idx in range(self.ensemble_n_clips):
            curr_clip, video_max_pts = self._load_video(
                vid_id, num_clips=self.ensemble_n_clips,
                clip_idx=clip_idx, safeguard_duration=True,
                video_max_pts=video_max_pts)
            if curr_clip is None:
                print("Copying prev clips as the current one is None")
                curr_clip = copy.deepcopy(prev_clip)
            else:
                prev_clip = curr_clip
            vid_frm_array_list.append(curr_clip)
        return torch.cat(vid_frm_array_list, dim=0)

    def _load_video_multi_clips_random(self, vid_id):
        """take multiple clips at random position"""
        vid_frm_array_list = []
        prev_clip = None
        for clip_idx in range(self.ensemble_n_clips):
            curr_clip, _ = self._load_video(
                vid_id, num_clips=None, clip_idx=None,
                safeguard_duration=False)
            if curr_clip is None:
                print("Copying prev clips as the current one is None")
                curr_clip = copy.deepcopy(prev_clip)
            else:
                prev_clip = curr_clip
            vid_frm_array_list.append(curr_clip)
        return None if any([e is None for e in vid_frm_array_list]) else torch.cat(vid_frm_array_list, dim=0)

    def return_video_sample(self, vid_id):
        # skip error videos:
        num_retries = 1
        vid_id = 'video' + str(vid_id)
        for _ in range(num_retries):
            if self.ensemble_n_clips > 1:
                # tensor (T*ensemble_n_clips, C, H, W), reshape as (T, ensemble_n_clips, C, H, W)
                if self.random_sample_clips:
                    vid_frm_array = self._load_video_multi_clips_random(vid_id)
                else:
                    vid_frm_array = self._load_video_multi_clips_uniform(vid_id)
            else:
                if self.random_sample_clips:
                    vid_frm_array, _ = self._load_video(vid_id)  # tensor (T, C, H, W)
                else:
                    vid_frm_array, _ = self._load_video(vid_id, num_clips=1, clip_idx=0)  # tensor (T, C, H, W)
            # vid_frm_array = torch.zeros_like(vid_frm_array)
            # Select a random video if the current video was not able to access.
            if vid_frm_array is None:
                LOGGER.info(f"Failed to load examples with video: {vid_id}. "
                            f"Will randomly sample an example as a replacement.")
                index = random.randint(0, len(self) - 1)
                continue

            return vid_frm_array
        else:
            raise RuntimeError(f"Failed to fetch video after {num_retries} retries.")


class VideoQADataset(Dataset):

    def __init__(self, answers, ans_candidates, ans_candidates_len, questions, questions_len, video_ids, q_ids,
                 app_feature_h5, app_feat_id_to_index, motion_feature_h5, motion_feat_id_to_index, video_lmdb_path):
        # convert data to tensor
        self.all_answers = answers
        self.all_questions = torch.LongTensor(np.asarray(questions))
        self.all_questions_len = torch.LongTensor(np.asarray(questions_len))
        self.all_video_ids = torch.LongTensor(np.asarray(video_ids))
        self.all_q_ids = q_ids
        self.app_feature_h5 = app_feature_h5
        self.motion_feature_h5 = motion_feature_h5
        self.app_feat_id_to_index = app_feat_id_to_index
        self.motion_feat_id_to_index = motion_feat_id_to_index
        self.video_lmdb_path = video_lmdb_path
        self.ClipBertVideoQADataset = ClipBertVideoQADataset(
            task_type=None,
            datalist=None,
            tokenizer=None,
            img_lmdb_dir=self.video_lmdb_path,
            ans2label=None,
            max_img_size=224,
            max_txt_len=None,
            fps=3,
            num_frm=8,
            frm_sampling_strategy='uniform',
            ensemble_n_clips=4
        )
        if not np.any(ans_candidates):
            self.question_type = 'openended'
        else:
            self.question_type = 'mulchoices'
            self.all_ans_candidates = torch.LongTensor(np.asarray(ans_candidates))
            self.all_ans_candidates_len = torch.LongTensor(np.asarray(ans_candidates_len))

    def __getitem__(self, index):
        answer = self.all_answers[index] if self.all_answers is not None else None
        ans_candidates = torch.zeros(5)
        ans_candidates_len = torch.zeros(5)
        if self.question_type == 'mulchoices':
            ans_candidates = self.all_ans_candidates[index]
            ans_candidates_len = self.all_ans_candidates_len[index]
        question = self.all_questions[index]
        question_len = self.all_questions_len[index]
        video_idx = self.all_video_ids[index].item()
        question_idx = self.all_q_ids[index]
        # app_index = self.app_feat_id_to_index[str(video_idx)]
        # motion_index = self.motion_feat_id_to_index[str(video_idx)]
        # with h5py.File(self.app_feature_h5, 'r') as f_app:
        #     appearance_feat = f_app['resnet_features'][app_index]  # (8, 16, 2048)
        # with h5py.File(self.motion_feature_h5, 'r') as f_motion:
        #     motion_feat = f_motion['resnext_features'][motion_index]  # (8, 2048)
        # appearance_feat = torch.from_numpy(appearance_feat)
        # motion_feat = torch.from_numpy(motion_feat)
        appearance_feat, motion_feat = 1, 1
        raw_video = self.ClipBertVideoQADataset.return_video_sample(video_idx)  # torch.Size([128, 3, 224, 224])

        assert raw_video is not None

        return (
            video_idx, question_idx, answer, ans_candidates, ans_candidates_len, appearance_feat, motion_feat, question,
            question_len, raw_video)

    def __len__(self):
        return len(self.all_questions)


class VideoQADataLoader(DataLoader):

    def __init__(self, **kwargs):
        vocab_json_path = str(kwargs.pop('vocab_json'))
        print('loading vocab from %s' % (vocab_json_path))
        vocab = load_vocab(vocab_json_path)

        question_pt_path = str(kwargs.pop('question_pt'))
        video_lmdb_path = str(kwargs.pop('video_lmdb'))
        print('loading questions from %s' % (question_pt_path))
        question_type = kwargs.pop('question_type')
        with open(question_pt_path, 'rb') as f:
            obj = pickle.load(f)
            questions = obj['questions']
            questions_len = obj['questions_len']
            video_ids = obj['video_ids']
            q_ids = obj['question_id']
            answers = obj['answers']
            glove_matrix = obj['glove']
            ans_candidates = np.zeros(5)
            ans_candidates_len = np.zeros(5)
            if question_type in ['action', 'transition']:
                ans_candidates = obj['ans_candidates']
                ans_candidates_len = obj['ans_candidates_len']

        if 'train_num' in kwargs:
            trained_num = kwargs.pop('train_num')
            if trained_num > 0:
                questions = questions[:trained_num]
                questions_len = questions_len[:trained_num]
                video_ids = video_ids[:trained_num]
                q_ids = q_ids[:trained_num]
                answers = answers[:trained_num]
                if question_type in ['action', 'transition']:
                    ans_candidates = ans_candidates[:trained_num]
                    ans_candidates_len = ans_candidates_len[:trained_num]
        if 'val_num' in kwargs:
            val_num = kwargs.pop('val_num')
            if val_num > 0:
                questions = questions[:val_num]
                questions_len = questions_len[:val_num]
                video_ids = video_ids[:val_num]
                q_ids = q_ids[:val_num]
                answers = answers[:val_num]
                if question_type in ['action', 'transition']:
                    ans_candidates = ans_candidates[:val_num]
                    ans_candidates_len = ans_candidates_len[:val_num]
        if 'test_num' in kwargs:
            test_num = kwargs.pop('test_num')
            if test_num > 0:
                questions = questions[:test_num]
                questions_len = questions_len[:test_num]
                video_ids = video_ids[:test_num]
                q_ids = q_ids[:test_num]
                answers = answers[:test_num]
                if question_type in ['action', 'transition']:
                    ans_candidates = ans_candidates[:test_num]
                    ans_candidates_len = ans_candidates_len[:test_num]

        # print('loading appearance feature from %s' % (kwargs['appearance_feat']))
        # with h5py.File(kwargs['appearance_feat'], 'r') as app_features_file:
        #     app_video_ids = app_features_file['ids'][()]
        # app_feat_id_to_index = {str(id): i for i, id in enumerate(app_video_ids)}
        # print('loading motion feature from %s' % (kwargs['motion_feat']))
        # with h5py.File(kwargs['motion_feat'], 'r') as motion_features_file:
        #     motion_video_ids = motion_features_file['ids'][()]
        # motion_feat_id_to_index = {str(id): i for i, id in enumerate(motion_video_ids)}
        app_feat_id_to_index, motion_feat_id_to_index = None, None

        self.app_feature_h5 = kwargs.pop('appearance_feat')
        self.motion_feature_h5 = kwargs.pop('motion_feat')
        self.dataset = VideoQADataset(answers, ans_candidates, ans_candidates_len, questions, questions_len,
                                      video_ids, q_ids,
                                      self.app_feature_h5, app_feat_id_to_index, self.motion_feature_h5,
                                      motion_feat_id_to_index, video_lmdb_path)

        self.vocab = vocab
        self.batch_size = kwargs['batch_size']
        self.glove_matrix = glove_matrix

        super().__init__(self.dataset, **kwargs)

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)
