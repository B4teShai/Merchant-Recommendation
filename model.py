# -*- coding: utf-8 -*-
"""
PyTorch Recommender aligned with SelfGNN (SIGIR 2024): multi-graph GNN (concat+project),
GRU over intervals, multi-head attention, sequence encoder, personalized SSL.
Uses Params, DataHandler, Utils (NNLayers, attention, compat).
"""

from __future__ import annotations

import os
import pickle
from random import randint

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from Params import args
from DataHandler import negSamp, transpose, transToLsts
from Utils.NNLayers import FC, get_activation, regularize
from Utils.attention import AdditiveAttention, MultiHeadSelfAttention
import Utils.TimeLogger as time_logger
from Utils.TimeLogger import log


def _segment_sum_pad(
    src_embeds: torch.Tensor,
    tgt_nodes: torch.Tensor,
    pad_size: int,
    device: torch.device,
    min_rows: int = 0,
) -> torch.Tensor:
    """Aggregate src_embeds by tgt_nodes (segment_sum), then pad rows.
    Output has at least min_rows rows so that lat[0..min_rows-1] is valid."""
    E, dim = src_embeds.shape
    max_tgt = tgt_nodes.max().item()
    out_size = max(max_tgt + 1, min_rows) + pad_size
    out = torch.zeros(out_size, dim, dtype=src_embeds.dtype, device=device)
    tgt_exp = tgt_nodes.unsqueeze(1).expand(-1, dim)
    out.scatter_add_(0, tgt_exp, src_embeds)
    return out


class RecommenderNet(nn.Module):
    """Core network: embeddings, GNN, RNN, attention, sequence encoder, prediction, SSL."""

    def __init__(
        self,
        num_user: int,
        num_item: int,
        max_time: int,
        sub_adj_indices: list[tuple[torch.Tensor, torch.Tensor, list]],
        sub_tp_adj_indices: list[tuple[torch.Tensor, torch.Tensor, list]],
        device: torch.device,
    ):
        super().__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.max_time = max_time
        self.device = device
        self.sub_adj = sub_adj_indices  # list of (idx, val, shape) per graph
        self.sub_tp_adj = sub_tp_adj_indices

        self.act_fn = get_activation("leakyRelu", args.leaky)
        if isinstance(self.act_fn, nn.Module):
            self.add_module("act_fn", self.act_fn)

        # Embeddings: [graphNum, user, latdim], [graphNum, item, latdim], etc.
        self.u_embed = nn.Parameter(
            torch.empty(args.graphNum, num_user, args.latdim, device=device).uniform_(
                -0.1, 0.1
            )
        )
        self.i_embed = nn.Parameter(
            torch.empty(args.graphNum, num_item, args.latdim, device=device).uniform_(
                -0.1, 0.1
            )
        )
        self.pos_embed = nn.Parameter(
            torch.empty(args.pos_length, args.latdim, device=device).uniform_(
                -0.1, 0.1
            )
        )
        self.time_embed = nn.Parameter(
            torch.empty(max_time + 1, args.latdim, device=device).uniform_(-0.1, 0.1)
        )

        # Time embed projection (used in messagePropagate but TF comment leaves it out; keep for compatibility)
        self.time_fc = FC(args.latdim, args.latdim, activation=None)
        self.time_fc.to(device)

        # Short-term GNN: concat layer outputs (Eq 4) then project to d (SelfGNN paper)
        num_layer_outputs = args.gnn_layer + 1  # initial + gnn_layer propagated
        self.short_user_proj = nn.Linear(
            num_layer_outputs * args.latdim, args.latdim, device=device
        )
        self.short_item_proj = nn.Linear(
            num_layer_outputs * args.latdim, args.latdim, device=device
        )

        # RNN over graph dimension (paper: GRU, Eq 5)
        self.rnn = nn.GRU(
            args.latdim,
            args.latdim,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
        ).to(device)

        # Multi-head self-attention for user/item over graphs
        self.multihead_user = MultiHeadSelfAttention(
            args.latdim, args.num_attention_heads
        ).to(device)
        self.multihead_item = MultiHeadSelfAttention(
            args.latdim, args.num_attention_heads
        ).to(device)

        # Sequence encoder: stacked self-attention
        self.seq_att_layers = nn.ModuleList(
            [
                MultiHeadSelfAttention(args.latdim, args.num_attention_heads).to(
                    device
                )
                for _ in range(args.att_layer)
            ]
        )
        self.layer_norm = nn.LayerNorm(args.latdim).to(device)

        # User-weight MLP for SSL (paper Eq 14: input ē + e_t + ē⊙e_t, dim d)
        self.meta_fc1 = FC(
            args.latdim,
            args.ssldim,
            use_bias=True,
            activation="leakyRelu",
            leaky=args.leaky,
        ).to(device)
        self.meta_fc2 = FC(
            args.ssldim, 1, use_bias=True, activation="sigmoid", leaky=args.leaky
        ).to(device)

    def _message_propagate(
        self,
        srclats: torch.Tensor,
        adj_idx: torch.Tensor,
        adj_val: torch.Tensor,
        shape: list,
        node_type: str,
        keep_rate: float = 1.0,
    ) -> torch.Tensor:
        # adj: COO indices [E, 2] (row, col) = (tgt, src); for user-item adj row=user, col=item
        # TF: srcNodes = indices[:, 1], tgtNodes = indices[:, 0]
        E = adj_idx.shape[0]
        if E == 0:
            n_nodes = args.user if node_type == "user" else args.item
            return torch.zeros(
                n_nodes, args.latdim, dtype=srclats.dtype, device=self.device
            )
        tgt_nodes = adj_idx[:, 0]
        src_nodes = adj_idx[:, 1]
        # Clamp to embedding size so gather never gets index out of bounds (e.g. 1-based data)
        n_src = int(srclats.shape[0])
        n_tgt = int(args.user if node_type == "user" else args.item)
        src_nodes = src_nodes.clamp(0, n_src - 1).long()
        tgt_nodes = tgt_nodes.clamp(0, n_tgt - 1).long()
        # Use index_select to avoid any advanced-indexing OOB path
        src_embeds = srclats.index_select(0, src_nodes)
        # Edge dropout: mask out some edges (scale by 1/keep_rate to keep expectation)
        if keep_rate < 1.0 and self.training:
            mask = (torch.rand(E, device=srclats.device) < keep_rate).to(srclats.dtype) / (
                keep_rate + 1e-8
            )
            src_embeds = src_embeds * mask.unsqueeze(1)
        pad_size = 100
        lat = _segment_sum_pad(
            src_embeds, tgt_nodes, pad_size, self.device, min_rows=n_tgt
        )
        if node_type == "user":
            users = torch.arange(args.user, device=self.device, dtype=torch.long)
            out = lat[users]
        else:
            items = torch.arange(args.item, device=self.device, dtype=torch.long)
            out = lat[items]
        return self.act_fn(out)

    def forward(
        self,
        uids: torch.Tensor,
        iids: torch.Tensor,
        sequence: torch.Tensor,
        mask: torch.Tensor,
        u_locs_seq: torch.Tensor,
        suids_list: list[torch.Tensor],
        siids_list: list[torch.Tensor],
        keep_rate: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        """
        Returns: preds, sslloss, preds_one (list per graph for SSL).
        """
        user_vectors = []
        item_vectors = []
        for k in range(args.graphNum):
            embs0 = [self.u_embed[k]]
            embs1 = [self.i_embed[k]]
            idx, val, shape = self.sub_adj[k]
            idx_tp, val_tp, shape_tp = self.sub_tp_adj[k]
            for _ in range(args.gnn_layer):
                a_emb0 = self._message_propagate(
                    embs1[-1], idx, val, shape, "user", keep_rate
                )
                a_emb1 = self._message_propagate(
                    embs0[-1], idx_tp, val_tp, shape_tp, "item", keep_rate
                )
                embs0.append(a_emb0 + embs0[-1])
                embs1.append(a_emb1 + embs1[-1])
            # Paper Eq 4: concat layer outputs then project to d (SelfGNN)
            user_concat = torch.cat(embs0, dim=-1)
            item_concat = torch.cat(embs1, dim=-1)
            user = self.short_user_proj(user_concat)
            item = self.short_item_proj(item_concat)
            user_vectors.append(user)
            item_vectors.append(item)

        # [graphNum, user, latdim] -> [user, graphNum, latdim]
        user_vector = torch.stack(user_vectors, dim=0).transpose(0, 1)
        item_vector = torch.stack(item_vectors, dim=0).transpose(0, 1)

        # RNN over graph dimension
        user_rnn, _ = self.rnn(user_vector)
        item_rnn, _ = self.rnn(item_vector)
        user_vector = user_rnn
        item_vector = item_rnn

        # Multi-head self-attention then sum over intervals (paper Eq 7)
        user_ln = self.layer_norm(user_vector)
        item_ln = self.layer_norm(item_vector)
        multihead_user = self.multihead_user.attention(user_ln)
        multihead_item = self.multihead_item.attention(item_ln)
        final_user_vector = multihead_user.sum(dim=1)
        final_item_vector = multihead_item.sum(dim=1)
        i_embed_att = final_item_vector

        # Sequence encoder: mask @ item_embed(sequence) + mask @ pos_embed(pos)
        pos = torch.arange(
            args.pos_length, device=self.device, dtype=torch.long
        ).unsqueeze(0).expand(sequence.size(0), -1)
        seq_item = torch.nn.functional.embedding(sequence, i_embed_att)
        seq_pos = torch.nn.functional.embedding(pos, self.pos_embed)
        sequence_batch = self.layer_norm(
            mask.unsqueeze(2) * seq_item + mask.unsqueeze(2) * seq_pos
        )
        att_layer = sequence_batch
        for mha in self.seq_att_layers:
            att_out = mha.attention(self.layer_norm(att_layer))
            att_layer = self.act_fn(att_out) + att_layer
        att_user = att_layer.sum(dim=1)

        # Prediction (paper Eq 10): ê^(u) = ē^(u) + tilde{e}^(u), Â = ê^(u)^T · ē^(v)
        pck_ulat = final_user_vector[uids]  # interval-level user ē^(u)
        att_user_at_loc = att_user[u_locs_seq]  # instance-level user tilde{e}^(u)
        user_agg = pck_ulat + att_user_at_loc  # ê^(u)
        pck_ilat = final_item_vector[iids]  # ē^(v)
        preds = (user_agg * pck_ilat).sum(dim=-1)

        # User weights for SSL (paper Eq 14: Gamma = sigma((ē + e_t + ē⊙e_t) W1 + b1), w = sigm(Gamma W2 + b2))
        user_weight_list = []
        for i in range(args.graphNum):
            e_t = user_vectors[i]
            meta1 = final_user_vector + e_t + final_user_vector * e_t
            meta2 = self.meta_fc1(meta1)
            w = self.meta_fc2(meta2).squeeze(-1)
            user_weight_list.append(w)
        user_weight = torch.stack(user_weight_list, dim=0)

        # SSL loss
        sslloss = torch.tensor(0.0, device=self.device)
        preds_one_list = []
        for i in range(args.graphNum):
            samp_num = suids_list[i].size(0) // 2
            pck_ulat = final_user_vector[suids_list[i]]
            pck_ilat = final_item_vector[siids_list[i]]
            pck_uweight = user_weight[i][suids_list[i]]
            pck_ilat_att = i_embed_att[siids_list[i]]
            S_final = (self.act_fn(pck_ulat * pck_ilat)).sum(dim=-1)
            pos_pred_final = S_final[:samp_num].detach()
            neg_pred_final = S_final[samp_num:].detach()
            pos_weight = pck_uweight[:samp_num]
            neg_weight = pck_uweight[samp_num:]
            S_final = pos_weight * pos_pred_final - neg_weight * neg_pred_final
            pck_ulat_g = user_vectors[i][suids_list[i]]
            pck_ilat_g = item_vectors[i][siids_list[i]]
            preds_one = (self.act_fn(pck_ulat_g * pck_ilat_g)).sum(dim=-1)
            pos_pred = preds_one[:samp_num]
            neg_pred = preds_one[samp_num:]
            sslloss = sslloss + torch.clamp(
                1.0 - S_final * (pos_pred - neg_pred), min=0.0
            ).sum()
            preds_one_list.append(preds_one)
        return preds, sslloss, preds_one_list


class Recommender:
    """Recommender: prepareModel, train/test loops, save/load. Uses handler and device."""

    def __init__(self, handler, device: torch.device):
        self.handler = handler
        self.device = device
        self.metrics = {
            "TrainLoss": [],
            "TrainpreLoss": [],
            "TrainHR": [],
            "TrainNDCG": [],
            "TestLoss": [],
            "TestpreLoss": [],
            "TestHR": [],
            "TestNDCG": [],
        }
        for k in args.topk:
            self.metrics["TrainHR@%d" % k] = []
            self.metrics["TrainNDCG@%d" % k] = []
            self.metrics["TestHR@%d" % k] = []
            self.metrics["TestNDCG@%d" % k] = []
        print("USER", args.user, "ITEM", args.item)

    def _results_base_dir(self) -> str:
        """Return results/<dataset_name>/ and ensure it exists."""
        base = os.path.join("results", args.data)
        os.makedirs(base, exist_ok=True)
        return base

    def make_print(self, name: str, ep: int, reses: dict, save: bool) -> str:
        ret = "Epoch %d/%d, %s: " % (ep, args.epoch, name)
        for metric, val in reses.items():
            ret += "%s = %.4f, " % (metric, val)
            key = name + metric
            if save and key in self.metrics:
                self.metrics[key].append(val)
        ret = ret[:-2] + "  "
        return ret

    def _build_sub_adj_tensors(self):
        """Build list of (indices, values, shape) for sub_adj and sub_tp_adj on device."""
        sub_adj = []
        sub_tp_adj = []
        for i in range(args.graphNum):
            seqadj = self.handler.subMat[i]
            idx, data, shape = transToLsts(seqadj, norm=True)
            # Force indices in-bounds (guard against 1-based or corrupted data)
            nrow, ncol = shape[0], shape[1]
            idx = np.clip(idx, [0, 0], [nrow - 1, ncol - 1]).astype(np.int64)
            idx_t = torch.from_numpy(idx).long().to(self.device)
            val_t = torch.from_numpy(data.astype(np.float32)).to(self.device)
            sub_adj.append((idx_t, val_t, shape))
            seqadj_tp = transpose(seqadj)
            idx2, data2, shape2 = transToLsts(seqadj_tp, norm=True)
            nrow2, ncol2 = shape2[0], shape2[1]
            idx2 = np.clip(idx2, [0, 0], [nrow2 - 1, ncol2 - 1]).astype(np.int64)
            idx_t2 = torch.from_numpy(idx2).long().to(self.device)
            val_t2 = torch.from_numpy(data2.astype(np.float32)).to(self.device)
            sub_tp_adj.append((idx_t2, val_t2, shape2))
        return sub_adj, sub_tp_adj

    def prepare_model(self) -> None:
        assert len(self.handler.subMat) == args.graphNum, (
            "subMat length (%d) must equal args.graphNum (%d). "
            "Build data with T=%d time intervals."
        ) % (len(self.handler.subMat), args.graphNum, args.graphNum)
        sub_adj, sub_tp_adj = self._build_sub_adj_tensors()
        self.model = RecommenderNet(
            args.user,
            args.item,
            self.handler.maxTime,
            sub_adj,
            sub_tp_adj,
            self.device,
        )
        self.model.to(self.device)
        self.act_fn = get_activation("leakyRelu", args.leaky)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=args.decay_step, gamma=args.decay
        )

    def _pre_loss(self, preds: torch.Tensor, samp_num: int) -> torch.Tensor:
        pos_pred = preds[:samp_num]
        neg_pred = preds[samp_num:]
        return torch.clamp(1.0 - (pos_pred - neg_pred), min=0.0).mean()

    def run(self) -> None:
        self.prepare_model()
        log("Model Prepared")
        if args.load_model is not None:
            self.load_model()
            stloc = len(self.metrics["TrainLoss"]) * args.tstEpoch - (args.tstEpoch - 1)
        else:
            stloc = 0
            for p in self.model.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
            log("Variables Inited")
        maxndcg = 0.0
        maxres = {}
        maxepoch = 0
        for ep in range(stloc, args.epoch):
            test = (ep % args.tstEpoch == 0)
            reses = self.train_epoch()
            log(self.make_print("Train", ep, reses, test))
            if test:
                reses = self.test_epoch()
                log(self.make_print("Test", ep, reses, test))
            primary_ndcg = reses.get("NDCG@%d" % args.topk[0], reses.get("NDCG", 0))
            if ep % args.tstEpoch == 0 and primary_ndcg > maxndcg:
                self.save_history()
                maxndcg = primary_ndcg
                maxres = reses
                maxepoch = ep
            print()
        reses = self.test_epoch()
        log(self.make_print("Test", args.epoch, reses, True))
        log(self.make_print("max", maxepoch, maxres, True))
        # Metrics result summary
        log("--- Metrics result ---")
        for k in args.topk:
            log(
                "Best epoch: %d | Test HR@%d = %.4f, Test NDCG@%d = %.4f"
                % (maxepoch, k, maxres.get("HR@%d" % k, 0), k, maxres.get("NDCG@%d" % k, 0))
            )
        if self.metrics["TestHR"]:
            log("Test HR  (all): %s" % (", ".join("%.4f" % v for v in self.metrics["TestHR"])))
        if self.metrics["TestNDCG"]:
            log("Test NDCG (all): %s" % (", ".join("%.4f" % v for v in self.metrics["TestNDCG"])))
        for k in args.topk:
            key_hr, key_ndcg = "TestHR@%d" % k, "TestNDCG@%d" % k
            if self.metrics.get(key_hr):
                log("Test HR@%d  (all): %s" % (k, ", ".join("%.4f" % v for v in self.metrics[key_hr])))
            if self.metrics.get(key_ndcg):
                log("Test NDCG@%d (all): %s" % (k, ", ".join("%.4f" % v for v in self.metrics[key_ndcg])))
        log("----------------------")
        self._save_metrics_file(maxepoch, maxres)
        self._save_metrics_plots()

    def _save_metrics_file(self, maxepoch: int, maxres: dict) -> None:
        """Write metrics summary to results/<data>/<save_path>_metrics.txt."""
        base_dir = self._results_base_dir()
        path = os.path.join(base_dir, args.save_path + "_metrics.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("dataset: %s\n" % args.data)
            f.write("save_path: %s\n" % args.save_path)
            f.write("topk: %s\n" % args.topk)
            f.write("--- Metrics result ---\n")
            for k in args.topk:
                f.write(
                    "Best epoch: %d | Test HR@%d = %.4f, Test NDCG@%d = %.4f\n"
                    % (maxepoch, k, maxres.get("HR@%d" % k, 0), k, maxres.get("NDCG@%d" % k, 0))
                )
            for k in args.topk:
                key_hr, key_ndcg = "TestHR@%d" % k, "TestNDCG@%d" % k
                if self.metrics.get(key_hr):
                    f.write(
                        "Test HR@%d  (all): %s\n"
                        % (k, ", ".join("%.4f" % v for v in self.metrics[key_hr]))
                    )
                if self.metrics.get(key_ndcg):
                    f.write(
                        "Test NDCG@%d (all): %s\n"
                        % (k, ", ".join("%.4f" % v for v in self.metrics[key_ndcg]))
                    )
            if self.metrics.get("TrainLoss"):
                f.write("Train Loss (all): %s\n" % ", ".join("%.4f" % v for v in self.metrics["TrainLoss"]))
            if self.metrics.get("TestLoss"):
                f.write("Test Loss (all): %s\n" % ", ".join("%.4f" % v for v in self.metrics["TestLoss"]))
        log("Saved: %s" % path)

    def _save_metrics_plots(self) -> None:
        """Plot Train/Test Loss, HR, NDCG and save to results/<data>/."""
        base_dir = self._results_base_dir()
        base = os.path.join(base_dir, args.save_path)

        n_train = len(self.metrics["TrainLoss"])
        if n_train == 0:
            return
        steps_train = list(range(n_train))
        n_test = len(self.metrics["TestHR"])
        steps_test = list(range(n_test))

        # 1) Loss plot
        fig, ax = plt.subplots(figsize=(8, 5))
        if self.metrics["TrainLoss"]:
            ax.plot(steps_train, self.metrics["TrainLoss"], "b-", label="Train Loss", alpha=0.8)
        if self.metrics["TrainpreLoss"]:
            ax.plot(steps_train, self.metrics["TrainpreLoss"], "b--", label="Train preLoss", alpha=0.7)
        if self.metrics["TestLoss"] and steps_test:
            ax.plot(steps_test, self.metrics["TestLoss"], "r-", label="Test Loss", alpha=0.8)
        if self.metrics["TestpreLoss"] and steps_test:
            ax.plot(steps_test, self.metrics["TestpreLoss"], "r--", label="Test preLoss", alpha=0.7)
        ax.set_xlabel("Eval step")
        ax.set_ylabel("Loss")
        ax.set_title("Training and test loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(base + "_loss.png", dpi=150)
        plt.close(fig)
        log("Saved: %s_loss.png" % base)

        # 2) HR and NDCG plot (all K)
        fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(args.topk) * 2, 2)))
        for idx, k in enumerate(args.topk):
            key_hr, key_ndcg = "TestHR@%d" % k, "TestNDCG@%d" % k
            if self.metrics.get(key_hr) and steps_test:
                ax1.plot(
                    steps_test,
                    self.metrics[key_hr],
                    color=colors[idx % len(colors)],
                    label="Test HR@%d" % k,
                    alpha=0.8,
                )
            if self.metrics.get(key_ndcg) and steps_test:
                ax2.plot(
                    steps_test,
                    self.metrics[key_ndcg],
                    color=colors[idx % len(colors)],
                    label="Test NDCG@%d" % k,
                    alpha=0.8,
                )
        if self.metrics["TrainHR"]:
            ax1.plot(steps_train, self.metrics["TrainHR"], "b-", label="Train HR", alpha=0.8)
        if self.metrics["TrainNDCG"]:
            ax2.plot(steps_train, self.metrics["TrainNDCG"], "b-", label="Train NDCG", alpha=0.8)
        ax1.set_ylabel("HR")
        ax1.set_title("Hit Rate (HR)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax2.set_xlabel("Eval step")
        ax2.set_ylabel("NDCG")
        ax2.set_title("NDCG")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        fig2.tight_layout()
        fig2.savefig(base + "_hr_ndcg.png", dpi=150)
        plt.close(fig2)
        log("Saved: %s_hr_ndcg.png" % base)

        # Write full run log to file
        log_path = os.path.join(base_dir, args.save_path + "_run.log")
        time_logger.flush_log(log_path)

    def sample_train_batch(
        self, bat_ids, label_mat, time_mat, train_sample_num
    ):
        tem_tst = self.handler.tstInt[bat_ids]
        tem_label = label_mat[bat_ids].toarray()
        batch = len(bat_ids)
        temlen = batch * 2 * train_sample_num
        u_locs = [None] * temlen
        i_locs = [None] * temlen
        u_locs_seq = [None] * temlen
        sequence = [None] * args.batch
        mask = [None] * args.batch
        cur = 0
        for i in range(batch):
            posset = self.handler.sequence[bat_ids[i]][:-1]
            samp_num = min(train_sample_num, len(posset))
            choose = 1
            if samp_num == 0:
                poslocs = [np.random.choice(args.item)]
                neglocs = [poslocs[0]]
            else:
                choose = randint(
                    1, max(min(args.pred_num + 1, len(posset) - 3), 1)
                )
                poslocs = [posset[-choose]] * samp_num
                neglocs = negSamp(
                    tem_label[i],
                    samp_num,
                    args.item,
                    [
                        self.handler.sequence[bat_ids[i]][-1],
                        tem_tst[i],
                    ],
                    self.handler.item_with_pop,
                )
            for j in range(samp_num):
                posloc = poslocs[j]
                negloc = neglocs[j]
                u_locs[cur] = u_locs[cur + temlen // 2] = bat_ids[i]
                u_locs_seq[cur] = u_locs_seq[cur + temlen // 2] = i
                i_locs[cur] = posloc
                i_locs[cur + temlen // 2] = negloc
                cur += 1
            sequence[i] = np.zeros(args.pos_length, dtype=np.int64)
            mask[i] = np.zeros(args.pos_length, dtype=np.float32)
            posset = posset[:-choose] if samp_num > 0 else posset
            if len(posset) <= args.pos_length:
                sequence[i][-len(posset) :] = posset
                mask[i][-len(posset) :] = 1
            else:
                sequence[i] = np.array(posset[-args.pos_length :], dtype=np.int64)
                mask[i] = np.ones(args.pos_length, dtype=np.float32)
        u_locs = u_locs[:cur] + u_locs[temlen // 2 : temlen // 2 + cur]
        i_locs = i_locs[:cur] + i_locs[temlen // 2 : temlen // 2 + cur]
        u_locs_seq = u_locs_seq[:cur] + u_locs_seq[temlen // 2 : temlen // 2 + cur]
        if batch < args.batch:
            for i in range(batch, args.batch):
                sequence[i] = np.zeros(args.pos_length, dtype=np.int64)
                mask[i] = np.zeros(args.pos_length, dtype=np.float32)
        return u_locs, i_locs, sequence, mask, u_locs_seq

    def sample_ssl_batch(self, bat_ids, label_mat_list, use_epsilon=True):
        batch = len(bat_ids)
        temlen = batch * 2 * args.sslNum
        u_locs = [[None] * temlen for _ in range(args.graphNum)]
        i_locs = [[None] * temlen for _ in range(args.graphNum)]
        u_locs_seq = [[None] * temlen for _ in range(args.graphNum)]
        for k in range(args.graphNum):
            tem_label = label_mat_list[k][bat_ids].toarray()
            cur = 0
            for i in range(batch):
                posset = np.reshape(np.argwhere(tem_label[i] != 0), [-1])
                ssl_num = min(args.sslNum, len(posset) // 2)
                if ssl_num == 0:
                    poslocs = [np.random.choice(args.item)]
                    neglocs = [poslocs[0]]
                else:
                    all_idx = np.random.choice(posset, ssl_num * 2)
                    poslocs = all_idx[: ssl_num]
                    neglocs = all_idx[ssl_num:]
                for j in range(ssl_num):
                    posloc = poslocs[j]
                    negloc = neglocs[j]
                    u_locs[k][cur] = u_locs[k][cur + 1] = bat_ids[i]
                    u_locs_seq[k][cur] = u_locs_seq[k][cur + 1] = i
                    i_locs[k][cur] = posloc
                    i_locs[k][cur + 1] = negloc
                    cur += 2
            u_locs[k] = u_locs[k][:cur]
            i_locs[k] = i_locs[k][:cur]
            u_locs_seq[k] = u_locs_seq[k][:cur]
        return u_locs, i_locs, u_locs_seq

    def train_epoch(self) -> dict:
        num = args.user
        sf_ids = np.random.permutation(num)[: args.trnNum]
        num = len(sf_ids)
        sample_num_list = [40]
        steps = int(np.ceil(num / args.batch))
        epoch_loss = 0.0
        epoch_pre_loss = 0.0
        self.model.train()
        for s in range(len(sample_num_list)):
            for i in range(steps):
                st = i * args.batch
                ed = min((i + 1) * args.batch, num)
                bat_ids = sf_ids[st:ed]
                u_locs, i_locs, sequence, mask, u_locs_seq = self.sample_train_batch(
                    bat_ids,
                    self.handler.trnMat,
                    self.handler.timeMat,
                    sample_num_list[s],
                )
                su_locs, si_locs, su_locs_seq = self.sample_ssl_batch(
                    bat_ids, self.handler.subMat, False
                )
                uids = torch.tensor(u_locs, dtype=torch.long, device=self.device)
                iids = torch.tensor(i_locs, dtype=torch.long, device=self.device)
                seq_t = torch.tensor(
                    np.array(sequence), dtype=torch.long, device=self.device
                )
                mask_t = torch.tensor(
                    np.array(mask), dtype=torch.float32, device=self.device
                )
                u_locs_seq_t = torch.tensor(
                    u_locs_seq, dtype=torch.long, device=self.device
                )
                suids_list = [
                    torch.tensor(su_locs[k], dtype=torch.long, device=self.device)
                    for k in range(args.graphNum)
                ]
                siids_list = [
                    torch.tensor(si_locs[k], dtype=torch.long, device=self.device)
                    for k in range(args.graphNum)
                ]
                preds, sslloss, _ = self.model(
                    uids,
                    iids,
                    seq_t,
                    mask_t,
                    u_locs_seq_t,
                    suids_list,
                    siids_list,
                    keep_rate=args.keepRate,
                )
                samp_num = uids.size(0) // 2
                pre_loss = self._pre_loss(preds, samp_num)
                reg_loss = args.reg * regularize(self.model, "L2") + args.ssl_reg * sslloss
                loss = pre_loss + reg_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                epoch_pre_loss += pre_loss.item()
                log(
                    "Step %d/%d: preloss = %.2f, REGLoss = %.2f         "
                    % (i + s * steps, steps * len(sample_num_list), pre_loss.item(), reg_loss.item()),
                    save=False,
                    oneline=True,
                )
        self.scheduler.step()
        return {
            "Loss": epoch_loss / steps,
            "preLoss": epoch_pre_loss / steps,
        }

    def sample_test_batch(self, bat_ids, label_mat):
        batch = len(bat_ids)
        tem_tst = self.handler.tstInt[bat_ids]
        tem_label = label_mat[bat_ids].toarray()
        temlen = batch * args.testSize
        u_locs = [None] * temlen
        i_locs = [None] * temlen
        u_locs_seq = [None] * temlen
        tst_locs = [None] * batch
        sequence = [None] * args.batch
        mask = [None] * args.batch
        val_list = [None] * args.batch
        cur = 0
        for i in range(batch):
            if args.test:
                posloc = tem_tst[i]
            else:
                posloc = self.handler.sequence[bat_ids[i]][-1]
                val_list[i] = posloc
            rdn_neg_set = np.array(
                self.handler.test_dict.get(bat_ids[i] + 1, [])[: args.testSize - 1]
            )
            if len(rdn_neg_set) > 0:
                rdn_neg_set = rdn_neg_set - 1
            else:
                rdn_neg_set = np.random.randint(0, args.item, size=args.testSize - 1)
            locset = np.concatenate((rdn_neg_set, np.array([posloc])))
            tst_locs[i] = locset
            for j in range(len(locset)):
                u_locs[cur] = bat_ids[i]
                i_locs[cur] = locset[j]
                u_locs_seq[cur] = i
                cur += 1
            sequence[i] = np.zeros(args.pos_length, dtype=np.int64)
            mask[i] = np.zeros(args.pos_length, dtype=np.float32)
            if args.test:
                posset = self.handler.sequence[bat_ids[i]]
            else:
                posset = self.handler.sequence[bat_ids[i]][:-1]
            if len(posset) <= args.pos_length:
                sequence[i][-len(posset) :] = posset
                mask[i][-len(posset) :] = 1
            else:
                sequence[i] = np.array(posset[-args.pos_length :], dtype=np.int64)
                mask[i] = np.ones(args.pos_length, dtype=np.float32)
        if batch < args.batch:
            for i in range(batch, args.batch):
                sequence[i] = np.zeros(args.pos_length, dtype=np.int64)
                mask[i] = np.zeros(args.pos_length, dtype=np.float32)
        return (
            u_locs,
            i_locs,
            tem_tst,
            tst_locs,
            sequence,
            mask,
            u_locs_seq,
            val_list,
        )

    def test_epoch(self) -> dict:
        epoch_hits = {k: 0 for k in args.topk}
        epoch_ndcgs = {k: 0.0 for k in args.topk}
        ids = self.handler.tstUsrs
        num = len(ids)
        tst_bat = args.batch
        steps = int(np.ceil(num / tst_bat))
        self.model.eval()
        with torch.no_grad():
            for i in range(steps):
                st = i * tst_bat
                ed = min((i + 1) * tst_bat, num)
                bat_ids = ids[st:ed]
                (
                    u_locs,
                    i_locs,
                    tem_tst,
                    tst_locs,
                    sequence,
                    mask,
                    u_locs_seq,
                    val_list,
                ) = self.sample_test_batch(bat_ids, self.handler.trnMat)
                su_locs, si_locs, _ = self.sample_ssl_batch(
                    bat_ids, self.handler.subMat, False
                )
                uids = torch.tensor(u_locs, dtype=torch.long, device=self.device)
                iids = torch.tensor(i_locs, dtype=torch.long, device=self.device)
                seq_t = torch.tensor(
                    np.array(sequence), dtype=torch.long, device=self.device
                )
                mask_t = torch.tensor(
                    np.array(mask), dtype=torch.float32, device=self.device
                )
                u_locs_seq_t = torch.tensor(
                    u_locs_seq, dtype=torch.long, device=self.device
                )
                suids_list = [
                    torch.tensor(su_locs[k], dtype=torch.long, device=self.device)
                    for k in range(args.graphNum)
                ]
                siids_list = [
                    torch.tensor(si_locs[k], dtype=torch.long, device=self.device)
                    for k in range(args.graphNum)
                ]
                preds = self.model(
                    uids,
                    iids,
                    seq_t,
                    mask_t,
                    u_locs_seq_t,
                    suids_list,
                    siids_list,
                    keep_rate=1.0,
                )[0]
                preds = preds.cpu().numpy()
                if args.uid != -1:
                    print(preds[args.uid])
                if args.test:
                    res_batch = self.calc_res(
                        np.reshape(preds, [ed - st, args.testSize]),
                        tem_tst,
                        tst_locs,
                    )
                else:
                    res_batch = self.calc_res(
                        np.reshape(preds, [ed - st, args.testSize]),
                        val_list,
                        tst_locs,
                    )
                for k in args.topk:
                    epoch_hits[k] += res_batch[k][0]
                    epoch_ndcgs[k] += res_batch[k][1]
                # log line uses primary K
                pk = args.topk[0]
                log(
                    "Steps %d/%d: hit%d = %d, ndcg%d = %d"
                    % (i, steps, pk, res_batch[pk][0], pk, res_batch[pk][1]),
                    save=False,
                    oneline=True,
                )
        primary_k = args.topk[0]
        reses = {
            "HR": epoch_hits[primary_k] / num,
            "NDCG": epoch_ndcgs[primary_k] / num,
        }
        for k in args.topk:
            reses["HR@%d" % k] = epoch_hits[k] / num
            reses["NDCG@%d" % k] = epoch_ndcgs[k] / num
        return reses

    def calc_res(self, preds, tem_tst, tst_locs):
        """For each K in args.topk, compute (hit_count, ndcg_sum) for this batch."""
        max_k = max(args.topk)
        out = {k: [0, 0.0] for k in args.topk}  # hit count, ndcg sum
        for j in range(preds.shape[0]):
            predvals = list(zip(preds[j], tst_locs[j]))
            predvals.sort(key=lambda x: x[0], reverse=True)
            top_items = list(map(lambda x: x[1], predvals[:max_k]))
            for k in args.topk:
                shoot = top_items[:k]
                if tem_tst[j] in shoot:
                    out[k][0] += 1
                    out[k][1] += np.reciprocal(np.log2(shoot.index(tem_tst[j]) + 2))
        return {k: (out[k][0], out[k][1]) for k in args.topk}

    def save_history(self) -> None:
        if args.epoch == 0:
            return
        os.makedirs("History", exist_ok=True)
        with open("History/" + args.save_path + ".his", "wb") as fs:
            pickle.dump(self.metrics, fs)
        os.makedirs("Models", exist_ok=True)
        torch.save(self.model.state_dict(), "Models/" + args.save_path)
        log("Model Saved: %s" % args.save_path)

    def load_model(self) -> None:
        self.model.load_state_dict(
            torch.load("Models/" + args.load_model, map_location=self.device)
        )
        his_path = "History/" + args.load_model + ".his"
        if os.path.isfile(his_path):
            with open(his_path, "rb") as fs:
                self.metrics = pickle.load(fs)
            # Ensure per-K keys exist (e.g. when loading history from older runs)
            for k in args.topk:
                for prefix in ("Train", "Test"):
                    for m in ("HR", "NDCG"):
                        key = "%s%s@%d" % (prefix, m, k)
                        if key not in self.metrics:
                            self.metrics[key] = []
        log("Model Loaded")