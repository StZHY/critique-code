"""Residual-LLM Critique model: fully frozen backbone + a non-propagating residual head + multi-task loss."""
import os
import numpy as np
import torch
from torch import nn
import scipy.sparse as sp
import utility.tools as tools
import utility.losses_rllm as losses


class Critique(nn.Module):
    def __init__(self, args, original_user_embedding_weights, original_item_embedding_weights,
                 user_fp=None, user_fp_by_round=None):
        super(Critique, self).__init__()
        self.device = torch.device("cuda:" + str(args.gpu)) if args.cuda else torch.device("cpu")
        self.model_name = "Critique"
        self.activation = nn.Sigmoid()
        self.args = args

        # ---- backbone fully frozen (plugin base) ----
        self.user_embedding = nn.Embedding.from_pretrained(
            original_user_embedding_weights.clone().detach(), freeze=True)
        self.item_embedding = nn.Embedding.from_pretrained(
            original_item_embedding_weights.clone().detach(), freeze=True)

        # GCN adjacency (D^-1/2 A D^-1/2, no self-loop, same as LightCCF)
        adj_path = os.path.join(args.dataset_path, args.dataset, "pre_Adj.npz")
        adj = sp.load_npz(adj_path).astype(np.float32)
        self.adj_mat = tools.convert_sp_mat_to_sp_tensor(adj).to(self.device)
        self.gcn_layer = int(args.gcn_layer)
        self.critique_encoder = getattr(args, 'critique_encoder', 'MF')

        # ---- non-propagating residual head (zero-init => starts at backbone level) ----
        # Added after the user representation, never enters sparse.mm => zero D6 propagation.
        self.critique_residual = nn.Embedding(
            original_user_embedding_weights.shape[0], int(args.embedding_size))
        nn.init.zeros_(self.critique_residual.weight)
        self.npc_alpha = float(getattr(args, 'npc_alpha', 1.0))

        # ---- residual-LLM term config ----
        self.rllm_mode = getattr(args, 'rllm_mode', 'none')   # 'none' / 'reg' / 'pair'
        self.rllm_lambda = float(getattr(args, 'rllm_lambda', 1.0))
        self.rllm_tau = float(getattr(args, 'rllm_tau', 0.05))
        # Form B-v2 switches (prediction space / soft label / polarity stratification)
        self.rllm_pred_space = bool(getattr(args, 'rllm_pred_space', False))
        self.rllm_beta = float(getattr(args, 'rllm_beta', 0.0))
        self.rllm_within = float(getattr(args, 'rllm_within', 1.0))
        # (a) Orthogonal protection: residual projection on train-positive-mean direction -> 0.
        self.prot_lambda = float(getattr(args, 'prot_lambda', 0.0))
        # Residual propagation switch (npc_prop family vs npc_full family; backbone stays frozen).
        self.rllm_propagate = bool(getattr(args, 'rllm_propagate', False))
        # Polarity-aware propagation (npc_prop_neg family): two residual heads.
        #   prop head (critique_residual): baked into user emb + re-propagated => pos/CL (graph amplification).
        #   neg head (critique_residual_neg): post-aggregate, not in sparse.mm => neg (cuts D6).
        self.rllm_polarity = bool(getattr(args, 'rllm_polarity', False))
        if self.rllm_polarity:
            self.critique_residual_neg = nn.Embedding(
                original_user_embedding_weights.shape[0], int(args.embedding_size))
            nn.init.zeros_(self.critique_residual_neg.weight)
        self.rllm_alpha_neg = float(getattr(args, 'rllm_alpha_neg', 0.3))
        # D-3 (v2 Eq(19)): independent L2 knob on the residual head.
        self.delta_reg_lambda = float(getattr(args, 'delta_reg_lambda', 0.0))

        # ---- s_L-margin CER: B- pool = LLM dislike (round-level, all users [num_users,Kc]) ----
        self.use_cer_margin = bool(getattr(args, 'use_cer_margin', False))
        self.cer_margin = float(getattr(args, 'cer_margin', 0.5))
        self.cer_tau = float(getattr(args, 'cer_tau', 0.1))
        self.cer_cri_neg_ids = None
        self.cer_cri_neg_g = None

        # ---- P: gate-aware L_cri (dense gate penalty [num_users, num_items], built per active_round) ----
        self.lcri_gate_aware = bool(getattr(args, 'lcri_gate_aware', False))
        self.lcri_gate_M = float(getattr(args, 'lcri_gate_M', 1.0))
        self.gate_logit_dense = None

        # ---- Semantic-Offset L_cri: dense signed semantic offset q_dense (=-so_M*sigma(gamma*ell) <= 0) ----
        self.use_so = bool(getattr(args, 'use_semantic_offset_in_lcri', False))
        self.so_M = float(getattr(args, 'so_M', 1.0))
        self.cri_tau = float(getattr(args, 'cri_temperature', 1.0))
        self.so_q_dense = None

        # ---- Unified CCR (joint training score z=s_delta+q; ccr_joint_score distinguishes U1/U2) ----
        self.use_ccr = bool(getattr(args, 'use_ccr', False))
        self.ccr_joint_score = bool(getattr(args, 'ccr_joint_score', False))
        self.ccr_M = float(getattr(args, 'ccr_M', 1.0))
        self.ccr_tau = float(getattr(args, 'ccr_tau', 1.0))
        self.ccr_q_dense = None
        # Inference additive gate (M-DEC-002/003): logit += q=-integ_M*sigma(gamma*ell).
        self.integ_additive = bool(getattr(args, 'integ_additive', False))

        # ---- post-hoc gating (per-user FP demotion at inference) ----
        self.integ_M = float(getattr(args, 'integ_M', 10.0))
        self.integ_gamma = float(getattr(args, 'integ_gamma', 8.0))
        self.integ_hard_inf = bool(getattr(args, 'integ_hard_inference', True))
        num_users = original_user_embedding_weights.shape[0]
        self.fp_items = [None] * num_users
        self.fp_ell = [None] * num_users
        if user_fp is not None:
            for u, d in user_fp.items():
                if not d:
                    continue
                self.fp_items[u] = torch.tensor(list(d.keys()), dtype=torch.long, device=self.device)
                self.fp_ell[u] = torch.tensor(list(d.values()), dtype=torch.float, device=self.device)
        # Incremental gating: per-round accumulated FP (active_round=None = ungated baseline).
        self.fp_items_by_round = None
        self.fp_ell_by_round = None
        self.active_round = None
        if user_fp_by_round is not None:
            R = len(user_fp_by_round)
            self.fp_items_by_round = [None] * R
            self.fp_ell_by_round = [None] * R
            for r in range(R):
                fi = [None] * num_users
                fe = [None] * num_users
                for u, d in user_fp_by_round[r].items():
                    if not d:
                        continue
                    fi[u] = torch.tensor(list(d.keys()), dtype=torch.long, device=self.device)
                    fe[u] = torch.tensor(list(d.values()), dtype=torch.float, device=self.device)
                self.fp_items_by_round[r] = fi
                self.fp_ell_by_round[r] = fe

    def set_active_round(self, r):
        """Activate the accumulated-evidence state (incremental gating only).
        r=None => no gating (pure backbone baseline); r=k => use round k accumulated FP."""
        self.active_round = r
        if self.lcri_gate_aware and r is not None and self.fp_items_by_round is not None:
            self.gate_logit_dense = self._build_gate_logit_dense(r)
        else:
            self.gate_logit_dense = None
        if self.use_so and r is not None and self.fp_items_by_round is not None:
            self.so_q_dense = self._build_so_q_dense(r)
        else:
            self.so_q_dense = None
        if self.use_ccr and self.ccr_joint_score and r is not None and self.fp_items_by_round is not None:
            self.ccr_q_dense = self._build_ccr_q_dense(r)
        else:
            self.ccr_q_dense = None

    def _build_ccr_q_dense(self, r):
        """Unified CCR: build dense semantic offset [num_users, num_items], q[u,i]=-ccr_M*sigma(gamma*ell) (<=0, no grad)."""
        num_users = self.user_embedding.weight.shape[0]
        num_items = self.item_embedding.weight.shape[0]
        dense = torch.zeros((num_users, num_items), device=self.device)
        fp_items_r = self.fp_items_by_round[r]
        fp_ell_r = self.fp_ell_by_round[r]
        for u in range(num_users):
            if u < len(fp_items_r) and fp_items_r[u] is not None:
                items = fp_items_r[u]
                q = -1.0 * self.ccr_M * torch.sigmoid(self.integ_gamma * fp_ell_r[u])   # <=0
                dense[u, items] = q
        return dense

    def _build_so_q_dense(self, r):
        """Semantic-Offset: build dense signed offset [num_users, num_items], q[u,i]=-so_M*sigma(gamma*ell) (<=0, no grad)."""
        num_users = self.user_embedding.weight.shape[0]
        num_items = self.item_embedding.weight.shape[0]
        dense = torch.zeros((num_users, num_items), device=self.device)
        fp_items_r = self.fp_items_by_round[r]
        fp_ell_r = self.fp_ell_by_round[r]
        for u in range(num_users):
            if u < len(fp_items_r) and fp_items_r[u] is not None:
                items = fp_items_r[u]
                q = -1.0 * self.so_M * torch.sigmoid(self.integ_gamma * fp_ell_r[u])   # <=0
                dense[u, items] = q
        return dense

    def _build_gate_logit_dense(self, r):
        """P: build dense gate penalty [num_users, num_items], gate_pen[u,i]=lcri_gate_M*sigma(gamma*ell) (FP items only)."""
        num_users = self.user_embedding.weight.shape[0]
        num_items = self.item_embedding.weight.shape[0]
        dense = torch.zeros((num_users, num_items), device=self.device)
        fp_items_r = self.fp_items_by_round[r]
        fp_ell_r = self.fp_ell_by_round[r]
        for u in range(num_users):
            if u < len(fp_items_r) and fp_items_r[u] is not None:
                items = fp_items_r[u]
                pen = self.lcri_gate_M * torch.sigmoid(self.integ_gamma * fp_ell_r[u])
                dense[u, items] = pen
        return dense

    def set_cri_neg_cl(self, cri_neg_ids, cri_neg_g):
        """Inject the s_L-margin CER B- pool (round-level, all users [num_users,Kc])."""
        self.cer_cri_neg_ids = cri_neg_ids
        self.cer_cri_neg_g = cri_neg_g

    def _apply_gate(self, rating, user):
        """post-hoc gating: per-user FP demotion to free top-K slots.
        hard gate: rating *= exp(-M); soft gate: rating *= exp(-M*sigma(gamma*ell))."""
        if self.fp_items_by_round is not None:
            if self.active_round is None:
                return rating                       # baseline: no gating (pure backbone)
            fp_items_r = self.fp_items_by_round[self.active_round]
            fp_ell_r = self.fp_ell_by_round[self.active_round]
        elif any(x is not None for x in self.fp_items):
            fp_items_r = self.fp_items             # full static gating
            fp_ell_r = self.fp_ell
        else:
            return rating                          # no FP set => no gating
        if self.integ_M > 0:
            for bi, u in enumerate(user.tolist()):
                if u < len(fp_items_r) and fp_items_r[u] is not None:
                    items = fp_items_r[u]
                    if self.integ_hard_inf:
                        rating[bi, items] = rating[bi, items] * float(np.exp(-self.integ_M))
                    else:
                        demotion = self.integ_M * torch.sigmoid(self.integ_gamma * fp_ell_r[u])
                        rating[bi, items] = rating[bi, items] * torch.exp(-demotion).clamp(min=1e-6)
        return rating

    def _select_fp_round(self):
        """Select the FP set for the current active_round; returns (fp_items_r, fp_ell_r) or None."""
        if self.fp_items_by_round is not None:
            if self.active_round is None:
                return None                       # baseline: no gating
            return self.fp_items_by_round[self.active_round], self.fp_ell_by_round[self.active_round]
        if any(x is not None for x in self.fp_items):
            return self.fp_items, self.fp_ell      # full static gating
        return None

    def _apply_gate_additive(self, logit, user):
        """Additive inference gate (M-DEC-002/003): logit += q=-integ_M*sigma(gamma*ell) on FP items
        (joint score z=s_delta+q, applied in pre-sigmoid logit space)."""
        sel = self._select_fp_round()
        if sel is None:
            return logit
        fp_items_r, fp_ell_r = sel
        if self.integ_M > 0:
            for bi, u in enumerate(user.tolist()):
                if u < len(fp_items_r) and fp_items_r[u] is not None:
                    items = fp_items_r[u]
                    q = -1.0 * self.integ_M * torch.sigmoid(self.integ_gamma * fp_ell_r[u])   # <=0
                    logit[bi, items] = logit[bi, items] + q
        return logit

    def _finalize_rating(self, user_embed, all_item_embed, user):
        """Eval finalization: matmul -> (additive gate adds q to logit | multiplicative gate on rating) -> activation."""
        logit = torch.matmul(user_embed, all_item_embed.t())
        if self.integ_additive:
            logit = self._apply_gate_additive(logit, user)
            return self.activation(logit)
        rating = self.activation(logit)
        return self._apply_gate(rating, user)

    def supplement_average_items(self, train_user_set):
        user_num = len(train_user_set)
        aver_items_tensor_all = torch.empty(user_num, int(self.args.embedding_size)).to(self.device)
        item_weights = self.item_embedding.weight
        for user, items in train_user_set.items():
            items_tensor = torch.tensor(items, dtype=int, device=self.device)
            item_emb = item_weights[items_tensor]
            aver_items_tensor = torch.mean(item_emb, dim=0)
            aver_items_tensor_all[user] = aver_items_tensor
        combined_item_embedding = torch.cat((item_weights, aver_items_tensor_all), dim=0)
        self.aver_item_embedding = nn.Embedding.from_pretrained(combined_item_embedding, freeze=True).to(self.device)

    def aggregate_with_user(self, user_emb_tensor):
        """GCN propagation (same as LightCCF.aggregate): cat(user,item) -> gcn_layer sparse.mm(adj) -> mean.
        Backbone is frozen; the residual does not enter here (non-propagating)."""
        num_users = user_emb_tensor.shape[0]
        num_items = self.item_embedding.weight.shape[0]
        embeddings = torch.cat([user_emb_tensor, self.item_embedding.weight], dim=0)
        all_embeddings = [embeddings]
        for _ in range(self.gcn_layer):
            embeddings = torch.sparse.mm(self.adj_mat, embeddings)
            all_embeddings.append(embeddings)
        final_embeddings = torch.stack(all_embeddings, dim=1)
        final_embeddings = torch.mean(final_embeddings, dim=1)
        user_emb, item_emb = torch.split(final_embeddings, [num_users, num_items])
        return user_emb, item_emb

    def aggregate(self):
        return self.aggregate_with_user(self.user_embedding.weight)

    def _backbone_user_item(self, user, bpr_negative):
        """Frozen backbone user / bpr_neg item representations (no residual). GCN=post-propagation; MF=raw."""
        if self.critique_encoder == 'GCN':
            all_user_gcn, all_item_gcn = self.aggregate()
            return all_user_gcn[user.long()], all_item_gcn, all_item_gcn
        else:
            return self.user_embedding(user.long()), self.item_embedding.weight, self.item_embedding(bpr_negative.long())

    def forward(self, user, positive, bpr_negative, cl_negative,
                pos_candidates=None, neg_mask=None, pos_mask=None, neg_gate=None,
                pos_pool=None, pos_pool_scores=None,
                cri_pool=None, cri_pool_scores=None):
        """Multi-task loss = neg_w*neg_bpr + cl_lambda*K*cl + pos_lambda*pos_bpr + rllm_lambda*residual_LLM.
        All losses use u_aug = backbone_user + alpha*residual (residual receives all gradients, backbone frozen).
        Item representations are detached (frozen backbone + cut any item-side flow => zero D6)."""
        # ---- frozen backbone representations + residual routing ----
        # Two families (backbone always frozen = true plugin):
        #   non-propagating (npc_full, rllm_propagate=False): residual added after aggregate, not in sparse.mm.
        #   propagating (npc_prop, rllm_propagate=True): residual baked into the full user emb then re-propagated.
        res = self.critique_residual(user.long())                  # [B,d] prop head (the trainable param)
        use_prop = self.rllm_propagate or self.rllm_polarity       # polarity implies propagation
        if use_prop:
            full_user = self.user_embedding.weight + self.npc_alpha * self.critique_residual.weight
            if self.critique_encoder == 'GCN':
                all_user_gcn, all_item_gcn = self.aggregate_with_user(full_user)
                user_embed = all_user_gcn[user.long()]
                bpr_neg_embed = all_item_gcn[bpr_negative.long()]
                item_lookup = all_item_gcn
            else:
                user_embed = full_user[user.long()]
                bpr_neg_embed = self.item_embedding(bpr_negative.long())
                item_lookup = self.item_embedding.weight
            u_aug = user_embed                                     # residual baked + propagated (pos/CL use this)
            if self.rllm_polarity:
                # neg head: post-aggregate + detach(propagated reps) => not in sparse.mm, no flow to prop head.
                r_neg = self.critique_residual_neg(user.long())
                u_aug_neg = user_embed.detach() + self.rllm_alpha_neg * r_neg
            else:
                u_aug_neg = u_aug
        else:
            if self.critique_encoder == 'GCN':
                all_user_gcn, all_item_gcn = self.aggregate()      # frozen user weight, no residual
                user_embed = all_user_gcn[user.long()]
                bpr_neg_embed = all_item_gcn[bpr_negative.long()]
                item_lookup = all_item_gcn
            else:
                user_embed = self.user_embedding(user.long())
                bpr_neg_embed = self.item_embedding(bpr_negative.long())
                item_lookup = self.item_embedding.weight
            u_aug = user_embed + self.npc_alpha * res              # residual added post-aggregate (non-propagating)
            u_aug_neg = u_aug
        positive_embed = self.aver_item_embedding(positive.long())
        cl_neg_embed = self.item_embedding(cl_negative.long())

        # ---- neg BPR (item detach => gradient only to res) ----
        neg_embed_det = bpr_neg_embed.detach()
        if self.use_so and self.so_q_dense is not None:
            # Semantic-Offset L_cri: pairwise joint-margin BPR on s_delta+q.
            q_neg = self.so_q_dense[user.long()].gather(1, bpr_negative.long())   # [B,Kn] no grad
            neg_loss = losses.get_semantic_offset_bpr_loss(
                u_aug_neg, positive_embed.detach(), neg_embed_det, q_neg,
                self.cri_tau, neg_mask=neg_mask)
        elif self.lcri_gate_aware and self.gate_logit_dense is not None:
            # P: L_cri computed on the gated score. Gather the per-(user,bpr_neg) gate penalty into neg_score.
            gate_pen = self.gate_logit_dense[user.long()].gather(1, bpr_negative.long())   # [B,K]
            neg_loss = losses.get_critique_neg_loss_gateaware(
                u_aug_neg, neg_embed_det, gate_pen, mask=neg_mask, gate=neg_gate)
        elif neg_gate is not None:
            neg_loss = losses.get_critique_neg_loss_gated(u_aug_neg, neg_embed_det, neg_gate)
        elif neg_mask is not None:
            neg_loss = losses.get_critique_neg_loss_masked(u_aug_neg, neg_embed_det, neg_mask)
        else:
            neg_loss = losses.get_critique_loss_base(u_aug_neg, neg_embed_det)
        # ---- CL / CER: anchor = train-positive mean (CF global anchor); neg bucket = Bref random ----
        if self.use_cer_margin and self.cer_cri_neg_ids is not None:
            cri_ids_b = self.cer_cri_neg_ids[user.long()]              # [B,Kc]
            cri_g_b = self.cer_cri_neg_g[user.long()]                  # [B,Kc]
            cri_neg_emb_b = self.item_embedding(cri_ids_b.long())      # [B,Kc,d]
            cl_loss = losses.get_cer_margin_loss(u_aug, positive_embed, cl_neg_embed,
                                                 cri_neg_emb_b, cri_g_b,
                                                 self.cer_margin, self.cer_tau)
        else:
            cl_loss = losses.get_critique_InfoNCE_loss(u_aug, positive_embed, cl_neg_embed)
        neg_w = float(getattr(self.args, 'critique_neg_lambda', 1.0))
        # ---- Unified CCR: single set-wise contrast replacing neg_loss + cl_loss ----
        if self.use_ccr:
            if self.ccr_joint_score and self.ccr_q_dense is not None:
                q_cri = self.ccr_q_dense[user.long()].gather(1, bpr_negative.long())   # [B,Kn] no grad <=0
            else:
                q_cri = torch.zeros(user_embed.shape[0], bpr_negative.shape[1], device=user_embed.device)
            ccr_loss = losses.get_ccr_loss(u_aug, positive_embed, cl_neg_embed,
                                           bpr_neg_embed.detach(), q_cri, self.ccr_tau)
            total_loss = self.args.critique_cl_lambda * self.args.num_bpr_neg * ccr_loss
        else:
            total_loss = neg_w * neg_loss + self.args.critique_cl_lambda * self.args.num_bpr_neg * cl_loss

        # ---- pos BPR (item detach) ----
        if pos_candidates is not None:
            pos_embed = item_lookup[pos_candidates.long()] if self.critique_encoder == 'GCN' \
                else self.item_embedding(pos_candidates.long())
            pos_embed = pos_embed.detach()
            if pos_mask is None:
                pos_mask = torch.ones(user_embed.shape[0], device=user_embed.device)
            pos_loss = losses.get_critique_pos_loss_masked(u_aug, pos_embed, pos_mask)
            total_loss = total_loss + self.args.critique_pos_lambda * pos_loss

        # ---- residual-LLM term (pos/neg symmetric, gradient only to res) ----
        if self.rllm_mode != 'none' and cri_pool is not None and cri_pool_scores is not None:
            pool_embed = item_lookup[cri_pool.long()] if self.critique_encoder == 'GCN' \
                else self.item_embedding(cri_pool.long())
            pool_embed = pool_embed.detach()
            if self.rllm_mode == 'reg':
                rllm_loss = losses.get_residual_llm_loss_reg(res, pool_embed, cri_pool_scores)
            elif self.rllm_mode == 'pair':
                if self.rllm_polarity:
                    # Polarity-aware v2: pos members via propagated u_aug / neg members via non-propagated u_aug_neg.
                    score_prop = torch.sum(u_aug.unsqueeze(1) * pool_embed, dim=2)       # [B,P] -> prop head
                    score_neg = torch.sum(u_aug_neg.unsqueeze(1) * pool_embed, dim=2)    # [B,P] -> neg head
                    rllm_loss = losses.get_residual_llm_loss_pair_v2_polarity(
                        score_prop, score_neg, cri_pool_scores,
                        self.rllm_beta, self.rllm_within, self.rllm_tau)
                elif self.rllm_pred_space or self.rllm_beta > 0 or self.rllm_within < 1.0:
                    # v2: prediction-space score + soft label + polarity stratification
                    score_pred = torch.sum(u_aug.unsqueeze(1) * pool_embed, dim=2)   # [B,P]
                    rllm_loss = losses.get_residual_llm_loss_pair_v2(
                        score_pred, cri_pool_scores, self.rllm_beta, self.rllm_within, self.rllm_tau)
                else:
                    rllm_loss = losses.get_residual_llm_loss_pair(res, pool_embed, cri_pool_scores, self.rllm_tau)
            else:
                rllm_loss = torch.zeros((), device=user_embed.device)
            total_loss = total_loss + self.rllm_lambda * rllm_loss

        # ---- (a) orthogonal protection: residual projection on train-positive-mean direction -> 0 ----
        if self.prot_lambda > 0:
            num_items = self.item_embedding.weight.shape[0]
            anchor = self.aver_item_embedding((user + num_items).long())   # [B,d] frozen
            prot_loss = losses.get_residual_orth_loss(res, anchor)
            total_loss = total_loss + self.prot_lambda * prot_loss

        # ---- D-3 (v2 Eq(19)): lambda_reg*||delta||^2 L2 term (mini-batch mean) ----
        if self.delta_reg_lambda > 0:
            reg_term = (res ** 2).mean()
            if self.rllm_polarity:
                reg_term = reg_term + (self.critique_residual_neg(user.long()) ** 2).mean()
            total_loss = total_loss + self.delta_reg_lambda * reg_term

        return total_loss

    def get_rating_for_test(self, user):
        # Training (forward) / eval caliber are consistent.
        # Polarity-aware: prop head baked+propagated, neg head post-aggregate.
        # Propagating: residual baked into full user emb then re-propagated.
        # Non-propagating: aggregate uses frozen user weight (items clean), residual added post-aggregate.
        if self.rllm_polarity:
            full_user = self.user_embedding.weight + self.npc_alpha * self.critique_residual.weight
            if self.critique_encoder == 'GCN':
                all_user_embed, all_item_embed = self.aggregate_with_user(full_user)
            else:
                all_user_embed = full_user
                all_item_embed = self.item_embedding.weight
            user_embed = all_user_embed[user.long()]
            user_embed = user_embed + self.rllm_alpha_neg * self.critique_residual_neg(user.long())
            return self._finalize_rating(user_embed, all_item_embed, user)
        if self.rllm_propagate:
            full_user = self.user_embedding.weight + self.npc_alpha * self.critique_residual.weight
            if self.critique_encoder == 'GCN':
                all_user_embed, all_item_embed = self.aggregate_with_user(full_user)
            else:
                all_user_embed = full_user
                all_item_embed = self.item_embedding.weight
            user_embed = all_user_embed[user.long()]
            return self._finalize_rating(user_embed, all_item_embed, user)
        if self.critique_encoder == 'GCN':
            all_user_embed, all_item_embed = self.aggregate()
            user_embed = all_user_embed[user.long()]
        else:
            all_user_embed = self.user_embedding.weight
            all_item_embed = self.item_embedding.weight
            user_embed = all_user_embed[user.long()]
        user_embed = user_embed + self.npc_alpha * self.critique_residual(user.long())
        return self._finalize_rating(user_embed, all_item_embed, user)
