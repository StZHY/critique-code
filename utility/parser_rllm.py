"""Argument parser for simulate_rllm (residual-LLM multi-task critique parameters)."""
import argparse

def parse_args():
    parse = argparse.ArgumentParser(description="Run Reproduct")
    parse.add_argument('--seed', type=int, default=2025, help='random seed')
    parse.add_argument('--gpu', type=int, default=0, help='indicates which gpu to use')
    parse.add_argument('--cuda', type=bool, default=True, help='use gpu or not')
    parse.add_argument('--log', type=str, default='None', help='init log file name')
    parse.add_argument('--dataset_path', type=str, default='./dataset/', help='choice dataset')
    parse.add_argument('--dataset_type', type=str, default='.txt', help='choice dataset')
    parse.add_argument('--dataset', type=str, default='movielens-1m', help='choice dataset')
    parse.add_argument('--top_K', type=str, default='[5, 10, 20]')
    parse.add_argument('--train_epoch', type=int, default=600)
    parse.add_argument('--critique_round_num', type=int, default=5)
    parse.add_argument('--critique_epoch', type=int, default=12)
    parse.add_argument('--early_stop', type=int, default=10)
    parse.add_argument('--early_stop_metric', type=str, default='recall10',
                       choices=['recall5', 'recall10', 'recall20', 'ndcg5', 'ndcg10', 'ndcg20', 'train_ndcg5'])
    parse.add_argument('--embedding_size', type=int, default=64)
    parse.add_argument('--train_batch_size', type=int, default=2048)
    parse.add_argument('--test_batch_size', type=int, default=2048)
    parse.add_argument('--learn_rate', type=float, default=0.002)
    parse.add_argument('--critique_rate', type=float, default=0.001)
    parse.add_argument('--reg_lambda', type=float, default=0.0001)
    parse.add_argument('--gcn_layer', type=int, default=3)
    parse.add_argument('--test_frequency', type=int, default=1)
    parse.add_argument('--sparsity_test', type=int, default=0)
    parse.add_argument('--tau', type=float, default=0.28)
    parse.add_argument('--ssl_lambda', type=float, default=5.0)
    parse.add_argument('--encoder', type=str, default='MF')
    parse.add_argument('--critique_encoder', type=str, default='MF', choices=['MF', 'GCN'])

    parse.add_argument('--num_bpr_neg', type=int, default=10)
    parse.add_argument('--num_cl_neg', type=int, default=1000)
    parse.add_argument('--critique_cl_lambda', type=float, default=1.0)
    parse.add_argument('--critique_pos_lambda', type=float, default=0.8)
    parse.add_argument('--critique_neg_lambda', type=float, default=1.0)
    parse.add_argument('--lambda_cf', type=float, default=0.8)
    parse.add_argument('--lambda_cross', type=float, default=0.0)
    parse.add_argument('--prd_pos_lambda', type=float, default=0.0)
    parse.add_argument('--prd_tau', type=float, default=0.05)
    parse.add_argument('--prd_gap_weight', type=float, default=1.0)
    parse.add_argument('--force_neg_anchor', action='store_true', default=False)

    parse.add_argument('--neg_gate_mode', type=str, default='none',
                       choices=['none', 'linear', 'power', 'softmax', 'hard_topk', 'hard_thresh'])
    parse.add_argument('--neg_gate_gamma', type=float, default=5.0)
    parse.add_argument('--neg_gate_topk', type=int, default=5)
    parse.add_argument('--neg_gate_cftable', type=str, default=None)
    parse.add_argument('--neg_gate_invert', action='store_true', default=False)

    # Non-Propagating Critique residual: frozen backbone + non-propagating residual head (npc_alpha scales it).
    parse.add_argument('--npc_residual', action='store_true', default=False)
    parse.add_argument('--npc_full', action='store_true', default=False)
    parse.add_argument('--npc_prop', action='store_true', default=False)
    parse.add_argument('--npc_prop_neg', action='store_true', default=False)
    parse.add_argument('--npc_alpha', type=float, default=1.0,
                       help='非传播 residual head 权重（residual 零初始化，α 控制校正尺度）')
    parse.add_argument('--npc_alpha_neg', type=float, default=1.0)

    # residual-LLM multi-task term: train the non-propagating residual head to fit the continuous LLM signal s_L.
    # Unified pool pos|neg, sign-coded s_L (pos->+score / neg->-score) => pos/neg share one loss by construction.
    parse.add_argument('--rllm_mode', type=str, default='none', choices=['none', 'reg', 'pair'],
                       help='residual-LLM 项：none=npc_full基线(只BPR+CL)/reg=形式A回归/pair=形式B pairwise')
    parse.add_argument('--rllm_lambda', type=float, default=1.0,
                       help='residual-LLM 项权重（多任务融合系数）')
    parse.add_argument('--rllm_tau', type=float, default=0.05,
                       help='形式B pair tie 阈值：|s_L(i)-s_L(j)|>τ 才计入序对')
    # Form B-v2 improvements (prediction-space supervision / soft label / polarity stratification).
    parse.add_argument('--rllm_pred_space', action='store_true', default=False,
                       help='①预测空间监督：score 用 ũ_u·ĥ（含冻结bb项）而非原始 r·ĥ。'
                            '修过平滑（训练/使用空间一致）+ r 只校正 bb 排错的 pair（后验校正 δ 语义）。')
    parse.add_argument('--rllm_beta', type=float, default=0.0,
                       help='②软标签温度：target p_ij=σ(β·(s_i-s_j))。0=硬BPR；>0(如8)=软BCE鲁棒误排。')
    parse.add_argument('--rllm_within', type=float, default=1.0,
                       help='③同极性 pair 权重 κ：跨极性(pos vs neg)权重1/同极性(pos-pos,neg-neg)权重κ。'
                            '1.0=不分层；<1(如0.2)压制最噪的同极性细序。')
    parse.add_argument('--rllm_pool_size', type=int, default=60,
                       help='统一 critique 池每用户最大 item 数（pos∪neg 合并后截断）')
    parse.add_argument('--rllm_propagate', action='store_true', default=False,
                       help='残差传播式（npc_prop 族）：residual 烘焙进 user emb 后经 GCN 二次传播（backbone 仍冻结）。'
                            '默认关=非传播(npc_full 族，已知弱)；开=传播式(+53%% headroom 所在)，residual-LLM 忠实测试。')
    parse.add_argument('--rllm_polarity', action='store_true', default=False,
                       help='极性感知传播式（npc_prop_neg 族，2026-07-03）：prop head(critique_residual)烘焙+传播'
                            '收 pos/CL；neg head(critique_residual_neg)post-aggregate 非传播收 neg（切 D6）。'
                            'v2 pair loss 按极性路由：pos 成员 score 用传播表示(放大)/neg 用非传播(隔离)。'
                            '⓵ 架构依赖传播（GCN）；通用性见 residual-LLM v2 非传播（--rllm_pred_space，架构无关）。')
    parse.add_argument('--rllm_alpha_neg', type=float, default=0.3,
                       help='rllm_polarity 的 neg 非传播 head 权重（解耦：prop head 用 npc_alpha 全强度传，'
                            'neg head 用小 α 释放槽不放大 D6；npc_prop_neg 峰值 α_neg≈0.1-0.3）')

    # D-3: independent L2 term lambda_reg*||delta||^2 on the residual head (v2 Eq(19)).
    parse.add_argument('--delta_reg_lambda', type=float, default=0.0,
                       help='D-3: residual head 的 L2 正则 λ_reg（v2 Eq(19) λ_reg‖δ_u‖²）。0=关=byte-identical。'
                            '极性感知模式同时正则 prop/neg 两个 head。')
    # D-2: three-bucket CER ablation (v2 §2.6 Eq(18)), controlled at the sampling layer.
    parse.add_argument('--no_cer_random_ref', action='store_true', default=False,
                       help='D-2: CER 负样本剔除随机参考桶 Bʳᵉᶠ，仅留 past-critique B⁻（EXP-A05 消融）。默认关。')
    parse.add_argument('--no_cer_critique_replay', action='store_true', default=False,
                       help='D-2: CER 负样本剔除 past-critique 桶 B⁻，仅留随机参考 Bʳᵉᶠ（EXP-A05 消融）。默认关。')

    # s_L-margin CER: fold the L_pair LLM ordering into the CER contrastive structure (additive margin m*g on B-).
    parse.add_argument('--use_cer_margin', action='store_true', default=False,
                       help='CL loss 换成 s_L-margin CER（get_cer_margin_loss）：B⁻=LLM dislike 带 m·g margin。'
                            '默认关=旧 InfoNCE（byte-identical）。')
    parse.add_argument('--cer_margin', type=float, default=0.5,
                       help='s_L margin 尺度 m（dislike 负 logit 加 m·g_n）。score~0.05 量级，配合 cer_tau。')
    parse.add_argument('--cer_tau', type=float, default=0.1,
                       help='margin-CER 对比温度 τ（放对比锐度；0.1 让 m·g 在 1000 随机负中有可见贡献）。')
    parse.add_argument('--cer_gamma', type=float, default=1.0,
                       help='dislike 程度锐度 γ（g=σ(γℓ)）。neg_scores mean0.74 → γ=1 保 g∈[0.5,0.73] 有区分；'
                            'γ=8 饱和无区分（复用门控 γ 会失效）。')
    parse.add_argument('--cer_cri_neg_k', type=int, default=20,
                       help='B⁻ 桶大小 Kc（每用户采样 LLM dislike 数）。')

    # P: gate-aware L_cri training target (CF x LLM co-action): score neg on the gated score.
    parse.add_argument('--lcri_gate_aware', action='store_true', default=False,
                       help='P: L_cri(neg BPR) 在门控后打分上算（neg_score − gate_pen），让 δ_u 训练看到 LLM 门控。'
                            '默认关=旧纯 CF 打分（byte-identical）。')
    parse.add_argument('--lcri_gate_M', type=float, default=1.0,
                       help='P: 训练门控惩罚尺度（gate_pen=lcri_gate_M·σ(γ·ℓ)）。neg_score 量级~0.05，'
                            'M=1 中等梯度抑制 / M=3 强抑制 / M=10 近归零（=完全交给推理门控）。')

    # Semantic-Offset L_cri: pairwise joint-margin BPR (CL anchor vs critique dislike, joint score s_delta+q).
    parse.add_argument('--use_semantic_offset_in_lcri', action='store_true', default=False,
                       help='Semantic-Offset L_cri（文档§3.2）：pointwise neg_loss 换成 pairwise 联合 margin BPR'
                            '（CL 正锚 vs critique dislike，joint score s_δ+q）。默认关=旧 pointwise L_cri（byte-identical）。')
    parse.add_argument('--so_M', type=float, default=1.0,
                       help='SO: 训练侧语义偏移尺度 M（q=-M·σ(γ·ℓ)）。小 M（0.5）=软条件化温和降权；'
                            '大 M（≥4）≈硬移除 gated pair（SO 退化为只修 gate-miss，≈A3）。与推理 integ_M(=10) 独立。')
    parse.add_argument('--cri_temperature', type=float, default=1.0,
                       help='SO: 联合 margin 温度 τ_cri（L=-log σ(Δ_joint/τ)）。文档§6 先扫 {0.5,1,2,4} 解决 BPR 饱和。')

    # Unified CCR (v0.5.0 §4.3, M-DEC-003): set-wise Z+/Z- contrast with joint training score z=s_delta+q.
    parse.add_argument('--use_ccr', action='store_true', default=False,
                       help='Unified CCR（方法论§4.3）：neg_loss+cl_loss 换成单一集合式 CCR（Z+/Z-）。默认关=旧 InfoNCE+pointwise。')
    parse.add_argument('--ccr_joint_score', action='store_true', default=False,
                       help='U2: CCR 的 critique dislike 桶用联合打分 z=s_δ+q（q<0 降权）；不开=U1(纯 s_δ 无 q)。')
    parse.add_argument('--ccr_tau', type=float, default=1.0,
                       help='CCR 对比温度 τ（Z+/Z- softmax 锐度）。')
    parse.add_argument('--ccr_M', type=float, default=1.0,
                       help='CCR 训练侧语义偏移尺度 M（q=-ccr_M·σ(γℓ)）。与推理 integ_M 独立；加性门跑时设 integ_M=ccr_M 保一致。')
    parse.add_argument('--integ_additive', action='store_true', default=False,
                       help='推理用【加性门】（方法论 M-DEC-002/003 忠实）：logit += q=-integ_M·σ(γℓ)，即 z=s_δ+q。'
                            '默认关=乘法硬/软门（_apply_gate）。q 加在 pre-sigmoid logit 空间。')

    # (a) Orthogonal protection + (b) CL hard-neg from critique: keep the residual from disturbing positives.
    parse.add_argument('--prot_lambda', type=float, default=0.0,
                       help='(a) 正交保护：L_prot=mean(⟨δ_u,φ̄_pos⟩²)，强制 residual 在 train 正例均值方向投影→0，'
                            '不扰动正例方向（救 path A 的 -11%）。默认 0=关（byte-identical 旧行为）。')
    parse.add_argument('--cl_hardneg_critique', action='store_true', default=False,
                       help='(b) CL 对比负样本改用用户 neg critique 作 hard neg（随机:critique=1:10，'
                            '随机少量保 embedding uniform；critique 全部作 hard neg，不足随机补）。'
                            '默认关：CL neg 仍走旧 99% 随机 + 1% bpr_history。')

    # post-hoc x residual integration: frozen-graph residual + post-hoc soft gate (item-side FP demotion).
    parse.add_argument('--integ_M', type=float, default=10.0,
                       help='post-hoc 软门压降幅度 M（s_final=s_res−M·g；post-hoc 实测 Δ∈{0.2..100}不敏感，10 中值）')
    parse.add_argument('--integ_gamma', type=float, default=8.0,
                       help='软门锐度 γ（g=σ(γ·ℓ)·𝟙[ℓ>0]；γ→∞=硬门；8 复用 v2 soft-label 经验）')
    parse.add_argument('--integ_hard_inference', action='store_true', default=True,
                       help='推理用硬门（FP 全压 M）；默认开。--no-integ_hard_inference 切软门')
    parse.add_argument('--integ_smooth', type=float, default=0.0,
                       help='residual ‖δ‖² 正则（强迫只学可投影主成分）；默认 0=关')
    parse.add_argument('--integ_no_hard_inf', action='store_true', default=False,
                       help='关闭推理硬门，改软门（g=σ(γ·ℓ)）')
    parse.add_argument('--integ_exclude_cffp', action='store_true', default=False,
                       help='从门控 FP 集剔除 CF-FP（泄露源，见 memory cf-fp-is-leakage），只留 LLM-dislike+锚（可部署）')
    parse.add_argument('--integ_cffp_file', type=str, default=None,
                       help='CF-FP 文件（simulate_fp_*.json），--integ_exclude_cffp 时用于剔除')
    parse.add_argument('--integ_fp_file', type=str, default=None,
                       help='直接从 constrained LLM-dislike json 加载门控 FP 集（彻底绕开 union池/CF-FP）。'
                            '格式 {user_str: [[item_id, score], ...]}（generate_constrained_dislike_659 产物）。'
                            '开启时 build_user_fp 忽略 critique_round_path，FP 源 = 纯 constrained LLM-dislike（0泄露可部署）')
    parse.add_argument('--integ_incremental_gate', action='store_true', default=False,
                       help='逐轮增量门控：round r 评测时只用 round 0..r 的累积 FP 减分 → 凸升曲线（baseline=纯bb无门控）。'
                            '默认关=全量静态门控（旧行为，所有轮用同一全量 FP 集，峰在 r0/1）。'
                            '门控 schedule 由 --integ_gate_schedule 决定。')
    parse.add_argument('--integ_gate_schedule', type=str, default='strength',
                       choices=['strength', 'round_path'],
                       help='逐轮增量门控的 FP 切分策略（仅 --integ_incremental_gate 开时生效）：'
                            'strength=constrained dislike 按 score 降序均分 R 轮（round r=前 (r+1)/R 最强 FP，保高排强结果）；'
                            'round_path=critique_round_path 的 test_{0..r}_neg.txt 累积（真逐轮 LLM dislike，非高排）。')

    parse.add_argument('--out', type=str, default=None)
    parse.add_argument('--critique_round_path', type=str, default='./experiments/top200/critique_round/')
    parse.add_argument('--backbone', type=str, default=None,
                       help='backbone 权重路径覆盖；默认 model_save/best_model_LightCCF.pth')
    parse.add_argument('--save_name', type=str, default=None)

    return parse.parse_args()
