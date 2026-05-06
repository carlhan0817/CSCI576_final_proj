start file
usage of each folder:
    test:the unit test for the app and the testing video
    prompts:save all documentation markdown file here
    app: main program
    data: temp folder for the new generateed data


1. Set Up the Environment
Since you will likely be managing this in VSCode, setting up your Python environment and kernel correctly from day one will save your team a lot of headaches.

Initialize the Virtual Environment: Run python -m venv .venv in your root folder. Make sure your VSCode interpreter is pointed to this specific .venv.

**I run python -m venv .venv under src folder.**

Create pyproject.toml: Instead of a messy requirements.txt, define your dependencies here. You will need to install:

Core: numpy, pandas, pydantic

Media: opencv-python, librosa, ffmpeg-python (or imageio-ffmpeg)

AI Models: openai-whisper, open_clip_torch, sentence-transformers, torch

Backend: fastapi, uvicorn

**Now open terminal in VSCode and cd into CSCI576_final_proj folder, run the following command in order:**
python -m pip install --upgrade pip setuptools wheel
pip install -e .    



## Phase 1: Ingestion

Process a single MP4 into cached artifacts under `workspace/<video_stem>/`:

```bash
# Activate venv first
python -m backend.pipeline.ingest path/to/video.mp4
```

Optional flags:
- `--workspace DIR` — override workspace root (default: `./workspace`)
- `--model NAME` — faster-whisper model: `tiny`, `base`, `small`, `medium` (default: `base`)
- `--device {auto,cpu,cuda}` — inference device (default: `auto`)
- `--force` — ignore cache and re-run all stages

Artifacts produced:
- `workspace/<stem>/meta_raw.json` — video metadata
- `workspace/<stem>/audio_processed.wav` — mono 16 kHz PCM
- `workspace/<stem>/frames_cache/frame_XXXXXX.jpg` — 1 FPS, longest-side 512 px
- `workspace/<stem>/transcript.json` — sentence-level transcript with timestamps
- `workspace/<stem>/ingest.log` — run log

### Running tests

```bash
pip install -e ".[dev]"
pytest -v -m "not slow"       # fast suite (~seconds)
pytest -v -m slow             # real-Whisper test (downloads ~140 MB first run)
```

## Phase 4.5: Player

Run the player after a video has been ingested + analyzed end-to-end (Phases 1–3 produce `workspace/<stem>/metadata.json`).

```bash
# 1. Drop the source MP4 into ./videos/<stem>.mp4 (must match the workspace stem)
# 2. Run the server
uvicorn backend.server.app:app --port 8000

# 3. Open http://localhost:8000/ in Chrome
```

Endpoints:
- `GET  /api/videos`              — list analyzed videos
- `GET  /api/metadata/{video_id}` — read metadata.json
- `POST /api/metadata/{video_id}` — write metadata.json (full replacement; server forces `verified_by_human=true`)
- `GET  /videos/<stem>.mp4`       — static MP4 (browser uses Range for seek)

Tests: `pytest tests/test_server.py -v`

---

# 本分支 (ocrapproach) 改动总览

聚焦于把 ad detection 从 baseline (P=40%, R=60%, F1=48%) 提升到 P=91%, R=99%, F1=95%。改动分布在 Phase 2 特征 + Phase 3 融合，外加一组跨平台 bug 修复。

## Phase 2 — 新增特征通道

- **OCR + 商业模式检测**（`features/visual.py`, `features/ocr_signals.py`）
  - rapidocr-onnxruntime 在采样帧上跑 OCR（stride=3s + 黑帧/低方差跳过）
  - 正则匹配 URL / 价格 / 电话 / CTA / brand-lockup → `has_url`, `has_price`, `has_phone`, `has_cta`, `has_brand_lockup`
- **CLIP 池化 image embedding**（`features/visual.py`）：除 zero-shot 概率外额外输出 512-d 向量，给 style_drift 用
- **每秒 MFCC**（`features/audio.py`）：20 维 MFCC 向量，给 audio_drift 用

## Phase 3 — 融合阶段升级

- **`fusion/style_drift.py`**：基于 CLIP embedding 的视觉风格邻域散度
- **audio_drift**（`fusion/align.py`）：用 MFCC 算同样的音频风格漂移
- **`rule_ad_block`**（`fusion/rules.py`）：多信号广告检测，要求 `(visual_drift OR audio_drift) ≥ 0.30` 且区段内出现至少一个商业信号；带 hard-cut 边界则置信度 0.9，否则 0.7
- **`rule_ad_break` 降级**：从原 0.85 降为 0.5 advisory，避免 is_speech-only 单信号产生大量误报
- **`rule_low_energy_audio`**（新，本 session 加入）：RMS<0.18 + spectral_bandwidth<1900 持续 ≥10s → sponsorship。专门捕获 VAD 误判为 speech 的 rap / song 类广告（test_004 上 Ad1 rap 和 Ad3 song 都靠这条规则命中）
- **CLIP→sponsorship cut-density gate**（`fusion/classify.py`，本 session 加入）：CLIP fallback 把段标成 sponsorship 时，要求段内 hard_cut 密度 ≥ 0.25/sec（剔除 boundary 处的 cut，避免 boundary 自身被算进密度）。低于阈值降级 core_content。消除 lecture 中静态 slide 被 CLIP 误判为 "advertisement slide" 产生的大块 FP

## 跨平台 Bug 修复（本 session）

- **强制 UTF-8 编码**（`probe.py`, `transcribe.py`, `features/{audio,text,visual}.py`, `fusion/{align,run,export}.py`, `server/routes.py`）：所有 JSON 读写显式 `encoding="utf-8"`。修复中文 locale Windows 上 OCR 输出含 `•` (U+2022) 等字符时 `Path.write_text` 默认走 GBK 编码导致整 pipeline 崩溃的问题

## 在 test_004 (20 min, 3 ads, GT 总长 136s) 上的效果

| 指标 | 改前 | 改后 | Δ |
|---|---|---|---|
| Precision | 40.0% | **91.3%** | +51.3pp |
| Recall | 60.0% | **98.6%** | +38.6pp |
| F1 | 47.8% | **94.8%** | +47.0pp |
| 总段数 | 28 | **18** | 更连贯 |
| Ad1 (rap) 覆盖 | 10% | **100%** | |
| Ad2 (animation) 覆盖 | 100% | 100% | |
| Ad3 (song) 覆盖 | 55%（碎成 8 段） | **100%（1 段）** | |

## 仅重跑 Phase 3 的快速循环

修改 `fusion/*.py` 后不需要重跑 30+ 分钟的 Phase 1+2：

```bash
python -m backend.pipeline.fusion.run videos/<stem>.mp4   # 几秒
```

`metadata.json` 立刻更新，前端刷新即可看到新结果。

---

# 调参方式
  一、整体架构（分段是怎么产生的）                                                        

  Phase 2 特征提取 → Phase 3 融合分段                                                                                                          
    ├─ visual.py    （视觉切点 + CLIP 场景概率）
    ├─ audio.py     （speech/music/silence 分类）                                                                                              
    └─ text.py      （文本相似度 + 关键词匹配）
          ↓
    ├─ rules.py        硬规则 → 高置信度标签（dead_air / sponsorship / intro / ...）
    ├─ boundaries.py   候选边界（融合所有"变化"信号）
    ├─ classify.py     每个 [边界, 边界) 区间分类
    └─ smooth.py       合并 + 吸收短段

  调段数靠 boundaries.py + smooth.py；调标签靠 rules.py + classify.py；改"什么算 speech / 切点"靠 features/*。

  ---
  二、按症状对照表

  症状 A：分段被切得太碎（一堆短段）

  ┌──────────────────────┬──────┬─────────────────────────┬──────────────────────────────────────────┐
  │         参数         │ 默认 │          文件           │                调大或调小                │
  ├──────────────────────┼──────┼─────────────────────────┼──────────────────────────────────────────┤
  │ MIN_BOUNDARY_GAP_SEC │ 3    │ fusion/boundaries.py:18 │ ↑ 增大 → 边界互相挤压被丢弃，段更长      │
  ├──────────────────────┼──────┼─────────────────────────┼──────────────────────────────────────────┤
  │ MIN_SEGMENT_DURATION │ 2.0  │ fusion/smooth.py:12     │ ↑ 增大 → 短于该阈值的段会被并入邻居      │
  ├──────────────────────┼──────┼─────────────────────────┼──────────────────────────────────────────┤
  │ HIST_DIFF_THRESHOLD  │ 0.4  │ fusion/boundaries.py:14 │ ↑ 增大 → 视觉切点更难触发                │
  ├──────────────────────┼──────┼─────────────────────────┼──────────────────────────────────────────┤
  │ CLIP_KL_THRESHOLD    │ 0.3  │ fusion/boundaries.py:15 │ ↑ 增大 → CLIP 场景必须变得更剧烈才算边界 │
  ├──────────────────────┼──────┼─────────────────────────┼──────────────────────────────────────────┤
  │ TEXT_SIM_THRESHOLD   │ 0.35 │ fusion/boundaries.py:16 │ ↓ 减小 → 文本必须更不相似才算话题切换    │
  └──────────────────────┴──────┴─────────────────────────┴──────────────────────────────────────────┘

  症状 B：该分的地方没分（段太长／粘连）

  反过来调上面那五个：减小 MIN_BOUNDARY_GAP_SEC / MIN_SEGMENT_DURATION / HIST_DIFF_THRESHOLD / CLIP_KL_THRESHOLD；增大 TEXT_SIM_THRESHOLD。

  症状 C：sponsorship / intro / outro 漏判或误判

  这块完全由关键词表 + 规则窗口驱动：

  ┌───────────────────────────────────────────┬───────────────────────────────────────────┬────────────────────────────────────────────────┐
  │                   参数                    │                   文件                    │                      说明                      │
  ├───────────────────────────────────────────┼───────────────────────────────────────────┼────────────────────────────────────────────────┤
  │ SPONSOR_KEYWORDS                          │ fusion/rules.py:27                        │ 加你视频里实际出现的赞助话术（如 "check the    │
  │                                           │                                           │ description"、品牌名）                         │
  ├───────────────────────────────────────────┼───────────────────────────────────────────┼────────────────────────────────────────────────┤
  │ INTRO_KEYWORDS / OUTRO_KEYWORDS /         │ fusion/rules.py:32-48                     │ 同上                                           │
  │ SELF_PROMO_KEYWORDS / RECAP_KEYWORDS      │                                           │                                                │
  ├───────────────────────────────────────────┼───────────────────────────────────────────┼────────────────────────────────────────────────┤
  │ INTRO_WINDOW_SEC / OUTRO_WINDOW_SEC       │ fusion/rules.py:55-56                     │ 默认 90s，长视频可调到 120-180                 │
  ├───────────────────────────────────────────┼───────────────────────────────────────────┼────────────────────────────────────────────────┤
  │ sponsor 命中后的 ±15s 扩展                │ fusion/rules.py:105-106                   │ 决定 sponsor 段从关键词出现前后多远开始/结束   │
  ├───────────────────────────────────────────┼───────────────────────────────────────────┼────────────────────────────────────────────────┤
  │ recap 的 +30s 扩展                        │ fusion/rules.py:128                       │ recap 段比关键词晚结束多少秒                   │
  ├───────────────────────────────────────────┼───────────────────────────────────────────┼────────────────────────────────────────────────┤
  │ 各规则的 confidence                       │ fusion/rules.py:84,95,107,118,129,142,156 │ 多规则同时命中时，谁 confidence 高谁赢（见下方 │
  │                                           │                                           │  classify）                                    │
  └───────────────────────────────────────────┴───────────────────────────────────────────┴────────────────────────────────────────────────┘

  ▎ ⚠️  关键词表是单一真源——features/text.py 从同一份列表里抓 matched_keywords，再被 rules.py 复用（text.py:16-22）。所以你只在 rules.py 
  ▎ 改一处即可。但改完要删掉旧的 features/text_features.json 重跑 Phase 2，否则缓存里没有新关键词。

  症状 D：同一段内多条规则打架，标错了

  fusion/classify.py:74 里的 coverage_threshold=0.5：规则必须覆盖该段 ≥50% 时间才"算数"。
  - 调小 → 规则更容易接管标签（更激进）
  - 调大 → 规则更保守，让 CLIP/音频 fallback 接手

  打架时按规则的 confidence 取最高（classify.py:88）。如果你想让 sponsor 永远压过 intro，改 rules.py 里两者的 confidence 数值。

  症状 E：核心内容被误判成 filler / transition / 反过来

  fusion/classify.py:118-119：
  if speech_ratio < 0.1 and mapped_label == "core_content":
      mapped_label = "filler"
  这是唯一把 core_content 改判 filler 的硬触发。把 0.1 调低（如 0.05）= 更宽容，更多段保留 core_content。

  CLIP → label 的映射表本身在 classify.py:25-39 —— 想新增类别（比如把 "video game" 映射成自定义类别），改这里。

  症状 F：dead_air / holding_screen 误触发

  fusion/rules.py:52-54:
  DEAD_AIR_MIN_DURATION = 5
  DEAD_AIR_RMS_THRESHOLD = 0.01
  HOLDING_SCREEN_MIN_DURATION = 8
  - rules.py:92 的 hist_diff < 0.05（holding screen 的"画面静止"判定）。

  症状 G：声音类别判错（speech 被当 music 之类）

  features/audio.py:13-15：
  SILENCE_RMS_THRESHOLD = 0.001
  SPEECH_ZCR_THRESHOLD = 0.10
  MUSIC_RMS_THRESHOLD = 0.01
  判定顺序在 audio.py:29-45（低 RMS→silence；高 ZCR→speech；中等 RMS→music；其余→noise）。

  症状 H：视觉切点过敏 / 不敏感

  features/visual.py:15-17：
  DEFAULT_CUT_THRESHOLD = 0.6
  BLACK_FRAME_LUMINANCE_MAX = 10.0
  BLACK_FRAME_VARIANCE_MAX = 50.0
  SCENE_LABELS（visual.py:22-37）是 CLIP 用的 10 个 prompt——你也可以直接加新场景类别，但加完要同步更新 classify.py:25 的映射表。

  ---
  三、改完之后必须做的事（重要）

  Pipeline 有缓存：每一阶段都会检查输出文件是否存在，存在就 skip（如 features/audio.py:105-107）。

  所以改了：
  - features/* 的参数 → 删 workspace/{video_id}/features/*.json 重跑 Phase 2+3
  - fusion/* 的参数 → 删 workspace/{video_id}/metadata.json 重跑 Phase 3 即可

  或者用 run_pipeline(..., force=True) 强制全部重跑。

  ---
  四、推荐的调参顺序

  我的建议是自顶向下：
  1. 先看 metadata.json 里每段的 evidence.triggered_rules 和 confidence——如果误标段有规则触发，调 rules.py；没有规则触发，调 classify.py 的
  CLIP 映射或 speech_ratio 阈值。
  2. 段数对不上？先动 MIN_SEGMENT_DURATION 和 MIN_BOUNDARY_GAP_SEC，这俩最便宜（只要重跑 Phase 3）。
  3. 上面都不够，再去动 boundaries.py 的四个信号阈值。
  4. 实在不行，回到 Phase 2 调 audio / visual 阈值（这层改动重跑代价最大）。
