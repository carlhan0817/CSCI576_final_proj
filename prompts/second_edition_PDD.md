# 项目设计文档 (PDD)：长视频多模态内容分割系统 (V2.1)

**课程**: CSCI 576 Multimedia (Spring 2026)
**演示日期**: 2026-05-06 / 05-07 / 05-08
**团队规模**: 3 人
**文档版本**: V2.1 (基于 V1.0 修订，**[新增补充]** 加入了课内基础理论、人工验证机制及跨集检测)

---

## 1. 项目目的 (Purpose)

### 1.1 问题背景
现代长视频（讲座、播客、YouTube 视频、TED、动画、新闻等）中混杂着大量**非核心内容 (Non-Content)**：片头、片尾、赞助广告、频道自我宣传、回顾、过场画面、倒计时 "starting soon" 画面、死气沉沉的沉默段、无关插入等。这些片段打断观看体验、降低可达性、让长视频存档难以导航。

### 1.2 项目目标
构建一个**端到端的离线多模态分析系统**，对任意单个 MP4 输入视频进行分析，自动生成一份描述其内部结构的**内容地图 (Content Map)**，并通过一个自定义的 Web 播放器将该地图可视化、可交互地呈现给用户，以便用户快速跳过 / 仅播放 / 标记他不感兴趣的段落。

### 1.3 约束与范围
| 约束项 | 决定 |
|---|---|
| 输入 | 单个 MP4 文件 (每次处理一个，不考虑批处理)。 **[新增补充]**：但系统支持加载轻量级外部“特征字典”以辅助实现跨集重复内容的比对。 |
| 运行环境 | 完全离线 (所有模型本地化) |
| 用途 | 课程演示项目，非商业级部署；可为性能做合理简化 |
| 可用硬件 | 本地机器 (可使用 GPU，也须能在 CPU 下退化运行) |
| 视频类型 | **不假设特定类型**：系统必须对动画、讲座、TED、播客、YouTube 多样内容通用 |
| Python 环境 | 统一使用 `.venv`，依赖通过 `pyproject.toml` 管理 |

### 1.4 本项目"多模态"的意义
非核心内容没有单一特征 —— 赞助段可能视觉上与正片无异但文本里有 "sponsored by"；片头可能视觉非常特殊但声音平常；死气段在视觉和语义上都平静但音频能量极低。因此**必须融合三种模态**才能稳健识别：
* **视觉** (颜色结构、镜头切换、运动、黑帧)
* **音频** (能量包络、语音/音乐/静音分布)
* **文本** (ASR 转录出的语义内容、关键词)

---

## 2. 输出格式规范 (Output Format Specification)

### 2.1 核心交付物
系统的最终输出是一份 `metadata.json` 文件 —— 它是**离线分析阶段的终点**，也是**在线播放阶段的唯一数据来源**。前端播放器只读取这份 JSON 与原始 MP4，不做任何重复分析。

### 2.2 `metadata.json` 结构定义
```
metadata.json
├── video_info               视频基本元信息
│   ├── filename             源 MP4 文件名
│   ├── duration_sec         总时长 (秒)
│   ├── fps                  原始帧率
│   ├── width / height       分辨率
│   ├── analysis_version     分析管线版本号
│   └── verified_by_human    [新增补充] 是否经过人工审阅确认 (boolean)
│
├── segments[]               按时间顺序排列的片段列表 (核心字段)
│   ├── segment_id           片段唯一 ID (整数，0 起始)
│   ├── start_sec            开始时间戳 (float, 秒)
│   ├── end_sec              结束时间戳 (float, 秒)
│   ├── label                片段类别 (见 §2.3 分类体系)
│   ├── confidence           置信度 (0.0 ~ 1.0)
│   ├── evidence             模态证据 (解释为什么给这个标签)
│   │   ├── visual_score     视觉模态对该标签的贡献分
│   │   ├── audio_score      音频模态对该标签的贡献分
│   │   ├── text_score       文本模态对该标签的贡献分
│   │   └── triggered_rules  触发的硬规则列表 (如 "black_frame", "silence", "sponsor_keyword")
│   ├── summary              片段简短文本摘要 (1-2 句, 从该片段的转录中提取)
│   └── user_corrected       [新增补充] 人工是否在前端修正过此片段 (boolean)
│
├── chapters[]               章节标记 (合并相邻同类 core_content 后的粗粒度结构)
│   ├── chapter_id
│   ├── start_sec / end_sec
│   └── title                自动生成的章节标题
│
└── skip_suggestions[]       推荐跳过的片段 ID 列表 (all non-content segments)
```
### 2.3 分类体系 (Taxonomy)
采用课程文档建议的**多类分类**，共 10 个类别：

| label 值 | 含义 |
|---|---|
| `core_content` | 主要内容 (用户真正想看的部分) |
| `intro` | 片头 |
| `outro` | 片尾 |
| `sponsorship` | 赞助 / 广告 |
| `self_promotion` | 自我推广 / 频道宣传 |
| `recap` | 回顾 / 重复样板 |
| `transition` | 过场 / 过渡画面 |
| `dead_air` | 死气 / 长时间沉默 |
| `holding_screen` | 倒计时 / 等待画面 / "Starting soon" |
| `filler` | 无关插入 / 填充内容 |

### 2.4 输出格式如何映射到课程要求

| 课程 PDF 要求 | 在 metadata.json 中的体现 |
|---|---|
| "a timeline with labeled segments" | `segments[]` 数组，含 start/end/label |
| "chapter markers" | `chapters[]` 数组 |
| "skip suggestions" | `skip_suggestions[]` 数组 |
| "summaries of segment types" | 每个 segment 的 `summary` 字段 |
| "Where does the actual content begin?" | 第一个 `label == "core_content"` 的片段 `start_sec` |
| "Which intervals are likely sponsorships?" | `segments[]` 中所有 `label == "sponsorship"` 的片段 |
| "Which parts are repeated across episodes?" | `recap` 标签 (如实现跨视频比较则扩展)。**[新增补充] 通过比对外部特征字典实现跨集判定。** |
| "a cleaned playback interface" | 前端播放器基于 `skip_suggestions[]` 自动跳过 |
| "multimodal reasoning" 证据 | 每个 segment 的 `evidence` 字段 |
| **[新增补充]** "human verification" | **配合前端编辑模式，使用 `verified_by_human` 和 `user_corrected` 字段。** |

### 2.5 分类一致性保证
所有判定必须**显式记录证据**到 `evidence` 字段。团队在演示时可据此向评审说明分类理由，满足课程要求的"definitions must be explicit and consistently applied"。

---

## 3. 系统总体架构 (System Architecture)

系统采用 **"离线分析 + 在线播放"** 的两段式架构：
```
┌─────────────────────────────────────────────────────────┐
│  离线分析管线 (Offline Pipeline, 一次性运行)              │
│                                                         │
│  MP4 → [P1 摄入] → [P2 特征] → [P3 融合] → metadata.json │
│                                                         │
└─────────────────────────────────────────────────────────┘
↓
metadata.json + 原始 MP4
↓
┌─────────────────────────────────────────────────────────┐
│  在线播放服务 (Online Playback, FastAPI + HTML)          │
│                                                         │
│  FastAPI: 静态文件服务 (MP4) + JSON API (metadata 读写)   │
│  HTML/JS 前端: 视频播放器 + 交互式时间轴 + 跳过控制       │
│                                                         │
└─────────────────────────────────────────────────────────┘
### 3.1 关键架构决策

| 决策 | 选择 | 理由 |
|---|---|---|
| 分析与播放解耦 | 离线生成 JSON，在线只读 (及部分验证写回) | 分析耗时大 (可能几分钟)，播放必须流畅；解耦后演示稳定 |
| 融合策略 | **规则引擎 + 预训练多模态嵌入 + 轻量分类器** (Plan C) | 无需标注数据；在多种视频类型上泛化；完全离线 |
| 前端形态 | FastAPI + HTML/JS (非桌面 GUI)。**[技术升级]：推荐引入 Vue.js 管理播放状态，D3.js 绘制专业时间轴。** | 跨平台演示方便，浏览器原生支持 MP4 Range 请求 |
| MP4 服务方式 | 静态文件 | 浏览器 `<video>` 原生支持 Range seek，无需自实现 |
| 模型下载 | 首次运行时下载到本地缓存目录 | 演示时完全离线可用 |

### 3.2 模型清单与资源预算

| 模型 | 用途 | 磁盘 | 峰值 VRAM | CPU 回退 |
|---|---|---|---|---|
| Whisper `base` | 语音转录 + 时间戳 | ~140 MB | ~1 GB | 可 |
| CLIP ViT-B/32 (open_clip) | 视觉零样本分类 | ~340 MB | ~1 GB | 可 |
| sentence-transformers `all-MiniLM-L6-v2` | 文本语义嵌入 | ~90 MB | ~0.3 GB | 可 |
| Silero VAD | 语音活动检测 | ~2 MB | 可忽略 | 可 |
| **合计** | — | **~570 MB** | **~2 GB (模型按阶段加载，不共存)** | — |

**资源策略**：模型按阶段**懒加载**，每阶段用完即释放显存。任何 4 GB VRAM 的笔记本均可流畅运行，CPU-only 机器可降级运行。

### 3.3 目录与依赖管理
```
Final_proj/
├── .venv/                           统一的 Python 虚拟环境 (根目录)
├── pyproject.toml                   依赖声明
├── backend/
│   ├── pipeline/                    离线分析管线 (§4)
│   └── server/                      FastAPI 服务 (§4.5)
├── frontend/                        HTML/JS 播放器 (§4.5)
├── workspace/                       运行时缓存 (每个视频一个子目录)
│   └── <video_name>/
│       ├── audio_processed.wav
│       ├── transcript.json
│       ├── frames_cache/
│       ├── features/                特征 npy/json
│       └── metadata.json            最终产物
└── videos/                          用户放入待分析的 MP4
```
所有后端代码与脚本**必须**在 `.venv` 激活下运行。依赖通过 `pyproject.toml` 管理 (而非 `requirements.txt`)。

---

## 4. 分阶段详解 (Phases in Detail)

### 4.1 阶段一：数据摄入与预分离 (Data Ingestion & Demuxing)

#### 目的
将单个 MP4 拆解为三种**时间对齐的原始数据流**，供后续特征阶段读取；同时建立缓存机制，避免重复解码。

#### 输入 / 输出
- **输入**: 单个 MP4 文件路径
- **输出**:
  - `workspace/<video>/audio_processed.wav` — 单声道 16kHz WAV
  - `workspace/<video>/transcript.json` — 带 start/end/text 的句子级转录
  - `workspace/<video>/frames_cache/frame_XXXXXX.jpg` — 固定采样率关键帧
  - `workspace/<video>/meta_raw.json` — 时长、fps、分辨率等基础信息

#### 子任务
1. **容器探测** — 读取 MP4 时长、fps、分辨率、流信息
2. **音轨提取** — 抽出音频轨，转为单声道 16kHz WAV (Whisper 与 librosa 的标准输入格式)
3. **语音转录** — 对 WAV 跑 Whisper，生成句子级时间戳转录
4. **视觉帧采样** — 按 1 FPS (或 2 FPS) 抽帧存盘；文件名编码时间戳便于反查
5. **缓存落盘** — 以上产物全部写入 `workspace/<video>/`，下次跳过

#### 所需 Python 技术
| 用途 | 库 |
|---|---|
| 音视频容器解析、抽帧、抽音轨 | `ffmpeg-python` (包装 FFmpeg) 或直接 `subprocess` 调 ffmpeg |
| 语音转录 | `openai-whisper` (本地运行) |
| 视频元信息 | `ffmpeg-python.probe()` 或 `opencv-python` |
| 文件 I/O | 标准库 `json`, `pathlib` |

#### 关键设计原则
- **为什么 1 FPS 抽帧**：长视频几十万帧全量读取会爆内存；非内容片段 (片头/广告) 的判别特征在秒级尺度上已经充分，不需要逐帧。
- **为什么单声道 16kHz**：Whisper 内部就是这个采样率；librosa 频谱分析在此采样率下也足够。
- **为什么落盘缓存**：阶段二及之后会多次读取，缓存后所有下游模块变成纯"数据处理"，便于并行开发与调试。

---

### 4.2 阶段二：多模态特征提取 (Multimodal Feature Extraction)

#### 目的
对阶段一的三种数据流分别计算**稠密时序特征**，每种特征都带时间戳，供阶段三在统一时间轴上融合。

此阶段内部分三条**并行**的子管线：视觉、音频、文本。三者之间无依赖。

#### 4.2.1 视觉子管线
**提取的特征**:
- **镜头切换点 (shot boundaries)** — 相邻帧颜色直方图差异超阈值。**[新增补充]：结合 CSCI 576 基础，利用颜色二次采样格式 (Color-subsampling, 如 4:2:0 或 4:2:2) 优化信号数字化处理及颜色差异计算。**
- **黑帧 / 纯色帧检测** — 帧的平均亮度与方差
- **运动强度** — 相邻帧光流 / 帧差绝对值。**[新增补充]：引入基础离散余弦变换 (DCT) 原理辅助分析画面高频细节变化。**
- **CLIP 零样本场景分类** — 每帧对一组 prompt ("an advertisement slide", "an end credits screen", "a talking head", "a title card", "a video game UI", ...) 计算相似度，输出每帧的场景概率向量

**输出**: `features/visual.json` — 每个采样帧的时间戳 + 上述特征向量

**Python 技术**:
| 用途 | 库 |
|---|---|
| 颜色直方图、帧差、光流 | `opencv-python` (`cv2`) |
| CLIP 推理 | `open_clip_torch` (ViT-B/32 + laion2b 权重) |
| 张量运算 | `torch`, `numpy` |

**为什么需要 CLIP**: 纯规则方法 (直方图、运动) 只能发现"物理切换"，但无法理解"这一帧是不是广告"。CLIP 已在 4 亿图文对上预训练，对动画、TED、游戏等**跨类型**场景都有语义理解力，是本项目支持"视频类型不定"的关键。

#### 4.2.2 音频子管线
**提取的特征**:
- **RMS 能量包络** (秒级) — 检测沉默 / 响度突变。**[新增补充]：应用基础音频数字化与信号提取原理量化“死气”能量特征。**
- **频谱质心、频谱展宽、过零率** — 区分语音 / 音乐 / 噪声。**[新增补充]：结合信号信息熵 (Entropy) 的概念，分析声音频谱结构的随机性与规律性。**
- **语音活动检测 (VAD)** — Silero VAD 输出每个时间点的 speech 概率
- **音乐 vs 语音分类** — 基于频谱特征的启发式判别 (片头曲往往是纯音乐；赞助段常有背景音乐)

**输出**: `features/audio.json` — 时间网格化 (例如每 0.5 秒一格) 的特征矩阵

**Python 技术**:
| 用途 | 库 |
|---|---|
| 音频读取、RMS、频谱特征 | `librosa` |
| VAD | `silero-vad` (ONNX, 极轻量) |
| 数值计算 | `numpy`, `scipy` |

**为什么不用 pyannote.audio**: pyannote 体积大 (~500 MB)、需要 HuggingFace Token、主要做说话人分离 —— 对本项目的分类学不是核心需求。Silero VAD 仅 2 MB 就能满足"是否有人声"的判断。

#### 4.2.3 文本子管线
**提取的特征**:
- **句子级语义嵌入** — 对阶段一转录的每句话用 MiniLM 编码成 384 维向量
- **关键词命中** — 维护一组高区分度关键词表 (`sponsor_keywords` = ["sponsored by", "today's video is brought to you by", ...]; `intro_keywords` = ["welcome back", "in today's video", ...]; `outro_keywords` = ["thanks for watching", "subscribe", ...]; `recap_keywords` = ["last time", "previously", ...])
- **语义相邻度** — 相邻句子的 cosine 相似度，用于发现"话题跳变"边界

**输出**: `features/text.json` — 每句转录对应的嵌入向量、命中的关键词、与前句的相似度

**Python 技术**:
| 用途 | 库 |
|---|---|
| 句子嵌入 | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| 字符串匹配 | 标准库 `re` |
| 余弦相似度 | `numpy` / `scikit-learn` |

**为什么 MiniLM**: 90 MB、CPU 可用、推理极快，对短句语义表示质量充足。更大的模型 (如 MPNet) 收益微小而成本倍增。

#### 4.2.4 阶段二的共同原则
- **每条子管线独立运行、独立缓存**：三人小组可并行开发与测试。
- **所有输出必须带时间戳**：后续融合阶段才能在同一时间轴对齐。
- **模型用完释放**：CLIP 推理结束立即 `del model; torch.cuda.empty_cache()` 再加载 MiniLM，保证 VRAM 峰值不累加。

---

### 4.3 阶段三：时序分割与多模态融合 (Segmentation & Fusion)

#### 目的
将三条模态的特征在统一时间轴上对齐，产出**分段边界**与**每段的分类标签**。

#### 处理流程
1. **时间网格对齐** — 把视觉 (1 FPS)、音频 (可能 2 Hz)、文本 (句子级、非均匀) 重采样到统一的**秒级时间网格**。每秒一行特征向量，列包含三模态所有特征。
2. **硬规则触发** — 先用高置信度规则打低挂果：
   - 连续 ≥ N 秒 RMS 低于阈值且 VAD = 0 → `dead_air`
   - 连续 ≥ N 秒视觉为黑帧/纯色 + 无语音 → `transition` 或 `holding_screen`
   - 句子命中 `sponsor_keywords` → 该句前后若干秒标为 `sponsorship` 候选
   - 视频开头 / 结尾固定窗口 (如前 60 秒 / 后 60 秒) 若 CLIP "title card" 分数高 → `intro` / `outro` 候选
   - **[新增补充] 跨集比对触发：** 提取当前段落的音频频谱或文本嵌入，与预加载的外部“特征字典”（来自其他视频的已知片段）进行相似度比对，若高度匹配 → `recap` 或 `intro` 候选。
3. **边界候选生成** — 在未被硬规则锁定的区域，用以下信号融合产生候选边界点：
   - 视觉镜头切换点
   - CLIP 场景概率向量的跳变 (相邻秒的 KL 散度)
   - 文本语义嵌入的跳变 (相邻句 cosine 距离)
   - 音频 "speech→music" 或 "music→silence" 的状态切换
4. **分段分类** — 每两个候选边界之间形成一个 segment；对 segment 内的所有特征做聚合统计 (均值、众数、命中关键词列表)，得到一个 **segment 级特征向量**；使用以下两种模式之一给出标签：
   - **零样本模式 (MVP)**：计算该段的 CLIP 视觉均值嵌入与 10 个类别的 prompt 嵌入的相似度，取最高类；文本与音频作为额外加权信号。
   - **轻量监督模式 (若时间允许)**：对少量 (数十到数百) 自标注 segment 训练一个 `scikit-learn` 逻辑回归 / GBM，特征即 segment 级特征向量。
5. **时序平滑** — 相邻同类 segment 合并；极短 (< 2 秒) 的孤立 segment 被吸收进邻居；目的是避免碎片化标签。
6. **生成证据字段** — 每个 segment 的 `evidence` 字段记录视觉/音频/文本三个分项得分与触发的规则，供演示时解释。

#### 输出
- `workspace/<video>/metadata.json` (§2.2 定义的完整结构)

#### Python 技术
| 用途 | 库 |
|---|---|
| 时间网格对齐、聚合 | `numpy`, `pandas` |
| 相似度、KL 散度 | `scipy.spatial`, `scipy.stats` |
| 可选的监督分类器 | `scikit-learn` |
| HMM 式平滑 (可选升级) | `hmmlearn` 或手写 Viterbi |

#### 关键设计原则
- **硬规则优先、模型兜底**：能用规则判定的 (沉默、黑屏、关键词) 就不交给模型，规则又快又可解释。模型只处理"两可"的灰区。
- **可解释性**：`evidence` 字段是本项目的"杀手锏"—— 演示时评审会质疑"为什么这是广告"，我们能给出三模态分数 + 触发规则的具体列表。
- **无需标注数据即可跑通 MVP**：零样本模式让我们在没有人工标签的情况下产出完整结果；若时间允许，再用少量自标注数据训练监督模式提升准确率。

---

### 4.4 阶段四：元数据生成与持久化 (Metadata Export)

#### 目的
把阶段三的内部数据结构序列化为 §2.2 定义的 `metadata.json`，并做一层 Schema 校验，保证前端读取时结构稳定。

#### 子任务
1. **Schema 校验** — 用 `pydantic` 定义 `metadata.json` 的数据模型；序列化前做一次完整校验；任何字段缺失或类型错误直接报错。
2. **章节生成** — 合并相邻的 `core_content` segment 为 chapter；chapter 标题从该 chapter 内最高信息密度的句子抽取 (可以用关键词频率 + MiniLM 嵌入做简易摘要)。
3. **跳过建议** — 所有 `label != "core_content"` 的 segment ID 进入 `skip_suggestions[]`。
4. **人类可读摘要** — 每个 segment 从其转录里挑一句最具代表性的话作为 `summary`。
5. **版本标记** — 写入 `video_info.analysis_version`，便于演示时区分不同版本的管线输出。

#### Python 技术
| 用途 | 库 |
|---|---|
| 数据建模与校验 | `pydantic` v2 |
| JSON 序列化 | 标准库 `json` (配合 pydantic) |

---

### 4.5 阶段五：播放器后端与前端 (Player Backend & Frontend)

#### 5A. FastAPI 后端
**职责**: 极薄的数据服务层，不做任何分析。
- `GET /api/videos` — 列出可用视频
- `GET /api/metadata/{video_id}` — 返回对应 `metadata.json`
- **[新增补充] `POST /api/metadata/{video_id}`** — 接收前端人工验证和编辑后的 JSON 数据，并在服务器端落盘覆盖，完成闭环更新。
- 静态路径 `/static/videos/<video>.mp4` — 浏览器通过原生 HTTP Range 请求拖动播放

**Python 技术**:
| 用途 | 库 |
|---|---|
| Web 框架 | `fastapi` |
| ASGI 服务器 | `uvicorn` |
| 静态文件服务 | `fastapi.staticfiles.StaticFiles` |

#### 5B. HTML/JS 前端
**组件**:
1. **视频区** — 原生 `<video controls>` 标签，支持常规播放控制。
2. **时间轴** — 一条水平条，按 segment 着色 (green = core_content, red = non-content 各类);每个 segment 可点击跳转到 `start_sec`。
3. **Segment 列表面板** — 右侧列表展示所有 segment 的类别、时间、摘要；每行有 "Play" / "Skip" 按钮。
4. **全局控制** — "Play Content Only" (跳过所有非核心) / "Skip Non-Content" 复选框；"Next Segment" 按钮。
5. **可选**: hover 时间轴时预览该段摘要。
6. **[新增补充] 人工验证与编辑模式 (Human Verification Mode)** — 提供编辑切换按钮，将侧边栏的 Segment 列表变成表单状态。允许用户纠正 AI 预测的类别，并可保存更新至后端 API，以实现课程要求的 Review/Verification 闭环。

**前端技术**: 尽量简化以满足"非商业、过课"的定位 —— **原生 HTML + CSS + Vanilla JS**，不引入 React/Vue 这类框架。时间轴可用纯 CSS flex / grid 渲染；跳过逻辑纯 JS 事件监听 + `video.currentTime` 控制。
**[技术升级选项]**：如果团队对前端较为熟悉，强烈建议引入 **Vue.js** 来管理复杂的播放器状态跳转，并配合 **D3.js** 来动态绘制带刻度与颜色块的交互式时间轴仪表盘，这能大幅提升演示的高级感与代码可维护性。

#### 关键设计原则
- **后端零逻辑**：所有智能都在 `metadata.json` 里；后端换成任何静态文件服务器 (nginx / python -m http.server) 理论上都能跑。这让演示时最不易出错。
- **浏览器做同步**：HTML5 `<video>` 已内置严格音视频同步；我们不重新造轮子。
- **轻依赖**：前端不用 npm / 构建工具，一套 `.html` + `.css` + `.js` 文件即可演示 (若是采用 Vue/D3，也可以通过 CDN 直接引入静态使用)。

---

## 5. 验收条件 (Acceptance Criteria)

| 条目 | 满足条件 | 验证方式 |
|---|---|---|
| **分类一致性** | 10 类分类体系在所有测试视频上被一致应用；每个 segment 有 `evidence` 说明分类依据 | 人工审阅 3-5 个不同类型视频的 `metadata.json` |
| **分割精准度** | 片头、片尾、赞助段的边界误差 ≤ 3 秒 (相对人工标注) | 在 2-3 个人工标注的测试视频上比对 |
| **多模态融合证据** | 每个 segment 的 `evidence` 包含视觉 + 音频 + 文本三项分数 (非零) | 检查 JSON 字段 |
| **播放器同步** | 音视频无漂移 (浏览器原生保证) | 演示观察 |
| **可视化时间轴** | 每个 segment 以不同颜色显示、可点击跳转 | 演示操作 |
| **跳过导航** | "Skip Non-Content" 模式下自动跳过所有 non-content segment；"Next Segment" 按钮工作 | 演示操作 |
| **完全离线** | 断网情况下从 `.venv` 激活到生成 metadata 再到播放全流程可用 | 演示时断开网络 |
| **性能** | 一段 1 小时视频的完整离线分析在演示机上 ≤ 15 分钟；播放器加载 metadata 与视频 ≤ 3 秒 | 计时 |

---

## 6. 风险与缓解 (Risks & Mitigations)

| 风险 | 缓解策略 |
|---|---|
| Whisper 在特定口音 / 动画配音上转录质量差 | 先用 `base` 模型;若关键视频转录不佳,切到 `small` (代码不变,仅换模型名) |
| CLIP 零样本分类在某些类型 (如纯游戏画面) 上偏差大 | 演示前针对 3-5 段代表视频微调 prompt 列表;若仍不准,加一轮少量自标注 + 逻辑回归 |
| VRAM 不足 | 所有模型支持 CPU 回退;CLIP 改用 ViT-B/32 已是最小;最坏情况在 CPU 上跑一晚 |
| FFmpeg 路径 / Windows 环境问题 | 在 `pyproject.toml` 与 README 明确 ffmpeg 安装步骤;优先 `imageio-ffmpeg` 自带二进制包 |
| 前端浏览器兼容 | 仅测试 Chrome;不做跨浏览器保证 |
| 3 人并行开发冲突 | 阶段二三条子管线天然可并行;前端与后端通过 `metadata.json` 契约解耦;每人一个分支 |

---

## 7. 开发节奏建议 (Development Cadence)

演示日 2026-05-06,今天 2026-04-16,**剩余约 3 周**。建议节奏：

| 周 | 目标 |
|---|---|
| Week 1 (04-16 ~ 04-22) | 阶段一完整跑通 + 阶段二三条子管线各自出特征文件 (三人并行) |
| Week 2 (04-23 ~ 04-29) | 阶段三融合管线打通;阶段四 `metadata.json` Schema 固定;阶段五前后端骨架 |
| Week 3 (04-30 ~ 05-05) | 端到端联调;在 3-5 个代表视频上调参;制作演示稿;预演一遍 |

**建议**:
- 每周一次全员对齐,检查三条子管线输出格式是否一致。
- `metadata.json` 的 Schema **在 Week 1 末就必须冻结**,否则前后端无法并行开发。
- 演示视频 (3-5 个) 在 Week 1 就选定,覆盖动画 / TED / 播客 / YouTube vlog / 新闻 等多种类型。

---

## 8. 交付物清单 (Deliverables)

- [ ] `pyproject.toml` + 可用的 `.venv`
- [ ] 离线分析管线:可通过一条命令处理一个 MP4 并产出 `metadata.json`
- [ ] FastAPI 后端 + HTML 前端播放器
- [ ] 3-5 段代表视频的 `metadata.json` 样本
- [ ] README: 环境安装、运行步骤、演示脚本
- [ ] 演示当天的 slides / 讲解稿

---

**文档结束。** 本 PDD 为 V2.1;后续若技术选型有调整,在此基础上修订并提升版本号。
