# 统一信息提取服务设计文档

## 0. 一页结论

### 0.1 这是什么

这是一个统一信息提取服务，目标是把图片、PDF、docx、文本、URL 等输入统一转换为 `DocumentIR`，再按 schema 输出结构化结果。

它不是单独的 OCR 服务，也不是单独的模型演示程序，而是一条完整的服务链路：

- 输入接入
- 标准化
- 信息抽取
- 结果后处理
- 缓存、任务、事件流、结果查询

### 0.2 当前做到哪

当前已经达到：

- 主链路可跑
- API 可用
- 同步/异步可用
- 标准化输入可用
- SSE 进度流可用
- heuristic extractor 可用
- ONNX 字符串 I/O 路线可用
- tokenized/UIE 路线已有代码路径

当前阶段判断：

- 工程阶段：`Alpha / 可跑的 MVP 服务`
- 适合：内部验证、小范围接入、围绕真实样本持续打磨
- 暂不适合：直接作为生产级、多租户、长期稳定运行服务

### 0.3 当前最大缺口

距离“生产可用”最关键的缺口有四个：

1. 本地 OCR 已具备 MVP，但真实模型质量回归、classification 编排和基于真实 OCR 模型的扫描件端到端验证仍未完成。
2. storage 和 queue 仍是 in-memory，异步任务不具备真正的持久化恢复能力。
3. tokenized/UIE ONNX 虽然代码路径已实现，但还缺真实模型集成验证。
4. 缓存治理、租户隔离、可观测性仍未完成。

### 0.4 当前最优先顺序

建议优先级：

1. 本地 `local-onnx` OCR provider 最小闭环
2. SQLite 持久化 `tasks/results/events/cache`
3. 字符串 I/O ONNX 真实模型集成测试
4. tokenized/UIE ONNX 真实模型集成测试
5. 缓存 TTL 与后台清理

### 0.5 进入 Beta 的最低条件

满足以下条件后，才建议把当前项目从 `Alpha` 认定为 `Beta`：

- 图片链路不再强依赖外部 OCR worker
- 异步任务在服务重启后可恢复
- 至少一条 ONNX 模型链路有真实模型集成测试和回归样本
- 缓存具备 TTL 和基本治理能力
- 关键阶段具备基础日志、指标和错误排查能力

### 0.6 一句话判断

这不是“还停在原型脚本”的状态了，而是一个已经立住骨架、可以持续工程化推进的提取服务；接下来最重要的不是再发散设计，而是按优先级把 OCR、本地持久化和真实模型验证补齐。

## 1. 背景

当前已有原型可以对小红书笔记截图执行如下流程：

1. 输入图片。
2. 先通过 OCR 得到文本。
3. 再基于 schema 执行 NLP 信息抽取。

原型已经证明了产品方向成立，但仍存在以下问题：

- 输入形式分散：图片、PDF、文档、纯文本尚未统一成一条服务链路。
- 推理路径耦合：OCR、版面解析、信息抽取直接耦合在脚本中，不利于替换模型。
- 平台限制明显：部分 Paddle 多模态模型在 macOS arm64 上存在稳定性问题。
- 缺少服务能力：没有统一 API、任务状态、缓存、监控与可观测性。
- 成本不可控：如果所有输入都走重 OCR 和重模型，长期运行成本会偏高。

本设计目标是在尽量节省成本的前提下，将原型演进为一个统一的“文档/图片信息提取服务”。


## 2. 产品目标

### 2.1 核心目标

提供统一提取接口，支持图片、PDF、文本等多种输入，按照业务 schema 输出结构化信息。

核心能力：

- 接受 `image / pdf / txt / docx / html` 等输入。
- 自动判断预处理路径。
- 将不同格式统一转换为中间文档表示。
- 按 schema 提取结构化字段。
- 返回字段值、置信度、证据位置、标准化文本。
- 支持同步调用和异步任务调用。

### 2.2 非目标

以下内容不在第一阶段范围内：

- 训练自研大模型。
- 自研 OCR 引擎。
- 复杂版面编辑或文档重建。
- 完整工作流编排平台。
- 多租户权限体系的深度建设。


## 3. 设计原则

### 3.1 统一接口，分层处理

外部只有一个统一提取接口，内部按照输入类型选择不同的预处理链路。

### 3.2 优先低成本路径

避免所有输入一律 OCR、避免所有请求一律走重模型。优先：

- 直接文本提取。
- 轻量 OCR。
- 轻量文本抽取模型。
- 缓存命中。

只有在必要时才升级到更重的路径。

### 3.3 模型与服务解耦

服务层不直接依赖某一个 OCR 或 IE 模型实现，而是通过 provider 接口接入。

### 3.4 保留证据链

不要只输出字段结果，还要保留字段来自哪一页、哪一段、哪一个 OCR block，便于调试、审核和回放。

### 3.5 Rust 负责服务编排

服务层、任务层、缓存层、接口层尽量使用 Rust 构建，降低运行时开销、提升部署一致性。OCR 和推理层可按 provider 接入 C++/ONNX 能力。


## 4. 需求分析

### 4.1 功能需求

#### 输入

- 上传文件。
- 传入 URL。
- 直接传入文本。
- 传入 SDK 预处理后的标准化文档。

#### 支持的文件类型

- 图片：`png/jpg/jpeg/webp`
- 文档：`pdf/docx/txt/html/md`

#### 输出

- 抽取结果。
- 字段置信度。
- 证据片段。
- 页码与位置信息。
- 原始标准化文本。
- 任务状态。

#### 调用方式

- 同步：适合小图片、小文本、小 PDF。
- 异步：适合大 PDF、多页扫描件、批量任务。

### 4.2 非功能需求

- 低成本运行。
- 单机可部署。
- 可水平扩展。
- 服务启动快。
- 可替换 OCR / parser / extractor。
- 有清晰的日志、指标、链路追踪。


## 5. 约束与假设

### 5.1 技术约束

- 服务层优先使用静态语言，目标语言为 Rust。
- 尽量不将 Python 作为主服务运行时。
- 信息抽取运行时明确采用 `ONNX Runtime CPU Execution Provider`。
- Rust 侧优先通过 ONNX Runtime 绑定集成推理，不优先选择纯 Rust ONNX 实现。
- OCR 仍可按 provider 接入 C++ 推理库或独立 worker。
- 生产环境默认 Linux，不以 macOS 作为生产部署标准。

### 5.2 成本约束

- 优先 CPU 可运行方案。
- OCR 只在必要时触发。
- 大文件优先异步执行。
- 可接受在效果优先的局部环节引入单独 OCR worker，但不接受整套服务依赖 Python 常驻。

### 5.3 业务假设

- 用户关注的是“schema 抽取结果”，不是底层 OCR/模型细节。
- 当前核心场景以中文内容为主。
- 首批场景更偏短文档、图片截图、扫描件和业务模板文档。


## 6. 总体方案

### 6.1 系统总览

统一服务采用五段式处理链路：

1. `Ingestion`：接入输入，识别类型，生成任务。
2. `Normalization`：将不同来源统一转换为中间文档结构 `DocumentIR`。
3. `Extraction`：根据 schema 执行信息抽取。
4. `Postprocess`：归一化、去噪、冲突消解、置信度聚合。
5. `Persistence`：保存任务元数据、结果、缓存、审计信息。

### 6.2 核心思想

系统对外暴露的是“统一提取服务”，不是“统一 OCR 服务”。

不同输入的差异只存在于标准化阶段：

- 图片：需要 OCR。
- 扫描 PDF：优先回退 OCR provider。
- 文本 PDF：直接提取文本层。
- docx/html/txt：直接提取文本。

一旦进入 `DocumentIR`，后续抽取逻辑统一。

当前实现补充：

- `DocumentIR` 不再只保留一份大文本，而是尽量保留 `pages[].blocks[]` 颗粒度。
- 图片会按 OCR 行生成 block。
- 文本层 PDF 会按页拆分，并尽量按行生成 block。
- docx 会按段落生成 block。

### 6.3 双轨输入策略

为了降低服务端流量和计算成本，系统采用双轨输入策略：

- 轨道 A：原始输入轨道
- 轨道 B：标准化输入轨道

原始输入轨道面向通用客户端：

- 直接上传图片、PDF、docx、文本。
- 服务端负责前处理、抽取和后处理。

标准化输入轨道面向官方 SDK 或边缘节点：

- 客户端本地完成 OCR、PDF 文本层提取、docx 解包等前处理。
- 网络上传压缩后的标准化文档，而不是原始大文件。
- 服务端直接跳过 parser，进入抽取和后处理阶段。

设计原则：

- 服务端必须保留原始输入能力，不能只依赖 SDK。
- SDK 成功前处理时优先走标准化输入轨道。
- SDK 失败或客户端能力不足时回退到原始输入轨道。
- 两条轨道在进入 `DocumentIR` 后合并到同一条抽取内核。


## 7. 逻辑架构

### 7.1 模块划分

#### API Gateway

职责：

- 接收请求。
- 校验输入。
- 计算内容哈希。
- 选择同步或异步模式。
- 返回结果或任务 ID。

#### Job Service

职责：

- 维护任务状态。
- 调度预处理和抽取流程。
- 管理重试和超时。

#### Parser Service

职责：

- 根据文件类型执行解析。
- 对 PDF 判断是否存在文本层。
- 对无文本层 PDF 回退 OCR provider。
- 输出 `DocumentIR`。

#### OCR Provider

职责：

- 对图片或扫描页执行 OCR。
- 输出文本块、坐标、置信度。

设计补充：

- OCR 不应只抽象成“调用哪个 provider”，还应抽象成“前处理 / 运行时 / 结果适配”三层。
- 原因是 OCR 比 extractor 更容易发生引擎切换、部署方式切换和输出协议差异。
- 上层 parser 和 `DocumentIR` 应尽量只依赖统一的 `OcrOutput`，不直接感知底层 OCR 引擎细节。

#### Extractor Provider

职责：

- 根据 schema 从文本中抽取结构化字段。
- 做基础类型转换，例如 string / number / boolean。
- 支持子字段聚合、hint 命中和证据回绑。
- 支持多种实现：ONNX、规则、外部推理服务。

#### Postprocessor

职责：

- 归一化字段。
- 合并多候选结果。
- 计算最终置信度。
- 绑定证据。
- 清洗重复值与重复 evidence。

#### Storage

职责：

- 存储任务、输入文件元数据、缓存和抽取结果。

### 7.2 数据流

同步链路：

1. 客户端上传输入。
2. API 校验并识别文件类型。
3. Parser 将输入转换为 `DocumentIR`。
4. Extractor 按 schema 抽取。
5. Postprocessor 归一化结果。
6. 返回结构化 JSON。

异步链路：

1. 客户端上传输入。
2. API 创建任务并落库。
3. Job Worker 异步执行标准化和抽取。
4. 结果入库。
5. 客户端轮询或通过回调获取结果。


## 8. 核心数据模型

### 8.1 DocumentIR

`DocumentIR` 是系统的核心中间表示，用于承载统一的解析结果。

建议结构：

```rust
pub struct DocumentIr {
    pub doc_id: String,
    pub source_type: SourceType,
    pub pages: Vec<PageIr>,
    pub plain_text: String,
    pub metadata: DocumentMetadata,
}

pub struct PageIr {
    pub page_no: u32,
    pub width: Option<f32>,
    pub height: Option<f32>,
    pub blocks: Vec<TextBlock>,
}

pub struct TextBlock {
    pub block_id: String,
    pub page_no: u32,
    pub text: String,
    pub bbox: Option<BBox>,
    pub confidence: Option<f32>,
    pub source_kind: BlockSourceKind,
}

pub struct BBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
}
```

设计要点：

- `plain_text` 供轻量抽取模型直接消费。
- `blocks` 供证据定位、置信度聚合和回放使用。
- `source_kind` 用于区分文本层提取、OCR 提取、规则合成等来源。

### 8.1.1 NormalizedDocument

`NormalizedDocument` 是面向 SDK/边缘端上传的协议对象，可以看作 `DocumentIR` 的可传输版本。

建议结构：

```rust
pub struct NormalizedDocument {
    pub source_type: SourceType,
    pub pages: Vec<NormalizedPage>,
    pub plain_text: String,
    pub metadata: DocumentMetadata,
}

pub struct NormalizedPage {
    pub page_no: u32,
    pub width: Option<f32>,
    pub height: Option<f32>,
    pub blocks: Vec<NormalizedTextBlock>,
}

pub struct NormalizedTextBlock {
    pub page_no: u32,
    pub text: String,
    pub bbox: Option<BBox>,
    pub confidence: Option<f32>,
    pub source_kind: BlockSourceKind,
}
```

设计要点：

- SDK 侧只需要构建 `NormalizedDocument`，不需要感知服务端 parser 实现。
- 服务端接收后立即转换为 `DocumentIR`，后续抽取逻辑不区分来源。
- `metadata` 中建议附带 `sdk_version`、`ocr_provider`、`pdf_provider`、`source_hash` 等信息。

### 8.2 SchemaSpec

服务内部不直接暴露底层模型的 schema 语法，而是定义自己的通用 schema。

```rust
pub struct SchemaSpec {
    pub name: String,
    pub version: String,
    pub fields: Vec<FieldSpec>,
}

pub struct FieldSpec {
    pub key: String,
    pub field_type: FieldType,
    pub required: bool,
    pub multiple: bool,
    pub children: Vec<FieldSpec>,
    pub hints: Vec<String>,
}
```

这样做的好处：

- 客户端协议稳定。
- 可适配不同抽取引擎。
- schema 可做版本管理。

### 8.3 ExtractionResult

```rust
pub struct ExtractionResult {
    pub task_id: String,
    pub status: TaskStatus,
    pub fields: Vec<FieldValue>,
    pub raw_text: Option<String>,
    pub timings: TimingBreakdown,
}

pub struct FieldValue {
    pub key: String,
    pub value: serde_json::Value,
    pub confidence: Option<f32>,
    pub evidences: Vec<Evidence>,
}

pub struct Evidence {
    pub page_no: Option<u32>,
    pub text: String,
    pub bbox: Option<BBox>,
    pub source_block_ids: Vec<String>,
}
```

### 8.4 Task 状态机

```text
created -> queued -> parsing -> extracting -> postprocessing -> succeeded
                                                   \-> failed
```

可选状态：

- `cached_hit`
- `partial_success`
- `retrying`


## 9. 输入处理策略

### 9.1 图片

策略：

- 图片统一进入 OCR。
- OCR 输出 `TextBlock[]`。
- 聚合为 `plain_text`。
- 再执行 schema 抽取。
- OCR 契约现在还会尽量携带页级元信息，例如 `page_no / width / height / rotation_degrees`，用于前端标注和扫描页可视化；在 `local-onnx` 路线下，`rotation_degrees` 可来自 classification 输出的页内主方向聚合结果。

SDK 侧优化路径：

- 客户端本地 OCR。
- 上传 `NormalizedDocument`。
- 服务端跳过图片 parser 和 OCR provider。

适用场景：

- 截图。
- 扫描图片。
- 海报、表单、聊天截图。

### 9.2 PDF

策略分两类：

#### 文本型 PDF

- 优先直接提取文本层。
- 按页生成 block。
- 跳过 OCR。

#### 扫描型 PDF

- 将每页渲染成图片。
- 页级 OCR。
- 输出 block 和文本。

关键点：

- 不允许所有 PDF 默认 OCR。
- 要设计一个 `needs_ocr()` 判定逻辑。

判定维度可以包括：

- 文本层是否为空。
- 文本密度是否过低。
- 页内可提取字符数量是否异常。

SDK 侧优化路径：

- 客户端优先提取 PDF 文本层。
- 若需要，也可在客户端完成页级 OCR。
- 上传 `NormalizedDocument`，减少原始 PDF 传输体积。

### 9.3 docx / html / txt / md

策略：

- 优先使用轻量解析器直接转文本。
- 尽量保留段落边界。
- 用段落模拟 block。

SDK 侧优化路径：

- docx 在客户端本地解包为文本和段落块。
- html/txt/md 在客户端标准化为 `plain_text + blocks`。
- 上传 `NormalizedDocument`。


## 10. 技术选型

### 10.1 总体选型结论

推荐：

- 服务层：Rust
- API 框架：Axum
- 异步运行时：Tokio
- 序列化：Serde
- OpenAPI：Utoipa 或 Okapi
- 存储：Postgres，MVP 可先 SQLite
- 本地缓存：文件缓存 + SQLite/Postgres
- OCR：Provider 模式，短期支持 HTTP worker，后续补齐本地内置 ONNX/C++ OCR provider
- PDF：PDFium
- 抽取：`ONNX Runtime CPU Execution Provider` + 文本抽取模型

### 10.2 为什么选择 Rust

优点：

- 低内存占用。
- 单二进制部署方便。
- 并发模型适合高吞吐 API 和 worker。
- 类型系统有利于约束数据流。
- 服务层不依赖 Python 解释器与虚拟环境。

### 10.3 为什么不建议一开始全栈 C++

缺点：

- API 层与业务编排开发效率较低。
- JSON、HTTP、任务调度、数据库接入的工程复杂度更高。
- 团队维护门槛更高。

因此更合理的做法是：

- 用 Rust 做服务编排。
- 把需要极致推理性能的能力放到 C++ provider 或 ONNX Runtime。

### 10.4 OCR 选型

推荐采用 provider 抽象：

```rust
pub trait OcrProvider {
    fn recognize(&self, input: OcrInput) -> anyhow::Result<OcrOutput>;
}
```

但工程上建议进一步拆成三层，而不是把所有逻辑都塞进一个 provider：

1. `Preprocess Layer`
2. `Runtime Layer`
3. `Result Adapter Layer`

建议职责：

#### Preprocess Layer

- 图片缩放与归一化。
- 旋转纠正。
- PDF 页转图片。
- 大图切片或页图切分。

#### Runtime Layer

- 真正调用 OCR 引擎。
- 可以是 `http worker`、本地 ONNX、本地 C++、云 OCR。
- 负责模型加载、线程数、超时、预热和生命周期。

#### Result Adapter Layer

- 将不同 OCR 引擎输出统一转换为服务内部 `OcrOutput`。
- 统一字段至少包括：
  - `blocks[].block_id`
  - `blocks[].text`
  - `blocks[].line_count`
  - `blocks[].page_no`
  - `blocks[].bbox`
  - `lines[].text`
  - `lines[].block_id`
  - `lines[].confidence`
  - `lines[].page_no`
  - `lines[].bbox`

这样做的好处：

- 切换 OCR 引擎时，尽量不影响 parser。
- HTTP OCR 和本地 OCR 可以复用同一套上层逻辑。
- `DocumentIR` 和 evidence 链路保持稳定。
- 后续如果要接 block/line/word 多层级输出，也只需要扩展 adapter 层。

候选实现：

#### Provider A：PaddleOCR C++ Worker

优点：

- 中文 OCR 效果通常更强。
- 适合复杂截图和扫描件。

缺点：

- 部署相对复杂。
- 与主服务之间需要 RPC/子进程/FFI 边界。

适合：

- 生产主链路。

当前服务实现建议先保留两条路并行演进：

- 短期优先按 HTTP worker 集成，保持 Rust 主服务与 OCR 引擎解耦。
- 中期补齐本地内置 OCR provider，让单机部署不再强依赖外部 OCR 服务。

HTTP worker 约定：

- `POST /v1/ocr`
- 请求体直接传图片二进制
- `Content-Type` 透传原始图片 MIME
- 可选请求头 `X-File-Name`
- 请求头 `X-Ocr-Request-Id`
- 可选请求头 `X-Ocr-Source-Type`
- 可选请求头 `X-Ocr-Page-No-Hint`
- 可选请求头 `X-Ocr-Meta-*`，用于透传 `sdk_version / protocol_version / pdf_ocr_input` 等上下文
- 可选 `Authorization: Bearer <token>`

响应示例：

```json
{
  "provider": "rapidocr-worker",
  "model": "rapidocr-onnx",
  "request_id": "ocr-req-123",
  "timing_ms": 58,
  "warnings": ["page rotated by OCR classifier"],
  "pages": [
    {
      "page_no": 1,
      "width": 1242.0,
      "height": 1660.0,
      "rotation_degrees": 180.0,
      "blocks": [
        {
          "block_id": "ocr-http-b1",
          "bbox": {"x1": 10.0, "y1": 20.0, "x2": 160.0, "y2": 84.0},
          "confidence": 0.96,
          "lines": [
            {"text": "岗位类型：图像策略", "confidence": 0.98},
            {"text": "人设要点：边缘预处理", "confidence": 0.93}
          ]
        }
      ]
    }
  ]
}
```

主服务接到响应后：

- 优先把 `pages[].blocks[].lines[]` 展平为统一 `OcrOutput`；老的平铺 `blocks/lines` 响应仍继续兼容。
- 合并 OCR blocks/lines 生成 `plain_text`
- 将 `provider/model` 回写到 `DocumentIR.metadata.extra`
- 若 worker 返回 `request_id / timing_ms / warnings`，主服务会继续透传到 `DocumentIR.metadata.extra` 与 SSE 事件，便于排障。
- 会把 `source_type / page_no_hint / metadata` 等请求上下文以 header 透传给 worker，便于外部 OCR 服务做日志、追踪和多页路由。
- 记录 `ocr_transport=http-ocr-worker`，用于区分 transport 和真实 OCR 引擎

#### Provider B：本地内置 ONNX / C++ OCR Provider

候选形态：

- 通过 ONNX Runtime 在 Rust 进程内直接执行 OCR 检测/识别模型。
- 或通过 Rust 到本地 C++ 动态库 / 子进程桥接 PaddleOCR、RapidOCR 等引擎。

优点：

- 单机部署更完整，不依赖外部 OCR worker 网络链路。
- 更适合作为离线环境、边缘节点或内网部署默认方案。
- provider 元数据和错误语义更容易与主服务统一。

缺点：

- 模型文件管理、内存占用和平台兼容性会更复杂。
- 需要额外处理图片预处理、版面切分和多阶段 OCR pipeline 的工程细节。

适合：

- 单机版、一体化部署、对外部依赖敏感的场景。

建议：

- 保持与 `http` OCR worker 相同的 `OcrOutput` 契约。
- 在配置层新增 `local-onnx` 或 `local-paddle` provider 名称。
- 在元数据中补充 `ocr_transport=inproc`，用于区分进程内推理和远程 worker。

#### Provider C：Tesseract

优点：

- 纯本地、成熟、易部署。

缺点：

- 中文复杂截图效果不一定理想。

适合：

- 低成本备用方案。

#### Provider D：RapidOCR

优点：

- 原型快。

缺点：

- 当前生态更偏 Python。

适合：

- 验证期或离线评测。

### 10.5 PDF 解析选型

推荐：

- `PDFium` 作为 PDF 文本提取和渲染基础。

原因：

- 可同时支持文本层提取与页面渲染。
- 适合做“先抽文本，必要时 OCR”的双路径。

当前实现：

- 已支持“文本层优先 + OCR fallback”。
- 已引入 `CompositePdfProvider` 边界：文本层提取与页图栅格化可分别替换。
- 默认文本层提取仍使用 `lopdf`。
- 若配置 `pdftoppm` rasterizer，当 PDF 文本层为空时会先产出 `raster_pages`，再交给 OCR provider 逐页识别。
- 若未配置 rasterizer，仍保留 `original_pdf_bytes` 兼容兜底路径，但这不应视为扫描 PDF 的最终生产方案。

### 10.6 信息抽取模型选型

策略：

- 服务核心统一走“文本抽取”。
- 不把多模态模型作为核心依赖。
- 推理运行时固定为 `ONNX Runtime CPU Execution Provider`。

运行时决策：

- 生产主线采用官方 ONNX Runtime。
- 第一阶段只启用 CPU Execution Provider，不引入 GPU 依赖。
- Rust 服务内常驻 ONNX session，不按请求重复初始化。
- 部署形态优先 `Rust binary + onnxruntime shared library + Linux container`。

选择原因：

- 算子兼容性和工程成熟度通常优于纯 Rust ONNX 方案。
- CPU 路线更贴合当前成本约束和单机部署目标。
- 便于后续继续保留 heuristic / worker fallback，不会把服务绑定死在单一模型上。

候选路径：

#### 路径 A：UIE / 类 UIE 模型导出 ONNX

优点：

- 与现有原型思路一致。
- schema 抽取表达力较好。
- 可直接复用 `ONNX Runtime CPU Execution Provider` 这条部署链。

缺点：

- 模型导出与推理兼容性需验证。
- tokenizer、输入张量拼装与结果解码在 Rust 侧需要单独落地。

#### 路径 B：规则 + 轻量模型混合

优点：

- 特定场景成本低。

缺点：

- 泛化性较差。

#### 路径 C：LLM 兜底

优点：

- 对复杂非结构化文本鲁棒性更强。

缺点：

- 成本高，延迟高，可控性弱。

建议：

- 第一阶段以路径 A 为主。
- 路径 B 用于强模板场景。
- 路径 C 仅用于低置信 fallback。
- 若某个 UIE 模型在 ONNX Runtime CPU 路线上迟迟不稳定，不要硬卡主线，可先保留 heuristic extractor 或外部 worker 兜底。


## 11. 模块设计

### 11.1 parser 模块

职责：

- 根据 MIME 和文件头识别输入。
- 选择 `ImageParser`、`PdfParser`、`TextParser`。
- 输出 `DocumentIR`。

注意：

- 当请求已经携带 `NormalizedDocument` 时，parser 模块会被绕过。
- parser 仅服务于原始输入轨道。

trait 草案：

```rust
pub trait Parser {
    fn parse(&self, input: ParseInput) -> anyhow::Result<DocumentIr>;
}
```

### 11.2 extractor 模块

职责：

- 接收 `DocumentIR + SchemaSpec`。
- 选择合适的文本视图。
- 调用模型推理。
- 将结果映射为统一输出。

当前实现：

- 使用 heuristic extractor 作为启动版本。
- 已支持 schema key 和 `hints` 命中。
- 已支持 `string / number / boolean` 基础类型转换。
- 已支持 object/array 子字段聚合。
- 已接入 `onnx` extractor provider 配置入口。
- 已实现 ONNX Runtime CPU 路线的模型加载、session 启动与字符串 I/O 真推理。
- 已支持将 `json_output` 解码为统一 `ExtractionResult`，并按文本或 `source_block_ids` 回绑 evidence。
- tokenizer、张量拼装与 span 类输出解码仍是下一阶段工作。

trait 草案：

```rust
pub trait Extractor {
    fn extract(&self, doc: &DocumentIr, schema: &SchemaSpec) -> anyhow::Result<ExtractionResult>;
}
```

### 11.3 postprocess 模块

职责：

- 去噪。
- 合并重复字段。
- 归一化值格式。
- 绑定 evidence。

当前实现：

- 会对字符串做 trim。
- 会对数组值做去空、去重。
- 会对 evidence 去重并排序。

### 11.4 cache 模块

缓存维度：

- 文件级缓存：同一内容哈希不重复解析/OCR。
- 页级缓存：PDF 页 OCR 结果缓存。
- schema 级缓存：相同 `doc_hash + schema_hash` 复用抽取结果。

缓存键建议：

```text
parse:{file_hash}:{parser_version}
ocr:{page_hash}:{ocr_provider}:{ocr_version}
extract:{doc_hash}:{schema_hash}:{extractor_version}
```

当前实现：

- 缓存键已经按 `payload + schema + options` 哈希生成。
- 缓存项除结果本身外，还记录 `created_at_ms / last_accessed_at_ms / hit_count`。
- 命中缓存时会重置返回结果的阶段耗时，并把命中信息写入任务 message。

### 11.5 queue 模块

第一阶段：

- 进程内 Tokio 队列。

第二阶段：

- Redis / NATS / RabbitMQ。

### 11.6 storage 模块

建议持久化对象：

- 任务表。
- 文件表。
- 抽取结果表。
- 错误日志表。
- 缓存索引表。


## 12. API 设计

### 12.1 创建提取任务

`POST /v1/extractions`

请求体示例：

```json
{
  "mode": "sync",
  "schema": {
    "name": "xiaohongshu_note_profile",
    "version": "1",
    "fields": [
      {
        "key": "岗位类型",
        "field_type": "string",
        "required": false,
        "multiple": false,
        "children": [],
        "hints": []
      },
      {
        "key": "人设要点",
        "field_type": "string",
        "required": false,
        "multiple": true,
        "children": [],
        "hints": []
      }
    ]
  },
  "options": {
    "ocr": "auto",
    "return_raw_text": true,
    "return_evidence": true
  }
}
```

文件可通过 multipart 上传，或在 JSON 中传 `url` / `text`。

### 12.1.1 创建标准化提取任务

`POST /v1/extractions/normalized`

请求体示例：

```json
{
  "mode": "sync",
  "document": {
    "source_type": "image",
    "plain_text": "岗位类型：产品经理\n人设要点：结构化表达",
    "metadata": {
      "file_name": "note.png",
      "mime_type": "image/png",
      "extra": {
        "protocol_version": "1",
        "sdk_version": "0.1.0",
        "ocr_provider": "local-rapidocr",
        "source_hash": "sha256:..."
      }
    },
    "pages": [
      {
        "page_no": 1,
        "width": 1170,
        "height": 2532,
        "blocks": [
          {
            "page_no": 1,
            "text": "岗位类型：产品经理",
            "bbox": {"x1": 20, "y1": 40, "x2": 200, "y2": 80},
            "confidence": 0.96,
            "source_kind": "ocr"
          }
        ]
      }
    ]
  },
  "schema": {
    "name": "demo",
    "version": "1",
    "fields": [
      {"key": "岗位类型", "field_type": "string", "required": false, "multiple": false, "children": [], "hints": []}
    ]
  },
  "options": {
    "return_raw_text": true,
    "return_evidence": true
  }
}
```

返回逻辑：

- 与 `/v1/extractions` 保持一致。
- 服务端直接跳过 parser 阶段。
- 可继续使用同步/异步、缓存、任务回查能力。
- `document.metadata.extra` 至少需要包含 `protocol_version` 或 `sdk_version`。
- 当前服务端要求 `protocol_version = "1"`，同时兼容只有 `sdk_version` 的旧 SDK 请求。

同步响应示例：

```json
{
  "task_id": "task_123",
  "status": "succeeded",
  "cached": false,
  "result": {
    "task_id": "task_123",
    "status": "succeeded",
    "fields": [
      {
        "key": "岗位类型",
        "value": "校招产品经理",
        "confidence": 0.92,
        "evidences": [
          {
            "page_no": 1,
            "text": "岗位类型：校招产品经理",
            "bbox": {
              "x1": 32.0,
              "y1": 100.0,
              "x2": 220.0,
              "y2": 128.0
            },
            "source_block_ids": ["b1"]
          }
        ]
      }
    ],
    "raw_text": "岗位类型：校招产品经理",
    "timings": {
      "parse_ms": 0,
      "extract_ms": 12,
      "postprocess_ms": 1,
      "total_ms": 13
    }
  }
}
```

### 12.1.2 提取事件流

`GET /v1/extractions/{task_id}/events`

第一阶段采用 `SSE (text/event-stream)`，不优先引入 WebSocket。

设计原则：

- 输入端第一阶段仍以完整上传为主，不强行做真流式上传。
- 输出端优先做单向事件流，提升用户对处理进度和结果收敛过程的感知。
- 小请求继续使用普通同步 JSON；大请求或异步任务可订阅事件流。
- ONNX Runtime CPU Execution Provider 更适合做“分页/分阶段/分快照”的增量结果流，不适合 token 级流式输出。

事件类型建议：

- `task.accepted`
- `stage.changed`
- `document.ready`
- `page.parsed`
- `page.ocr_done`
- `block.extracted`
- `result.partial`
- `result.snapshot`
- `cache.hit`
- `completed`
- `failed`

事件数据示例：

```text
event: task.accepted
data: {"sequence":1,"event_type":"task.accepted","task_id":"task_123","created_at_ms":1710000000000,"payload":{"mode":"async","ingest_mode":"parsed"}}

event: stage.changed
data: {"sequence":2,"event_type":"stage.changed","task_id":"task_123","created_at_ms":1710000000100,"payload":{"stage":"parsing","status":"running"}}

event: page.parsed
data: {"sequence":3,"event_type":"page.parsed","task_id":"task_123","created_at_ms":1710000000500,"payload":{"page_no":1,"width":1242.0,"height":1660.0,"block_count":12,"text_chars":236}}

event: page.ocr_done
data: {"sequence":4,"event_type":"page.ocr_done","task_id":"task_123","created_at_ms":1710000000800,"payload":{"page_no":1,"width":1242.0,"height":1660.0,"rotation_degrees":0.0,"ocr_block_count":12,"ocr_line_count":18,"ocr_provider":"local-onnx-ocr","ocr_model":"det=...","ocr_transport":"inproc","ocr_request_id":"ocr-req-123","ocr_timing_ms":58,"ocr_warnings":["page rotated by OCR classifier"],"ocr_blocks_preview":[{"block_id":"ocr-p1-b1","text":"岗位类型：图像策略","text_chars":8,"bbox":{"x1":10.0,"y1":20.0,"x2":110.0,"y2":44.0},"confidence":0.98}],"pdf_ocr_input":"page_rasters","pdf_raster_provider":"pdftoppm"}}

event: result.snapshot
data: {"sequence":5,"event_type":"result.snapshot","task_id":"task_123","created_at_ms":1710000001200,"payload":{"cached":false,"field_count":3,"result":{...}}}

event: completed
data: {"sequence":6,"event_type":"completed","task_id":"task_123","created_at_ms":1710000001300,"payload":{"cached":false,"result":{...}}}
```

当前实现：

- 已提供任务级 SSE 事件流骨架。
- 已接入 `task.accepted / stage.changed / document.ready / page.parsed / page.ocr_done / block.extracted / result.partial / result.snapshot / cache.hit / completed / failed`。
- `result.partial` 已支持按页面前缀逐步修订，同一字段可随着后续页面继续更新。
- `page.parsed` 已可携带页级 `width / height`，便于前端先建立页面坐标系。
- `page.ocr_done` 已可携带 `width / height / rotation_degrees / ocr_block_count / ocr_line_count` 与 OCR provider 元信息，并附带 `ocr_request_id / ocr_timing_ms / ocr_warnings[] / ocr_blocks_preview[]`，便于前端直接做 bbox 叠加、排障和 OCR 调试展示；当存在页级 OCR 可观测字段时，事件优先使用 `ocr_page_{n}_*`，否则回退到全局 `ocr_*`。
- 当前实现里，增量型 `result.partial` 与 `block.extracted` 事件也已可携带页级 `width / height / rotation_degrees`，便于字段高亮直接挂回当前页面。
- `result.partial` 与 `block.extracted` 当前都可额外携带聚合后的 `bbox` 与去重后的 `bboxes[]`，便于前端直接绘制字段高亮框，而不必自己再做 evidence 框合并。

### 12.2 查询任务结果

`GET /v1/extractions/{task_id}`

### 12.3 获取健康状态

`GET /healthz`

### 12.4 获取版本信息

`GET /version`

返回：

- 服务版本。
- normalized 协议版本。
- 是否兼容仅携带 `sdk_version` 的旧 SDK。
- parser 版本。
- OCR provider 版本。
- extractor 版本。


## 13. 同步与异步策略

### 13.1 同步适用范围

建议限制：

- 文本长度小于阈值。
- 图片大小小于阈值。
- PDF 页数小于阈值。

示例阈值：

- 图片不超过 5MB。
- PDF 不超过 10 页。
- 同步超时时间 10 秒。

### 13.2 异步适用范围

- 多页扫描 PDF。
- 批量请求。
- OCR 或抽取耗时明显超过同步阈值的任务。

异步模式需要：

- 任务表。
- 可重试机制。
- 超时与取消机制。
- 可选 webhook。


## 14. 成本优化策略

### 14.1 路由优化

优先级如下：

1. 文本直取。
2. 轻量 OCR。
3. 轻量抽取模型。
4. 重 OCR / 重模型。
5. LLM fallback。

双轨路由优化：

1. 优先接收 SDK 预处理后的 `NormalizedDocument`。
2. 若无标准化输入，再走原始输入 parser。
3. 对同一 `NormalizedDocument + schema + options` 复用缓存。
4. 对同一原始输入 `parse_input + schema + options` 复用缓存。

### 14.2 缓存优化

- 同文件重复提交直接命中缓存。
- 相同 PDF 页内容复用 OCR。
- 相同 schema 和文档的重复抽取结果复用。

### 14.3 资源隔离

- API 服务和 OCR worker 分离。
- 抽取 worker 和 OCR worker 可独立扩容。

### 14.4 批处理优化

- 多页 PDF 页级并发处理。
- OCR 批量推理。
- 模型实例池复用。


## 15. 部署方案

### 15.1 MVP 部署

单机部署：

- Rust API
- Rust Worker
- SQLite
- 本地文件缓存
- 外部 OCR worker

适合：

- 早期验证。
- 内部试用。

### 15.2 生产部署

组件：

- API Pod
- Job Worker Pod
- OCR Worker Pod
- Postgres
- Redis/NATS
- 对象存储
- 监控组件

### 15.3 部署原则

- 生产环境优先 Linux。
- OCR 与模型文件镜像化。
- 版本和模型版本强绑定。
- 支持灰度切换 provider。

SDK 协议治理：

- `NormalizedDocument` 必须带协议版本或 `sdk_version`。
- 服务端应兼容至少一个旧版本协议。
- SDK 与服务端协议升级需要具备灰度发布能力。


## 16. 可观测性

### 16.1 日志

每个任务记录：

- task_id
- file_hash
- schema_hash
- parser_provider
- ocr_provider
- extractor_provider
- 总耗时
- 分阶段耗时
- 错误码

### 16.2 指标

建议指标：

- 请求量
- 成功率
- P50 / P95 / P99 延迟
- OCR 命中率
- 缓存命中率
- OCR 平均耗时
- 抽取平均耗时
- 低置信率

### 16.3 Trace

建议每个阶段生成 span：

- ingest
- parse
- ocr
- extract
- postprocess
- persist


## 17. 安全与治理

### 17.1 输入安全

- 限制上传大小。
- 校验 MIME 和文件头。
- 防止压缩炸弹和恶意文档。
- URL 拉取需要白名单或网络隔离。

### 17.2 数据安全

- 对象存储加密。
- 日志脱敏。
- 任务结果设置过期策略。

### 17.3 配额控制

- 按租户限流。
- 按文件大小计费或配额控制。


## 18. 风险与应对

### 18.1 OCR 质量不稳定

风险：

- 截图、手写、低清扫描件质量差。

应对：

- 多 OCR provider。
- 页级/块级置信度。
- 低置信回退策略。

### 18.2 模型导出兼容性不足

风险：

- 某些 UIE 模型无法稳定导出或在 ONNX Runtime 上表现不一致。

应对：

- 提前验证候选模型。
- 保留服务层 provider 抽象。
- 必要时允许单独 extractor worker 先独立实现。

### 18.3 PDF 类型复杂

风险：

- 混合文本层与扫描页、表格页、图片页并存。

应对：

- 页级判定是否 OCR。
- DocumentIR 保留页与 block 信息。

### 18.4 结果难解释

风险：

- 用户只看到错误字段，不知道为什么错。

应对：

- 返回 evidence。
- 提供调试接口。
- 保留 raw_text 和 block 来源。

### 18.5 SDK 结果漂移

风险：

- 不同平台、不同 SDK 版本可能产生不同的 OCR / 文本标准化结果。

应对：

- 在 `metadata.extra` 中写入 `sdk_version` 和 provider 信息。
- 设计兼容版本协议。
- 保留原始输入轨道作为服务端兜底。

### 18.6 外部 OCR worker 不稳定

风险：

- OCR worker 超时、网络抖动、版本漂移会导致图片链路成功率下降。
- 本地内置 OCR provider 也会带来模型体积、冷启动、内存峰值和平台兼容性风险。

应对：

- 通过 provider 抽象与主服务解耦。
- 用 `MUSE_OCR_TIMEOUT_MS` 控制单次调用上限。
- 用 `MUSE_OCR_WORKER_TOKEN` 做最小鉴权。
- 在元数据中记录 `ocr_provider / ocr_model / ocr_transport` 便于排障。
- 对本地 provider 增加模型存在性校验、启动预热和内存监控。
- 开发环境默认回退 `placeholder` provider。


## 19. 测试策略

### 19.1 单元测试

- schema 校验。
- 文件类型识别。
- parser 路由。
- 后处理规则。

### 19.2 集成测试

- 图片 OCR 到抽取全链路。
- 文本 PDF 直提取链路。
- 扫描 PDF OCR 链路。
- 缓存命中链路。
- HTTP OCR worker 成功响应与错误响应。
- 扫描 PDF 的 `page_rasters -> OCR -> DocumentIR` 集成链路。
- `multipart upload -> parser -> OCR -> extractor -> response` 的扫描 PDF API 集成链路。

### 19.3 回归评测

建立基准数据集：

- 小红书截图。
- 招聘 JD 截图。
- 扫描 PDF。
- 文本型 PDF。
- 中英混排文档。

评测指标：

- 字段级准确率。
- 文档级成功率。
- 平均延迟。
- 平均成本。


## 20. 目录结构建议

```text
docs/
  extraction-service-design.md

src/
  api/
  app/
  config/
  domain/
  ingestion/
  parser/
  ocr/
  extractor/
  postprocess/
  queue/
  storage/
  telemetry/
  main.rs
```

说明：

- `domain/` 放核心数据模型。
- `parser/ocr/extractor/` 放 provider trait 和实现。
- `app/` 负责服务编排。
- `telemetry/` 负责日志、metrics、trace。


## 21. 迭代路线

### Phase 1：MVP

目标：

- 支持 `image/pdf/text`
- 支持同步接口
- 能完成统一 schema 抽取
- 有基础缓存和日志

实现建议：

- Rust API
- PDFium
- 一个 OCR provider
- 一个 extractor provider
- SQLite

### Phase 2：服务化增强

目标：

- 异步任务
- 页级缓存
- webhook
- 监控指标
- provider 配置化
- `NormalizedDocument` 双轨输入

当前进度：

- 已实现 `NormalizedDocument` 标准化输入接口。
- 已实现 OCR provider 配置化，支持 `placeholder` 与 `http`。
- 已补齐 HTTP OCR worker 成功/失败测试。
- 已实现 SSE 提取事件流骨架。
- 已接入 ONNX Runtime CPU provider 的真实字符串 I/O 推理链路与 JSON 结果解码。
- 本地内置 OCR provider 已落地 `local-onnx` MVP，已具备图片前处理、det/cls/rec 基础编排与页级 OCR 元信息透传，但真实模型回归与稳定性验证仍在继续。

### Phase 3：效果和成本优化

目标：

- 低置信 fallback
- 多 OCR provider
- 批量处理
- 页级动态路由
- 官方 SDK 前处理能力

### Phase 4：平台化

目标：

- 多租户
- 配额
- 审计
- 控制台


## 22. 推荐结论

推荐采用如下落地路线：

1. 使用 Rust 构建统一提取服务。
2. 将图片、PDF、文档统一转换为 `DocumentIR`。
3. OCR 采用 provider 抽象，短期支持 HTTP worker，后续补齐本地内置 OCR 推理路径。
4. 信息抽取以文本抽取为核心路径，优先验证 ONNX 可部署模型。
5. 通过缓存、页级路由、同步/异步分流控制成本。
6. 通过 SDK 前处理和 `NormalizedDocument` 协议，将部分流量和计算成本前移到边缘端。

当前实现中的 OCR 配置项：

- `MUSE_OCR_PROVIDER=placeholder|http|local-onnx`
- `MUSE_OCR_WORKER_URL=http://host:port/v1/ocr`
- `MUSE_OCR_TIMEOUT_MS=5000`
- `MUSE_OCR_WORKER_TOKEN=<token>`
- `MUSE_OCR_MODEL_DIR=/path/to/ocr-models`
- `MUSE_OCR_THREADS=1`
- `MUSE_OCR_PREWARM=true|false`

当前实现中的 PDF 配置项：

- `MUSE_PDF_RASTER_PROVIDER=none|pdftoppm`
- `MUSE_PDFTOPPM_BIN=/usr/bin/pdftoppm`

当前实现状态说明：

- 目前完成的是 `Runtime Layer + Result Adapter Layer` 的 MVP 版本。
- `http` provider 已可视为真实运行时接入。
- `placeholder` 仅是开发兜底，不应视为生产 OCR。
- `local-onnx` 已完成启动配置、模型目录发现、图片解码/缩放/CHW 归一化前处理、ONNX Runtime session bootstrap、detector 输入张量构造与 session run、heatmap 连通域解码、recognition patch 裁片准备、recognition tensor 执行、以及可选预热。
- `local-onnx` 已可完成 `det + cls + rec + CTC decode` 的基础 OCR 推理，并可输出页级 `rotation_degrees`，但仍缺真实模型回归与稳定性验证。
- `Preprocess Layer` 目前仍较薄，后续应继续从 parser 中抽离并独立治理。
- PDF 已支持 `CompositePdfProvider` 组合模式：
  - 文本层路径：`lopdf`
  - 栅格化 fallback：可选 `pdftoppm`
  - 当前默认未强制开启 rasterizer，需要通过配置显式启用

当前 `local-onnx` 目录约定：

- 必需：`det.onnx` 或 `ocr_det.onnx` 或 `text_detection.onnx`
- 必需：`rec.onnx` 或 `ocr_rec.onnx` 或 `text_recognition.onnx`
- 必需：`ppocr_keys_v1.txt` 或 `dict.txt` 或 `ocr_keys.txt` 或 `keys.txt` 或 `charset.txt`
- 可选：`cls.onnx` 或 `ocr_cls.onnx` 或 `text_classification.onnx` 或 `text_direction.onnx`

当前 OCR worker 响应约定补充：

- `lines[].text`
- `lines[].confidence`
- `lines[].page_no`
- `lines[].bbox`

这些字段会继续传递到 `DocumentIR.pages[].blocks[]`，并在最终 evidence 中保留 `page_no / bbox / source_block_ids`。

当前实现中的 extractor 配置项：

- `MUSE_EXTRACTOR_PROVIDER=heuristic|onnx`
- `MUSE_ONNX_MODEL_PATH=/path/to/model.onnx`
- `MUSE_ONNX_MODEL_SPEC_PATH=/path/to/model.json`
- `MUSE_ONNX_THREADS=1`
- `MUSE_ONNX_INPUT_TEXT_NAME=text`
- `MUSE_ONNX_INPUT_SCHEMA_NAME=schema`
- `MUSE_ONNX_OUTPUT_JSON_NAME=json_output`

ONNX sidecar 约定：

- 若 `model.onnx` 同目录下存在 `model.json`，服务会自动加载。
- 也可以通过 `MUSE_ONNX_MODEL_SPEC_PATH` 显式指定 sidecar。
- 当前 sidecar 主要用于声明字符串输入输出名称和解码策略，避免将模型契约散落在环境变量中。
- 当前已能执行 `text/schema/json_output` 这一类字符串 I/O 真推理，并将结果解码回统一字段输出。

最终目标不是构建一个“模型演示程序”，而是构建一个“低成本、可扩展、可审计”的统一提取平台。


## 23. 当前进度总览

当前阶段判断：

- 方案成熟度：清晰，主链路已经确定。
- 工程阶段：`Alpha / 可跑的 MVP 服务`。
- 适合：内部验证、小范围接入、围绕真实样本继续迭代。
- 暂不适合：直接作为生产级多租户长期运行服务。

### 23.1 已完成

- 服务主链路已经成立：`ingestion -> parser -> extractor -> postprocess -> storage`。
- 已提供统一 API：同步提取、异步提取、标准化输入提取、任务查询、健康检查、版本接口。
- 已完成双轨输入：
  - 原始输入轨道：`image/pdf/docx/text/url`
  - 标准化输入轨道：`NormalizedDocument`
- `DocumentIR` 已具备页/块颗粒度，而不是只有一份纯文本：
  - 图片按 OCR 行生成 block
  - PDF 按页生成 block
  - docx 按段落生成 block
- OCR provider 抽象已经打通：
  - `placeholder`
  - `http`
  - `local-onnx bootstrap`
- OCR 证据链已经贯通：
  - `lines[].text / confidence / page_no / bbox`
  - 这些信息会传到 `DocumentIR.pages[].blocks[]`
  - 最终 evidence 会保留 `page_no / bbox / source_block_ids`
- PDF 已支持文本层优先和 OCR fallback。
- heuristic extractor 已可用，支持：
  - schema key 与 hints 命中
  - `string / number / boolean`
  - object/array 子字段聚合
  - evidence 回绑
- ONNX extractor 已具备可运行主链路：
  - sidecar 契约
  - session 启动
  - 字符串 I/O 真推理
  - JSON 输出解码
- SSE 流式事件已经成型：
  - `task.accepted`
  - `stage.changed`
  - `document.ready`
  - `page.parsed`
  - `page.ocr_done`
  - `block.extracted`
  - `result.partial`
  - `result.snapshot`
  - `cache.hit`
  - `completed`
  - `failed`
- `result.partial` 已支持按页面前缀修订，而不是只发一次最终字段。

### 23.2 半完成

- OCR 能力目前是“服务契约已完成，但本地真推理未完成”。
  - `http` provider 是真链路，但依赖外部 OCR worker
  - `placeholder` 只是开发兜底，不是生产 OCR
  - `local-onnx` 已完成图片前处理、detector 输入张量构造、session run、候选 bbox 解码、recognition patch 裁片准备、recognition tensor 执行与 CTC decode，但仍缺真实模型回归与稳定性验证
- ONNX 能力目前是“主链路已打通，但模型通用性还不够完整”。
  - 字符串 I/O 路线已完成
  - tokenized/UIE 路线已具备代码路径
  - 但 tokenized/UIE 仍缺真实模型集成验证，离生产完成态还有距离
- 异步任务能力目前是“接口可用，但底层还是进程内实现”。
  - queue 是 in-memory
  - storage 是 in-memory
  - 更偏 MVP，不是生产级任务系统
- 缓存能力目前是“命中和元数据已完成，但治理未完成”。
  - 已有 `hit_count / created_at_ms / last_accessed_at_ms`
  - 还没有 TTL、淘汰、租户隔离

### 23.3 未完成

- 本地内置 OCR provider：
  - `local-paddle`
  - 或本地 C++/Rust OCR runtime
- `local-onnx` 的真实 OCR inference 与 bbox/line 结果解码。
- 持久化 storage：
  - Postgres / SQLite
  - 结果表、任务表、缓存索引
- 生产级 queue：
  - Redis / NATS / RabbitMQ
- 页级 OCR 缓存与更细粒度成本路由。
- 更完整的可观测性：
  - metrics
  - tracing
  - dashboard
- Webhook / WebSocket / 控制台等交互层能力。
- 多租户、限流、配额、数据保留策略。

### 23.4 当前最优先下一步

如果按“最快把产品做成可用服务”排序，我建议是：

1. 补齐本地 OCR provider
2. 落地持久化 storage + queue
3. 选定一条稳定 ONNX 模型链路并做针对性适配
4. 增加页级 OCR 缓存和任务超时/重试

原因：

- 没有本地 OCR provider，当前图片能力仍然依赖外部 worker。
- 没有持久化 storage/queue，异步任务还不具备真正生产可用性。
- ONNX 虽然已经前进很多，但要先围绕一个明确模型收敛，不然容易陷入适配扩张。

### 23.5 待确认问题

下面这些问题已经不是“能不能开工”的问题，而是“接下来优先级怎么排”的问题：

1. 第一批核心业务场景具体有哪些，字段 schema 是否稳定。
2. 对时延的要求是在线实时优先，还是批量吞吐优先。
3. 是否接受 OCR 作为独立 worker 长期存在，还是必须收敛到本地内置 OCR。
4. 第一阶段是否要把证据框坐标稳定暴露给前端产品层。
5. 是否需要租户级缓存隔离、结果过期和数据保留策略。


## 24. 已完成

### 24.1 完成快照

- `API / 编排`
  状态：已完成 MVP
  说明：同步、异步、标准化输入、任务查询、SSE 事件流都已具备。
- `Parser / DocumentIR`
  状态：已完成 MVP
  说明：原始输入与标准化输入已统一收敛到 `DocumentIR`，且保留页/块颗粒度。
- `OCR`
  状态：半完成
  说明：抽象、HTTP worker 协议、证据链贯通已完成；`local-onnx` 已具备 detector 可运行链路、recognition patch 准备、recognition tensor 执行与基础文本解码，但真实模型验证仍未补齐。
- `Extractor / Heuristic`
  状态：已完成 MVP
  说明：可覆盖当前原型级字段抽取需求，并具备 evidence 回绑。
- `Extractor / ONNX 字符串 I/O`
  状态：已完成
  说明：已具备 sidecar、session、输入构造、运行、JSON 输出解码全链路。
- `Extractor / ONNX tokenized/UIE`
  状态：半完成
  说明：代码路径已实现，但真实模型集成验证和回归基线还未补齐。
- `Storage / Queue / Cache`
  状态：半完成
  说明：MVP 能跑，但目前仍是 in-memory，缺持久化与治理能力。
- `可观测性 / 生产治理`
  状态：未完成
  说明：日志已具备基础信息，但 metrics、tracing、配额、租户治理还未完成。

- Rust 服务骨架、同步/异步接口、任务查询、健康检查、版本接口。
- `image/pdf/docx/text/url` 原始输入轨道。
- `NormalizedDocument` 标准化输入轨道与协议校验。
- 图片 OCR provider 抽象，现支持 `placeholder`、`http` worker 与 `local-onnx bootstrap`。
- 本地 OCR 已具备统一抽象边界，以及 `local-onnx` 的目录发现、图片前处理、detector 输入张量构造、候选 bbox 解码、classification patch 旋转编排、recognition patch 准备、recognition tensor 执行、CTC 文本解码、session 启动与预热能力。
- OCR 内部已按 `Preprocess / Runtime / Result Adapter` 三层拆出 MVP 结构。
- OCR metadata 已区分 `ocr_provider` 与 `ocr_transport`，`/version` 也已暴露 transport。
- 已支持 `MUSE_OCR_FALLBACK_PROVIDER`，可在主 OCR provider 失败时自动回退到备用 provider。
- PDF 文本层优先与 OCR fallback。
- 扫描 PDF 已新增 `raster_pages` 契约，parser 会优先对页图逐页 OCR，并在 metadata 中记录 `pdf_ocr_input=page_rasters|original_pdf_bytes`。
- `/version` 已暴露 `pdf_raster_provider`，扫描 PDF metadata 也可记录 `pdf_raster_provider` 便于排障。
- `DocumentIR` 页/块颗粒度增强：图片按 OCR 行、PDF 按页、docx 按段落保留结构。
- OCR 页级元信息已可透传到 `DocumentIR.pages[]` 的 `width/height`，并通过 metadata 与 SSE 事件进一步透传 `rotation_degrees`，可直接用于前端 bbox 叠加与扫描页方向展示。
- heuristic extractor、基础类型转换、object/array 子字段聚合。
- ONNX Runtime CPU provider 配置入口、sidecar 约定、字符串 I/O 真推理与 JSON 输出解码。
- tokenized/UIE ONNX 路线的基础代码路径已经具备：
  - tokenizer 加载
  - prompt 线性化
  - 长文滑窗
  - 张量输入构造
  - span 输出解码
  - evidence 基础回绑
  - 输入滑窗 / 多窗去重 的基础回归测试
- postprocess 字段清洗、evidence 去重、结果归一化。
- in-memory task store 与带 `hit_count` 的结果缓存。
- SSE 事件流骨架、`stream_url` 返回、逐页修订的 `result.partial`、`page.ocr_done` 与 `block.extracted` 事件。
- 已引入共享 fixture asset manifest，统一管理 `fixtures/assets/{images,pdfs,docx}` 真实样本、文件名和 MIME 信息，避免 parser/API 回归样例重复维护。

## 25. TODO

### 25.0 执行看板

- `P0`
  目标：把服务从“可跑 MVP”推进到“可稳定内测”
  范围：本地 OCR、持久化、缓存治理、真实 ONNX 集成验证
- `P1`
  目标：把服务从“可稳定内测”推进到“可持续扩展”
  范围：Postgres、多实例任务、Webhook、真实 tokenized/UIE 模型回归
- `P2`
  目标：把服务从“可持续扩展”推进到“可平台化运营”
  范围：WebSocket、控制台、多租户、配额与数据治理

### 25.1 本地内置 OCR 推理

状态：

- 当前优先级：`P0`
- 当前风险：高
- 当前收益：高

目标：

- 在保留 `http` OCR worker 的同时，补齐进程内本地 OCR 推理能力。
- 让单机部署、离线环境和内网环境不再强依赖外部 OCR 服务。
- 保持 `OcrProvider -> OcrOutput` 契约稳定，不影响 parser 与上层服务编排。

设计原则补充：

- OCR 的变化面应尽量收敛在 `Preprocess / Runtime / Result Adapter` 三层内部。
- 不论底层接的是 Paddle、RapidOCR、ONNX 还是云 OCR，上层 parser 都不应感知具体差异。
- `DocumentIR` 和 evidence 回绑逻辑应继续只依赖统一的 `OcrOutput`。

建议拆成三层：

1. `Image Preprocess Layer`：负责缩放、归一化、旋转纠正、必要的页图切分。
2. `Local Runtime Layer`：负责 ONNX Runtime 或本地 C++ OCR 引擎生命周期、模型加载、线程数与预热。
3. `Result Adapter Layer`：负责把 OCR 检测/识别结果映射为统一 `OcrBlock[] + OcrLine[]`，并附带 `block_id / page_no / bbox / confidence`。

建议新增 provider：

```rust
pub struct LocalOnnxOcrProvider {
    // detector / recognizer / classifier sessions
}
```

建议配置项：

- `MUSE_OCR_PROVIDER=local-onnx`
- `MUSE_OCR_FALLBACK_PROVIDER=http|placeholder`
- `MUSE_OCR_MODEL_DIR=/path/to/ocr-models`
- `MUSE_OCR_THREADS=1`
- `MUSE_OCR_PREWARM=true|false`

当前已完成：

- `Config` 与 `AppState` 已支持 `local-onnx` provider 选择。
- 已完成模型目录发现与必需文件校验。
- 已完成图片解码、限边长缩放与 RGB->CHW 浮点张量前处理。
- 已完成 detector 的输入张量拼装与单输入 session run。
- 已完成 detector heatmap 的连通域解码，并可回映射到原图 bbox。
- 已完成 classification session bootstrap、分类输出解码与基于 patch 的 0/90/180/270 旋转编排主路径。
- 已完成 bbox 到 recognition patch 的基础裁片与缩放准备。
- 已完成 recognition session 的单 patch 张量执行与输出摘要。
- 已完成基于 charset sidecar 的 CTC greedy 文本解码主路径。
- 已完成基础 `OcrLine[]` 结果组装与 bbox/confidence/page_no 回填。
- 已具备可注入 backend 的 runtime 单测能力，便于脱离真实 ONNX 模型验证拼装链路。
- 已补充 fixture 驱动的 OCR 回归测试，可覆盖 `det -> rec -> decode -> OcrLine[]` 基础链路。
- 已完成 ONNX Runtime CPU session bootstrap。
- 已支持 `prewarm` 阶段对 detection / recognition / classification session 做基础自检。
- 已完成 `ocr_provider / ocr_transport` 元信息分离，并在 `/version` 与 `DocumentIR.metadata.extra` 中统一暴露。
- 已支持按配置在主 OCR provider 失败时自动回退到备用 provider。
- 已为扫描 PDF 补充 `raster_pages` 输入契约与逐页 OCR 聚合逻辑，并有集成测试覆盖多页页号回填。
- 已补充上传接口级扫描 PDF 集成测试，覆盖 `multipart upload -> parser -> OCR -> extractor -> response` 主链路。
- 已补充 `CompositePdfProvider` 单测，覆盖“有文本层不栅格化 / 无文本层走 rasterizer”两条路径。
- 已新增 fixture 驱动的扫描 PDF、文本层 PDF、图片、docx 上传成功回归、图片上传缓存命中回归、同步 OCR 失败回归，以及异步 OCR 失败、异步 PDF raster 失败回归样例，开始建立可扩展的 golden regression 基线。
- 上传回归已不再只依赖测试内联生成 bytes，仓库中已加入真实 `fixtures/assets/{images,pdfs,docx}` 样本资产，便于排查和后续扩展真实样本基线。
- 已新增共享 `fixtures/assets/manifest.json` 与测试辅助加载器，统一 parser/API 两套 fixture 对真实资产的引用方式，减少 `asset_path / file_name / content_type` 的重复声明。
- parser 层也已补充 fixture 驱动的真实 `image / scanned pdf / text-layer pdf / docx` 样本回归，开始把 `asset -> parser -> DocumentIR` 单独固定下来。
- 同步执行失败现在也会落任务失败状态并发布失败事件，便于和异步路径保持一致的排障语义。

当前未完成：

- classification 的真实模型验证与稳健性回归。
- 围绕真实模型样本的识别质量回归与解码稳健性验证。
- 基于真实 OCR 模型而非 stub/provider 的真实图片或扫描 PDF 端到端集成回归。

落地要点：

- 如果 OCR 模型拆成检测/方向分类/识别多个 ONNX 文件，要在启动时一次性校验齐全。
- 统一输出 `lines[].text / confidence / page_no / bbox`，避免和 `http` provider 走两套协议。
- 在 `DocumentIR.metadata.extra` 中记录 `ocr_provider / ocr_model / ocr_transport=inproc`。
- 扫描 PDF 的正确 OCR 输入应是逐页 raster image，而不是直接把整份 PDF bytes 传给 image OCR provider。
- `raster_pages` 契约需要校验：`page_no >= 1`、页号不重复、页图 bytes 非空。
- 对大图和扫描 PDF 需要限制单页最大分辨率，避免内存峰值失控。
- 本地 provider 失败时，允许按配置回退 `http` worker 或 `placeholder`。

验收标准：

- 不依赖外部 OCR worker 也能完成图片 OCR 到抽取全链路。
- `/version` 能暴露当前 OCR provider 名称与 transport。
- 至少有一组真实图片或扫描 PDF 集成测试覆盖“输入 -> OCR -> DocumentIR -> extractor”链路。

下一动作建议：

1. 先围绕真实模型补识别质量回归和解码稳健性验证。
2. 再评估 classification 路径是否真的还需要保留。
3. 最后补真实图片集成测试，而不是先追求多 provider。

### 25.2 ONNX 运行时增强

状态：

- 当前优先级：`P0-P1`
- 当前风险：中
- 当前收益：高

目标：

- 已落地 ONNX session / input adapter / output adapter 的基础抽象，继续演进更通用的 ONNX 输入输出适配层。
- 保持当前 provider 抽象不变，避免把服务层绑死在某一种模型输入格式上。

与 OCR 的关系：

- extractor 侧建议拆 `Session Layer / Input Adapter Layer / Output Adapter Layer`。
- OCR 侧建议拆 `Preprocess Layer / Runtime Layer / Result Adapter Layer`。
- 两者的共同目标都是把“模型/引擎差异”限制在适配层，不向 API、parser、`DocumentIR` 和 storage 扩散。

当前状态总结：

- `25.2.1` 可视为已完成。
- `25.2.2` 可视为基础链路已打通：
  - tokenizer + tensor 输入构造已实现
  - float tensor 输出读取已接通
  - span 解码与 synthetic 单测已覆盖
  - 真实模型集成验证仍未完成
- `25.2.3` 基础能力已完成，但仍缺真实模型回归测试和生产验证。

建议拆成三层：

1. `Session Layer`：负责 ONNX Runtime environment、session 生命周期、线程数与模型热加载。
2. `Input Adapter Layer`：负责把 `DocumentIR + SchemaSpec` 转换成模型输入。
3. `Output Adapter Layer`：负责把模型输出解码成统一字段结果，并回填 evidence。

建议新增抽象：

```rust
pub trait OnnxInputAdapter {
    fn build_inputs(&self, doc: &DocumentIr, schema: &SchemaSpec) -> anyhow::Result<OnnxInputs>;
}

pub trait OnnxOutputAdapter {
    fn decode(
        &self,
        output: OnnxOutputs,
        doc: &DocumentIr,
        schema: &SchemaSpec,
    ) -> anyhow::Result<ExtractionResult>;
}
```

#### 25.2.1 第一阶段：字符串 I/O 模型打通

适用模型：

- 输入直接接受 `text` 和 `schema` 两个字符串。
- 输出直接返回 `json_output` 字符串。

当前这一阶段已经落地：

- sidecar 读取。
- 输入输出名称校验。
- ONNX Runtime session 启动校验。
- 已通过 `Session Layer / Input Adapter Layer / Output Adapter Layer` 抽象承载当前字符串 I/O 路线。
- 按 batch size = 1 构造字符串输入张量。
- 执行 ONNX Runtime `Run(...)`。
- 将输出字符串解析为 JSON。
- 将 JSON 映射为 `FieldValue[]`。
- 当 JSON 缺失字段时，按 schema 生成空结果或默认值。

sidecar 建议继续保留以下最小字段：

```json
{
  "protocol_version": 1,
  "input_text_name": "text",
  "input_schema_name": "schema",
  "output_json_name": "json_output",
  "decode_strategy": "json_string_scalar"
}
```

当前解码规则：

- 输出必须是单个 JSON 字符串。
- JSON 顶层允许是 `{"fields": [...]}` 或直接是对象。
- 若模型直接返回对象，则按 schema key 读取。
- 若模型返回数组形式字段列表，则统一归一为 `FieldValue[]`。
- evidence 支持优先使用返回的 `source_block_ids`，缺失时退化为基于文本片段的 block 匹配。

失败策略：

- `session.run(...)` 失败时直接返回 extractor error，不伪装成成功。
- 可通过配置决定是否允许回退 heuristic extractor。
- 回退发生时必须在日志和任务 message 中明确标记 `extractor_fallback=heuristic`。

#### 25.2.2 第二阶段：tokenizer + tensor 模型接入

适用模型：

- UIE / 类 UIE 模型。
- 输入不是字符串，而是 `input_ids / attention_mask / token_type_ids` 等张量。

建议把 tokenizer 与张量拼装做成单独 adapter，不污染当前字符串 I/O 路线。

建议 sidecar 扩展：

```json
{
  "protocol_version": 1,
  "runtime_contract": "tokenized",
  "tokenizer_path": "./tokenizer.json",
  "max_length": 512,
  "inputs": {
    "input_ids": "input_ids",
    "attention_mask": "attention_mask",
    "token_type_ids": "token_type_ids"
  },
  "outputs": {
    "start_probs": "start_probs",
    "end_probs": "end_probs"
  },
  "decode_strategy": "uie_span"
}
```

落地要点：

- tokenizer 文件与模型文件一起版本化。
- sidecar 已支持声明 `runtime_contract=tokenized`、tokenizer 路径和张量输入输出名。
- 当前代码已经具备 tokenized 路线的基础能力：
  - tokenizer 加载与 padding/truncation 配置
  - prompt 线性化
  - 长文滑窗
  - `input_ids / attention_mask / token_type_ids` 张量构造
  - `start_probs / end_probs` 输出解码
  - span 到 evidence 的基础回绑
  - 多窗口重叠场景下的基础去重回归测试
- 但这一条链路还缺少“真实模型集成验证”这一层，因此当前状态应视为“已实现代码路径，但尚未完成生产验证”，不是最终完成态。
- schema 需要先线性化成 prompt 或 instruction 文本。
- 文本切片策略必须支持长文分页或滑窗，不能把全部 `plain_text` 强塞到单次推理。
- 输出解码时需要保留 token offset 到原文字符区间的映射，便于 evidence 回绑。

长文策略建议：

- 对 `DocumentIR` 按 page 或 block 聚合成 chunk。
- 每个 chunk 独立推理。
- 后处理阶段做跨 chunk 合并、去重和置信度聚合。

#### 25.2.3 输出解码与 evidence 回绑

无论字符串 I/O 还是 tensor I/O，最终都要回到统一结果模型：

- 字段值必须映射到 `FieldValue.value`。
- 置信度必须写入 `FieldValue.confidence`。
- 证据必须尽量绑定到 `source_block_ids`，而不是只保留纯文本。

推荐回绑流程：

1. 先定位模型输出对应的原文片段。
2. 再在 `DocumentIR.pages[].blocks[]` 中做 block 级匹配。
3. 若找到多个 block，保留全部 `source_block_ids`。
4. 若无法精确定位，允许只返回 `page_no + text`，但要把 `bbox` 置空。

验收标准：

- `onnx` provider 在成功路径下不再 fallback 到 heuristic extractor。
- `/version` 能正确暴露当前 ONNX provider 名称。
- 至少有一组真实字符串 I/O 模型集成测试覆盖“输入 -> ONNX -> JSON -> evidence”完整链路。
- 至少有一组真实 tokenized/UIE 模型集成测试覆盖“输入 -> tokenizer -> tensor -> span decode -> evidence”完整链路。

下一动作建议：

1. 先固定一组真实字符串 I/O 模型做集成样本。
2. 再固定一组真实 tokenized/UIE 模型做回归样本。
3. 最后再考虑把更通用的模型适配抽象进一步产品化。

### 25.3 持久化存储替换 In-Memory Store

状态：

- 当前优先级：`P0`
- 当前风险：高
- 当前收益：高

目标：

- 服务重启后任务、结果、缓存元数据、事件历史仍可查询。
- 为异步任务、Webhook 回调、控制台查询打基础。

建议分两步：

#### 25.3.1 MVP：SQLite 持久化

适用场景：

- 单机部署。
- 内部试用。
- 低并发异步任务。

建议表结构：

- `tasks`
- `task_results`
- `task_events`
- `cache_entries`
- `artifacts`（可选，用于原始输入或标准化文档落盘索引）

建议字段：

```text
tasks:
  task_id PK
  tenant_id
  status
  message
  created_at_ms
  updated_at_ms

task_results:
  task_id PK
  result_json
  raw_text
  parse_ms
  extract_ms
  postprocess_ms
  total_ms

task_events:
  task_id
  sequence
  event_type
  payload_json
  created_at_ms

cache_entries:
  cache_key PK
  tenant_id
  namespace
  result_json
  created_at_ms
  last_accessed_at_ms
  expires_at_ms
  hit_count
```

接口层建议调整：

- `ExtractionStore` 继续保留 `upsert / get / get_cached / put_cached`。
- 补充 `append_event / list_events / delete_expired_cache`。
- 后续如果引入 worker 抢占，再补 `lease_task / renew_lease / fail_task`。

#### 25.3.2 生产版：Postgres

目标：

- 多实例共享状态。
- 支撑更稳定的异步 worker 与控制台查询。

建议：

- 任务表与结果表继续保留同样逻辑模型，避免 SQLite 到 Postgres 迁移时重写上层代码。
- 事件表按 `(task_id, sequence)` 建联合索引。
- 缓存表按 `tenant_id + namespace + expires_at_ms` 建索引，便于定时清理。

迁移顺序建议：

1. 先把当前 `InMemoryStorage` 抽象成可插拔实现。
2. 先落 SQLite，并让 SSE 历史从持久化事件表读取。
3. 再切换到 Postgres。

验收标准：

- 服务重启后 `GET /v1/extractions/{task_id}` 仍能查到历史结果。
- 事件流补连时能从持久化事件表回放历史。
- 异步任务执行过程中服务异常退出，不会让任务状态永久停在 `extracting` 且不可恢复。

下一动作建议：

1. 先上 SQLite，不要直接跳 Postgres。
2. 优先持久化 `tasks/results/events/cache` 四类核心数据。
3. 先让 SSE 历史回放依赖持久化事件表，再继续扩展 worker 协议。

### 25.4 缓存淘汰、TTL 与租户隔离

状态：

- 当前优先级：`P0`
- 当前风险：中
- 当前收益：中高

目标：

- 让缓存从“可命中”升级为“可治理”。
- 控制磁盘/数据库占用，避免不同租户之间互相污染结果。

建议缓存继续按三层治理：

- `parse` 缓存
- `ocr` 缓存
- `extract` 缓存

每层建议独立 TTL：

- `parse`：7 天
- `ocr`：7 到 30 天
- `extract`：1 到 7 天

淘汰策略建议：

- 先按 TTL 过期。
- 再按 namespace 内的 LRU 或 LFU 兜底淘汰。
- 当达到容量上限时，优先淘汰“长时间未访问且 hit_count 低”的缓存项。

租户隔离建议：

- 所有任务与缓存记录必须显式带 `tenant_id`。
- 缓存键从

```text
extract:{doc_hash}:{schema_hash}:{extractor_version}
```

扩展为

```text
extract:{tenant_id}:{doc_hash}:{schema_hash}:{extractor_version}
```

若后续确认允许跨租户共享“纯公共模板文档”缓存，可增加白名单模式，但默认不共享。

治理补充：

- 缓存命中日志必须包含 `tenant_id / namespace / hit_count / expires_at_ms`。
- 对 OCR 和抽取缓存建议支持后台定时清理任务。
- 当 schema 版本升级时，应允许按 `schema_hash` 精确失效，而不是全量清空。

验收标准：

- 缓存可配置 TTL。
- 缓存容量达到上限后不会无限增长。
- 不同租户提交相同文档时不会读到彼此的任务记录与缓存结果。

下一动作建议：

1. 先做 TTL 和后台清理。
2. 再做 namespace 分层。
3. 最后再补 tenant 级隔离和共享策略。

### 25.5 Webhook / WebSocket / 控制台

状态：

- 当前优先级：`P1-P2`
- 当前风险：低
- 当前收益：中

目标：

- 补齐异步任务的对外交互闭环。
- 让用户不只依赖轮询和临时 SSE。

优先级建议：

1. Webhook
2. 控制台查询页
3. WebSocket

#### 25.5.1 Webhook

建议能力：

- 创建异步任务时允许传 `callback_url`。
- 任务 `succeeded / failed / partial_success` 时回调。
- 至少支持 3 次重试和指数退避。
- 通过签名头保证最小安全性。

建议回调体：

```json
{
  "task_id": "task_123",
  "status": "succeeded",
  "attempt": 1,
  "result": {"...": "..."},
  "occurred_at_ms": 1710000001300
}
```

#### 25.5.2 WebSocket

不是第一阶段必需。

建议：

- 如果后续要支持前端控制台多任务并发监听，再引入 WebSocket。
- 事件 envelope 应与当前 SSE 保持一致，避免维护两套事件协议。

#### 25.5.3 控制台

建议最小页面：

- 任务列表页。
- 任务详情页。
- 原文 / block / evidence 对照页。
- provider 与阶段耗时调试信息页。

控制台最小价值：

- 看任务进度。
- 看字段结果与 evidence。
- 看失败原因与阶段耗时。
- 看缓存是否命中。

### 25.6 建议落地顺序

P0：

- SQLite 持久化 `tasks/results/events/cache`。
- 缓存 TTL 与后台清理任务。
- 为字符串 I/O ONNX 路线补真实模型集成测试与观测指标。
- 本地 `local-onnx` OCR provider 最小闭环。

P1：

- Postgres 存储实现。
- Webhook 回调。
- 为 tokenized/UIE ONNX 路线补真实模型集成测试与回归集。

P2：

- WebSocket。
- 控制台。
- 更细粒度的租户治理与缓存共享策略。

完成定义：

- 文档中的 TODO 至少要落成对应的模块接口、配置项和集成测试，不以“预留配置项”视为完成。
- 所有新增能力都需要在 `/version` 或可观测性指标中暴露当前启用状态。
