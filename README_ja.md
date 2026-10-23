# try_LLM_MoE — 日英 Wikipedia で自作 MoE LLM を学習して GGUF を出力する

[English](README.md) | **日本語**

日本語版と英語版の Wikipedia を使い、Mixture of Experts (MoE) 型の言語モデルをゼロから学習します。
完成したモデルはトークナイザも含めて **1 つの GGUF ファイル** にまとめて出力します。

GGUF は Mixtral 互換の構造（`general.architecture = llama` ＋ `llama.expert_count`）で書き出すため、
**llama.cpp でそのまま読み込んで推論できます**（llama.cpp b11756 で動作を確認済み）。

## 全体の流れ

```
download_wiki.py     Wikipedia の Parquet を data/raw/ に保存 (一度だけ。以降は再利用)
      ↓
prepare_tokenizer.py SentencePiece の語彙データを学習       -> data/tokenizer.model
      ↓
prepare_data.py      日英混合のトークン列バイナリを作成       -> data/train.bin, val*.bin, meta.json
      ↓
train.py             MoE モデルを学習                         -> out/ckpt.pt, out/ckpt_best.pt
      ↓
export_gguf.py       チェックポイント + 語彙データ -> GGUF    -> out/wiki-moe-ja-en-f16.gguf
      ↓
verify_gguf.py       GGUF の中身を表示し、元モデルとの一致を確認 (--llama_cpp 指定時は llama.cpp での推論も)
      ↓
eval_facts.py        (任意) 事実の穴埋めテストで回答精度を測る -> out/eval_facts_baseline.json
```

`run_pipeline.ps1` を実行すると、`download_wiki.py` から `verify_gguf.py` までの工程をまとめて実行できます（llama.cpp での推論確認と `eval_facts.py` は含まれないので、必要なら個別に実行します）。

## 動作環境

このドキュメントの実測値は、すべて次の環境で計測したものです。

| 項目 | 動作確認した環境 | 補足 |
|---|---|---|
| OS | Windows 11 Pro（64 ビット） | `run_pipeline.ps1`・`demo.ps1` は PowerShell スクリプトで、コマンド例のパスも Windows 形式です。Linux・macOS では動作を確認していません |
| シェル | PowerShell | 実行ポリシーで止められる場合は `-ExecutionPolicy Bypass` を使います（後述） |
| Python | 3.14.0（64 ビット） | |
| GPU | NVIDIA GeForce RTX 4070 Ti SUPER（VRAM 16GB） | 学習には CUDA 対応の NVIDIA GPU が事実上必須です。12GB でもバッチを小さくすれば動きます（「学習時間の目安」を参照） |
| NVIDIA ドライバ | 616.92 | CUDA 12.8 に対応している必要があります |
| PyTorch | 2.11.0 + CUDA 12.8（`cu128`） | `requirements.txt` とは別にインストールします |
| その他の Python パッケージ | numpy 2.5.4、datasets 5.1.0、huggingface_hub 2.2.0、pyarrow 26.0.0、sentencepiece 0.2.2、gguf 0.19.0、tqdm 4.70.1 | `requirements.txt` |
| ディスクの空き容量 | 20GB 以上を推奨 | 本番実行時の実測：Wikipedia の Parquet 約 6.0GB、`data/*.bin` 約 2.0GB、`out/` 約 4.6GB（チェックポイント約 2.1GB × 2 ＋ GGUF 356MB） |
| ネットワーク | 初回のダウンロード時に必要 | Hugging Face から約 5.4GB。2 回目以降は不要です |

GGUF を使う場合の任意のツール：

| ツール | 動作確認したバージョン |
|---|---|
| llama.cpp | b11756（`win-cpu-x64`。`win-cuda` 版もあります） |
| Ollama | 0.40.6 |

macOS（Apple Silicon）で実行する場合は、「[macOS で実行する](#macos-で実行する)」を参照してください。

## セットアップ

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python -m pip install -r requirements.txt
```

`python` には Python 3.14 を使います。スクリプトはすべて `.venv` の Python で実行する前提です（`run_pipeline.ps1` と `demo.ps1` も `.venv\Scripts\python.exe` を直接呼び出します）。

## 実行方法

### まとめて実行する

```powershell
.\run_pipeline.ps1 -Smoke            # 動作確認用。数分で終わる (出力先: smoke\)
.\run_pipeline.ps1                   # 本番用。small プリセットで 20000 ステップ
.\run_pipeline.ps1 -SkipTo train     # 指定したステップから再開する
```

実行ポリシーで止められる場合は、`powershell -ExecutionPolicy Bypass -File .\run_pipeline.ps1` で実行してください。

### 1 ステップずつ実行する

```powershell
$py = ".\.venv\Scripts\python"
& $py download_wiki.py --en_shards 8            # 日本語は全 15 ファイル (3.9GB)、英語は 8/41 ファイル (約 2.3GB)
& $py prepare_tokenizer.py                      # 語彙 32000。所要 10〜20 分
& $py prepare_data.py --ja_tokens 5e8 --en_tokens 5e8
& $py train.py --preset small --max_steps 20000
& $py generate.py --prompt "日本の首都は" --show_routing
& $py export_gguf.py --ckpt out\ckpt_best.pt --outtype f16
& $py verify_gguf.py out\wiki-moe-ja-en-f16.gguf --ckpt out\ckpt_best.pt
& $py verify_gguf.py out\wiki-moe-ja-en-f16.gguf --llama_cpp D:\tools\llama.cpp   # (任意) llama.cpp でも推論して比較
& $py eval_facts.py --ckpt out\ckpt_best.pt --ckpt out\ckpt.pt --json out\eval_facts_baseline.json   # (任意) 回答精度
```

`--llama_cpp` には、llama.cpp の配布物（`llama-completion.exe` など）を展開したフォルダを指定します。

## macOS で実行する

Apple Silicon の Mac では、`macosx/` の Mac 用スクリプトを使います。学習には Apple GPU（MPS）を使います。

> **注意：** Mac 用スクリプトは Windows 上で CPU を使って動作を確認しただけで、実機の Mac では試していません。

### 想定環境

| 項目 | 想定 |
|---|---|
| Mac | M2 Pro（メモリ 32GB）または M4（メモリ 32GB） |
| macOS | 14 Sonoma 以降（MPS で bf16 を使うため） |
| Python | 3.14（python.org 版または Homebrew 版） |
| PyTorch | 2.11（MPS 対応の標準版） |

### セットアップ

```bash
python3.14 -m venv .venv
.venv/bin/python -m pip install torch
.venv/bin/python -m pip install -r requirements.txt
chmod +x macosx/*.sh    # 実行権限が付いていない場合
```

### 実行方法

コマンドはリポジトリのルートで実行します。オプション名は Windows 版と同じ意味で、小文字のハイフン区切りです。

```bash
./macosx/run_pipeline.sh --smoke             # 動作確認用 (出力先: smoke/)
./macosx/run_pipeline.sh                     # 本番用。small プリセットで 20000 ステップ
./macosx/run_pipeline.sh --skip-to train     # 指定したステップから再開する
./macosx/run_pipeline.sh --batch-size 4 --grad-accum 16   # メモリが足りない場合 (実効バッチは同じ 64)
./macosx/demo.sh                             # 発表用の実演 (demo.ps1 と同じ 7 ステップ)
```

1 ステップずつ実行する場合は、学習・生成・回答精度テストだけ `macosx/` の版を使います。それ以外のスクリプトは Windows と共通です。

```bash
py=.venv/bin/python
$py download_wiki.py --en_shards 8
$py prepare_tokenizer.py
$py prepare_data.py --ja_tokens 5e8 --en_tokens 5e8
$py macosx/train.py --preset small --max_steps 20000
$py macosx/generate.py --prompt "日本の首都は" --show_routing
$py export_gguf.py --ckpt out/ckpt_best.pt --outtype f16
$py verify_gguf.py out/wiki-moe-ja-en-f16.gguf --ckpt out/ckpt_best.pt
$py macosx/eval_facts.py --ckpt out/ckpt_best.pt --json out/eval_facts_baseline.json
```

### Windows 版との違い

| ファイル | 変更点 |
|---|---|
| `macosx/train.py` | Apple GPU（MPS）を使う（使えなければ CPU。その場合は `--dtype fp32` に切り替え）。`--dtype fp16` 用の GradScaler を MPS 向けに作成 |
| `macosx/generate.py`、`macosx/eval_facts.py` | Apple GPU（MPS）を使う（使えなければ CPU） |
| `macosx/moe_model.py` | RoPE を複素数ではなく cos/sin の実数演算で計算する（MPS では複素数演算の対応が不完全なため）。計算結果は同じです |
| `macosx/run_pipeline.sh`、`macosx/demo.sh` | `run_pipeline.ps1`・`demo.ps1` の bash 版（macOS 標準の bash 3.2 でも動くように、bash 4 以降の機能を使っていません）。学習には `macosx/train.py` を使う |

- チェックポイントと GGUF は Windows 版と共通の形式で、互いに読み込めます。Windows で学習したチェックポイントでも、`macosx/moe_model.py` との出力の差は最大 0.00002 程度です（`out/ckpt_best.pt` で確認）。
- `.venv` 以外の Python を使う場合は、環境変数 `PYTHON` でパスを指定します（例：`PYTHON=python3.14 ./macosx/run_pipeline.sh --smoke`）。
- llama.cpp は、[リリースページ](https://github.com/ggml-org/llama.cpp/releases) の `macos-arm64` 版などを使います。`demo.sh` では `--llama-cpp-dir` で場所を指定できます（省略時は PATH から探します）。

### メモリと学習時間の目安

- small プリセットの既定値（`--batch_size 8`）は、RTX 4070 Ti SUPER で VRAM を約 10GB 使いました。メモリ 32GB の Mac なら収まる見込みです。足りない場合は、`--batch-size 4 --grad-accum 16` のように実効バッチを保ったまま減らします。
- Mac での学習速度は計測していません。RTX 4070 Ti SUPER より大幅に遅くなると見込まれるので、まず `--smoke` で速度（`tok/s`）と残り時間（`ETA`）を確認し、必要なら `--max-steps` を減らしてください。

## ダウンロードしたデータの再利用

データは段階ごとにファイルとして残るため、2 回目以降は前の段階をやり直す必要がありません。

| 段階 | 保存先 | 再利用のされ方 |
|---|---|---|
| Wikipedia 原文 | `data/raw/wikipedia/20231101.{ja,en}/*.parquet` | このファイルがあれば、全スクリプトが自動的にローカルから読む。無ければ Hugging Face からストリーミングで取得する（その場合は毎回ダウンロードが発生する） |
| トークナイザ | `data/tokenizer.model` | `run_pipeline.ps1` は作成済みならスキップする |
| トークン列 | `data/train.bin` など | `prepare_data.py` は既存ファイルを上書きしない（作り直すときは `--overwrite`） |
| 学習の途中経過 | `out/ckpt.pt` | `train.py --resume` で続きから学習する。Ctrl+C で止めたときも保存される |

- `download_wiki.py` を再実行すると、取得済みのファイルはスキップします。中断しても続きから取得できます。
- 英語版のシャードを後から増やすには、`--en_shards 16` のように指定して再実行します。
- ローカルに一部のシャードしか無い場合は、そのシャードだけが使われます。データが目標トークン数に届かなかったときは、`prepare_data.py` が言語ごとの集計行の末尾に `(data exhausted)` と表示します。

## モデル構成 (`moe_model.py`)

| プリセット | 層 | d_model | エキスパート (top-k) | 1 エキスパートの d_ff | 総パラメータ | 活性パラメータ |
|---|---|---|---|---|---|---|
| tiny  | 2  | 128 | 4 (2) | 256  | 9M   | 9M   |
| small | 8  | 512 | 8 (2) | 1408 | 178M | 74M  |
| base  | 12 | 768 | 8 (2) | 2048 | 521M | 181M |

（パラメータ数は語彙 32000 の場合）

- 各層は RMSNorm → GQA Attention（RoPE）→ RMSNorm → MoE FFN の順に並びます。
- MoE FFN は softmax ルーターで Top-2 を選び、重みを再正規化してから SwiGLU エキスパートに渡します（Mixtral と同じ方式）。
- 学習時には、エキスパートの偏りを防ぐロードバランシング損失（Switch Transformer 型）も加えます。
- エキスパートの重みは `[n_expert, d_ff, d_model]` の 3 次元テンソルにまとめて保持し、そのまま GGUF の `blk.N.ffn_{gate,up,down}_exps.weight` として書き出します。

## 学習時間の目安 (RTX 4070 Ti SUPER 16GB)

- small プリセット（batch 8 × 累積 8 × 1024 トークン = 1 ステップ約 6.5 万トークン）の実測値は約 **51k トークン/秒**、VRAM 使用量は約 **10GB** です。
- 20000 ステップ（約 13 億トークン）で約 **8 時間** かかります。既定値の `run_pipeline.ps1` を通した実測値は次のとおりです。

  | 工程 | 所要時間 |
  |---|---|
  | [0] ダウンロード（Parquet 5.4GB） | 約 4 分（回線速度しだい、2 回目以降は 0 分） |
  | [1] 語彙データ（32,000 語） | 約 1.5 分 |
  | [2] 学習データ（10 億トークン） | 約 2.5 分 |
  | [3] 学習（20000 ステップ） | 7 時間 58 分 |
  | [4] GGUF 変換と検証 | 約 40 秒 |
- `--batch_size 16` にすると VRAM 16GB を超えて共有メモリにはみ出し、速度が約 1/4 に落ちます。大きくする場合は `--grad_accum` のほうを増やしてください。
- RTX 3060 12GB の場合は、`--batch_size 4 --grad_accum 16` を目安にしてください。

## 学習中に確認できること

- **20 ステップごと**: Loss、ロードバランス損失、学習率、処理速度、残り時間、エキスパート稼働率（全層の平均）
- **500 ステップごと**: 検証 Loss（日英混合・日本語のみ・英語のみ）と、層ごとのエキスパート稼働率（日本語と英語に分けて表示）
- **1000 ステップごと**: プロンプトからの生成サンプル
- ログは `out/train_log.jsonl` にも追記されます。

動作確認（tiny、300 ステップ）の時点でも、層 1 で「日本語は Expert 0 と 2、英語は Expert 1 と 3」という言語ごとの専門化が表れていました。

```
L1  ja [45.2  1.2 49.1  4.5] | en [ 9.1 43.3  3.5 44.1]
```

## GGUF の利用

```powershell
# llama.cpp (https://github.com/ggml-org/llama.cpp/releases の win-cuda / win-cpu 版)
llama-completion -m out\wiki-moe-ja-en-f16.gguf -p "日本の首都は" -n 64 -no-cnv

# さらに量子化する場合
llama-quantize out\wiki-moe-ja-en-f16.gguf out\wiki-moe-ja-en-q4_k_m.gguf Q4_K_M
```

- `export_gguf.py --outtype` には `f32`、`f16`、`bf16`、`q8_0` を指定できます。
- RMSNorm とルーターの重みは、精度を保つため常に F32 で書き出します。
- small プリセットの d_ff = 1408 は 256 の倍数ではありません。そのため Q4_K などの K-quant では、`ffn_down_exps` の部分だけ llama-quantize が別の量子化型に切り替えます。
- このモデルは事前学習だけを行ったベースモデルで、対話用の学習（SFT）はしていません。チャットではなく「文章の続きを書く」使い方になります。

### Ollama で使う

`out\Modelfile` を次の内容で作り、`ollama create` で登録します。本番の small は 1024 トークンの文脈で学習しているので、`num_ctx` は 1024 にします（実演用の tiny は 256）。

```
FROM ./wiki-moe-ja-en-f16.gguf
TEMPLATE "{{ .Prompt }}"
PARAMETER temperature 0
PARAMETER num_ctx 1024
PARAMETER num_predict 40
```

```powershell
cd out
ollama create wiki-moe -f Modelfile
ollama run wiki-moe "日本の首都は"
ollama rm wiki-moe    # 後片付け
```

- Modelfile は BOM なしの UTF-8 で保存してください。
- このモデルはベースモデルなので、TEMPLATE はプロンプトをそのまま渡す形にしています。

### 本番モデルの確認結果

20000 ステップ学習したモデル（検証 Loss が最良だった 19500 ステップのチェックポイント）を GGUF（f16、356MB）にして確認しました。生成は temperature 0（greedy）です。

| 入力 | 続き（先頭部分） |
|---|---|
| 日本の首都は | 、北海道札幌市東区にある首都。概要 首都は、札幌市東区にある首都の1つである。… |
| 富士山は | 、長野県長野市富士見町富士見にある富士山。標高は1,000m。概要 … |
| The capital of France is | the capital of the French colony of Saint-Denis. History … |
| Albert Einstein was | a German-born American film director, screenwriter, and film producer. Life and career … |

- **一致：** 4 つの入力すべてで、PyTorch・llama.cpp（b11756 win-cpu-x64）・Ollama 0.40.6 の出力が一致しました。
- **変換の誤差：** `verify_gguf.py` で GGUF から組み直したモデルとチェックポイントを比べました。logits の差は最大 0.008 で、次トークン予測の一致率は 100% でした。
- **出力の質：** 「概要」の見出しや経歴の書き方など、Wikipedia らしい文体は身に付いています。ただし内容の事実は正しくありません（1.8 億パラメータ・13 億トークンの規模での限界）。

### 回答精度のテスト

```powershell
.venv\Scripts\python.exe eval_facts.py --ckpt out\ckpt_best.pt --ckpt out\ckpt.pt --verbose --json out\eval_facts_baseline.json
```

`--json` を付けると、結果（集計値と問題ごとの結果）が `out\eval_facts_baseline.json` に保存されます。付けなければ画面に表示するだけです。`--ckpt` を省略すると `out\ckpt_best.pt` だけを評価します。

`eval_facts.jsonl` の 100 問（「日本の首都は → 東京」「The capital of France is → Paris」など、日英 50 問ずつ）について、プロンプトの直後に正解が続く確率を logits から計算します。生成はしないので、結果は毎回同じです。

| 指標 | 意味 |
|---|---|
| 正解率 | greedy（temperature 0）で正解をそのまま出す割合 |
| 上位10位 | 正解の 1 トークン目が、32,000 語中の上位 10 位以内に入る割合 |
| 順位中央値 | 正解の 1 トークン目の順位の中央値 |
| 正解の確率 | 正解全体に付けた確率の幾何平均 |

作者の環境（「動作環境」を参照）で本番モデルを評価した結果は次のとおりです。シードは固定していますが、GPU の計算は完全には決定的でないため、別の環境で学習したモデルでは値が多少異なることがあります。

| チェックポイント | 言語 | 正解率 | 上位10位 | 順位中央値 | 正解の確率 |
|---|---|---|---|---|---|
| ckpt_best.pt（19500） | 日本語 | 4% | 48% | 11 | 0.0023 |
| | 英語 | 2% | 36% | 16 | 0.0010 |
| | 全体 | 3% | 42% | 14 | 0.0015 |
| ckpt.pt（20000） | 全体 | 2% | 37% | 15 | 0.0012 |

## 当初の構想からの主な変更点

| 当初の構想 | 本実装 | 理由 |
|---|---|---|
| `cl-tohoku/wikipedia-ja`、旧 `wikipedia` スクリプト | `wikimedia/wikipedia` (20231101.ja / .en) | 日英とも同じ公式データセットで、Parquet をそのまま保存・再利用できるため |
| 語彙データを Gemma-2 から借りる（語彙 256k） | 語彙データ（`tokenizer.model`）を自前で学習（語彙 32k）。分割プログラムはどちらも SentencePiece | Gemma-2 は Gated Model のため利用申請が必要（※）。また語彙 256k だと、small 設定では単語の表（Embedding 層＋出力層）が 262M と全体の 64% を占め、出力層の計算結果だけで 1 バッチ 8.4GB の VRAM を使うため（語彙 32k なら 33M・18%・1.0GB） |
| uint32 のトークン列 | uint16 のトークン列 | 語彙数が 65536 未満なので、ファイルサイズを半分にできるため |
| Embedding → MoE → LM Head の 1 層のみ | Attention を含む多層トランスフォーマー | 文脈を扱えず、言語モデルとして成立しないため |
| GGUF アーキテクチャ `custom-moe` | `llama` ＋ `expert_count`（Mixtral 互換） | 既存の推論エンジン（llama.cpp）でそのまま動かせるようにするため |
| ハイパーパラメータを変換スクリプトに直接記述 | チェックポイントにモデル設定を同梱 | 学習時とモデル形状が食い違うのを防ぐため |

※ Gemma-2 は Hugging Face で「Gated Model（承認制モデル）」として公開されているため、利用申請（ライセンスへの同意）が必要です。モデルのページでライセンスに同意し、アクセストークンでログインしないとダウンロードできません。

### 当初の構想どおり Gemma-2 の語彙データを借りる場合（`borrow_tokenizer.py`）

借りるのは `tokenizer.model`（SentencePiece の語彙データ、約 4MB）の 1 ファイルだけです。モデルの重みは使わないので、蒸留ではありません。

1. https://huggingface.co/google/gemma-2-2b を開き、ライセンス（Gemma Terms of Use）に同意する
2. https://huggingface.co/settings/tokens で Read 権限のアクセストークンを作り、ログインする
   ```powershell
   .venv\Scripts\hf.exe auth login
   ```
3. 語彙データを取得し、続けて学習まで実行する（出力は `data_gemma\` と `out_gemma\`。自前の `data\`・`out\` は上書きしない）
   ```powershell
   .\run_pipeline.ps1 -GemmaVocab               # ダウンロード済みの Parquet はそのまま再利用
   # 1 ステップずつなら
   python borrow_tokenizer.py --out_dir data_gemma
   python prepare_data.py --out_dir data_gemma --tokenizer data_gemma\tokenizer.model
   python train.py --data_dir data_gemma --out_dir out_gemma --batch_size 2 --grad_accum 32
   ```

自前の語彙データとの違いは、各スクリプトが自動で吸収します。

| 項目 | 自前（32k） | Gemma-2（256k） | 対応 |
|---|---|---|---|
| トークン列の型 | uint16 | uint32（`train.bin` が約 2 倍） | `prepare_data.py` が語彙数で切り替え、`meta.json` の `dtype` を `train.py` が読む |
| 先頭の空白付与 | あり | なし | `export_gguf.py` が判定して `add_space_prefix` に書く |
| 特殊トークン ID | unk 0 / bos 1 / eos 2 | pad 0 / eos 1 / bos 2 / unk 3 | 全スクリプトが `sp.bos_id()` などで参照 |
| VRAM（small） | `--batch_size 8` | 出力層が 8 倍になるため `--batch_size 2 --grad_accum 32`（実効バッチは同じ 64） | `run_pipeline.ps1 -GemmaVocab` が自動指定 |

注意:
- 総パラメータは約 407M（単語の表 262M）に増えるので、学習時間・GGUF サイズとも増えます（未計測）。
- 語彙データは Gemma の配布物の一部です。作ったモデルや GGUF を公開する場合は、Gemma Terms of Use の条件（利用制限の引き継ぎなど）に従ってください。
- `--file` で、ダウンロード済みの `tokenizer.model` を指定することもできます。

## ファイル一覧

### 一括実行

| ファイル | 役割 |
|---|---|
| `run_pipeline.ps1` | 全工程の一括実行 |
| `demo.ps1` | 発表用の実演。7 ステップを Enter キーで 1 つずつ実行（`-Reset` で前回分を消去、`-NoPause` でリハーサル）。事前に `download_wiki.py` で Parquet を取得しておく必要があります |

### パイプライン（実行順）

| ファイル | 役割 |
|---|---|
| `download_wiki.py` | Parquet のダウンロード |
| `prepare_tokenizer.py` | SentencePiece の学習 |
| `borrow_tokenizer.py` | `prepare_tokenizer.py` の代わりに、Gemma-2 の語彙データを Hugging Face から取得（要ライセンス同意・ログイン） |
| `prepare_data.py` | 日英混合バイナリの作成 |
| `train.py` | 学習（監視表示・チェックポイント保存・再開に対応） |
| `export_gguf.py` | GGUF への変換 |

### 共通モジュール

| ファイル | 役割 |
|---|---|
| `moe_model.py` | MoE トランスフォーマー本体（`train.py`・`export_gguf.py`・`generate.py`・`verify_gguf.py`・`eval_facts.py` が使用） |
| `wiki_common.py` | Wikipedia 記事の読み出し（ローカル Parquet が無ければストリーミング取得）。`download_wiki.py`・`prepare_tokenizer.py`・`prepare_data.py` が使用 |

### 評価・検証

| ファイル | 役割 |
|---|---|
| `verify_gguf.py` | GGUF の検証 |
| `generate.py` | 文章生成とルーティングの可視化 |
| `eval_facts.py` / `eval_facts.jsonl` | 事実の穴埋めテスト（日英 100 問）で回答精度を測る。`--ckpt` を複数指定すると並べて比較、`--json` で結果を保存 |

### macOS 用（`macosx/`）

| ファイル | 役割 |
|---|---|
| `macosx/run_pipeline.sh` / `macosx/demo.sh` | `run_pipeline.ps1` / `demo.ps1` の bash 版 |
| `macosx/train.py` / `macosx/generate.py` / `macosx/eval_facts.py` | Apple GPU（MPS）を使う版 |
| `macosx/moe_model.py` | RoPE を実数演算で計算する版（上の 3 本が使用） |

### その他

| ファイル | 役割 |
|---|---|
| `requirements.txt` | PyTorch 以外の依存パッケージ |
| `.gitattributes` | `.sh` の改行コードを LF に固定 |
| `LICENSE` | MIT License |

## ライセンス

このリポジトリのコードは [MIT License](LICENSE) で公開しています。

次のものはこのライセンスの対象外です。それぞれの条件に従ってください。

- 学習データに使う Wikipedia の本文：[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/deed.ja) / GFDL
- `borrow_tokenizer.py` で取得する Gemma-2 の語彙データ：[Gemma Terms of Use](https://ai.google.dev/gemma/terms)
