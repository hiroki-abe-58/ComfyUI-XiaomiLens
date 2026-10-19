# Xiaomi-Lens-0 Segmentation for ComfyUI（日本語概要）

テキストで指定したものの**マスクとボックス**を返すノードです。公式の **Xiaomi-Lens-0**
（[SeerRay-Lab/Xiaomi-Lens-0](https://github.com/SeerRay-Lab/Xiaomi-Lens-0)、[論文](https://arxiv.org/abs/2610.23279)。Qwen3.5-4B に
マスクデコーダーを加えた 4B モデル、Apache-2.0）を ComfyUI の中で動かします。

- 「the pink dress」「coin」のような指定でも、「飲み物をかき混ぜるのに使うものはどれ？」のような質問でも、インスタンスごとに
  マスクとボックスを返します。マスクは入力画像と同じサイズなので、インペイントや切り抜きにそのまま使えます。
- ノードは同梱した公式コードを実行するだけで、実行中に何もダウンロードしません。

> **非公式**のノードパックです。Xiaomi、Xiaomi-Lens-0 の作者、Comfy Org とは関係ありません。英語の [README.md](README.md) が
> 正本です。

## ノード

| ノード | 役割 |
| --- | --- |
| **Xiaomi-Lens-0 Loader** | `models/xiaomi_lens/<名前>/` からモデルを読み込みます（GPU または CPU） |
| **Xiaomi-Lens-0 Segment** | `IMAGE` + テキスト → `mask`（全インスタンス）、`instance_masks`（1 つずつ）、`preview`、`text`（モデルの回答）、`boxes`（JSON） |

- `prompt_mode` = `find`: 入力した語句を README の定型文（`Locate all instances ... "<query>". Report bbox coordinates and
  masks.`）に入れます。`instruction`: 入力した文をそのまま送ります（質問・依頼向け。英語を推奨）。
- デコードは公式 `infer.py` と同じ greedy です。`threshold` 0.5 は公式の値で、0 にすると確率そのもの（ソフトマスク）になります。

## 検証の範囲

Windows 11、RTX 5090、ComfyUI v0.42.1（詳細は [docs/VERIFICATION.md](docs/VERIFICATION.md)）:

- 6 ケースで、回答・ボックス・マスクの値が公式コード（`infer.py` の各ステップ）と**バイト単位で一致**（6/6。Transformers 5.4.0 と
  5.20.1、マスクデコーダーの 2 種類のカーネルのいずれでも）。
- 実際の ComfyUI キューで、同梱ワークフロー 3 本の保存マスク・プレビュー・切り抜きのアルファが公式出力と**画素単位で一致**。
- CI（Ubuntu / Windows、ComfyUI v0.38.0 / v0.42.1）: 実際の構造とトークナイザーを持つ小さなチェックポイントで、公式コードとの
  一致、ワークフローの検証、実際のキューを確認。

## インストール

1. ComfyUI v0.38.0 以降、Python 3.10 以上、**Transformers 5.4.0 以上**。CUDA GPU を推奨します（実測の VRAM ピーク: 1 MP 以下で
   9.2 GB、8 MP で 15 GB）。
2. `custom_nodes` に clone して `requirements.txt` をインストールします。Transformers 4.x の環境では 5.x に上がるため、4.x が必要な
   他のノードが動かなくなる場合があります。
3. モデル（9.2 GB）をダウンロードします。リビジョン固定、全ファイルの SHA-256 を確認します:

   ```
   python custom_nodes/ComfyUI-XiaomiLens/tools/download_model.py --models-dir models/xiaomi_lens
   ```

4. ComfyUI を再起動し、`workflows/gui/` のワークフロー（マスク作成・切り抜き・質問によるマスク）を開きます。

## Loader の設定

- `decoder_attention`: `auto` は flash-attn があればそれ（公式の既定）、なければ **SDPA**。`mha` は flash-attn がないときの公式の
  フォールバックで、公式とバイト単位で同じ結果になりますが、メモリが画像サイズの 2 乗で増えます（8 MP では 30 GB 超で実際に
  メモリ不足）。SDPA と MHA は回答とボックスが同じで、マスクの IoU は 0.997 以上でした。
- `after_run`: `move to CPU` にすると実行後に約 9 GB の VRAM を空けます。

## ライセンス

- このノードパックのコード: MIT。
- 同梱の Xiaomi-Lens-0 コード: Apache-2.0（Copyright 2026 Xiaomi Corporation、1 行のみ変更。[NOTICE](NOTICE) 参照）。
- モデル（`tools/download_model.py` でのみ取得）: Apache-2.0（Qwen3.5-4B 由来、Apache-2.0）。
