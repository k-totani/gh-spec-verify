# gh-spec-verify

SPEC駆動開発のための検証ツール。仕様書（SPEC）と実際のコードが一致しているかをAIで検証します。

**GitHub CLI Extension** として使用できます。

## 特徴

- **GitHub CLI統合**: `gh spec-verify` コマンドで実行可能
- **言語非依存**: どのプログラミング言語のプロジェクトでも使用可能
- **AI検証**: Claude APIを使用して仕様書とコードの一致度を判定
- **CI対応**: JSON出力でCI/CDパイプラインに組み込み可能
- **柔軟な設定**: プロジェクトごとにカスタマイズ可能

## インストール

### GitHub CLI Extension（推奨）

```bash
gh extension install k-totani/gh-spec-verify
```

### Go Install

```bash
go install github.com/k-totani/gh-spec-verify/cmd/gh-spec-verify@latest
```

### ソースからビルド

```bash
git clone https://github.com/k-totani/gh-spec-verify.git
cd gh-spec-verify
go build -o gh-spec-verify ./cmd/gh-spec-verify
```

## クイックスタート

### 1. 初期設定

```bash
gh spec-verify init
```

これにより `.specverify.yml` が作成されます。

### 2. 環境変数を設定

```bash
export ANTHROPIC_API_KEY=your_api_key_here
```

### 3. SPECファイルを配置

```
specs/
├── ui/
│   └── login.md      # ログイン画面の仕様
└── api/
    └── auth.md       # 認証APIの仕様
```

### 4. 検証を実行

```bash
gh spec-verify check
```

## 使い方

### 全てのSPECを検証

```bash
gh spec-verify check
```

### 特定のタイプのみ検証

```bash
gh spec-verify check ui    # UIのSPECのみ
gh spec-verify check api   # APIのSPECのみ
```

### 複数タイプを同時検証

```bash
gh spec-verify check ui api domain   # 複数タイプを一度に検証
```

### グループ単位で検証

```bash
gh spec-verify check --group frontend   # フロントエンドグループを検証
gh spec-verify check -g backend         # バックエンドグループを検証
```

### 利用可能なタイプ/グループを確認

```bash
gh spec-verify types    # 定義されているSPECタイプ一覧
gh spec-verify groups   # 定義されているグループ一覧
```

### JSON出力（CI向け）

```bash
gh spec-verify check --format json
```

### 合格ラインを指定

```bash
gh spec-verify check --threshold 70
```

## 設定ファイル

`.specverify.yml`:

### 基本設定（シンプル）

```yaml
# SPECファイルのディレクトリ
specs_dir: specs/

# ソースコードのルートディレクトリ
code_dir: src/

# 使用するAIプロバイダー (claude, openai, gemini)
ai_provider: claude

# SPECタイプごとのコードディレクトリマッピング（シンプル形式）
mapping:
  ui: client/components
  api: server/routes

# 検証オプション
options:
  # 並列実行数
  concurrency: 3
  # 合格ライン（%）
  pass_threshold: 50
  # 詳細出力
  verbose: false
```

### 詳細設定（spec_typesとgroups）

```yaml
specs_dir: specs/
code_dir: src/
ai_provider: claude

# SPECタイプの詳細定義
spec_types:
  ui:
    # 検証対象のコードパス（複数指定可能）
    code_paths:
      - client/components
      - client/pages
    # AI検証時の重点観点
    verification_focus:
      - コンポーネント構成
      - 画面遷移
      - 状態管理

  api:
    code_paths:
      - server/routes
      - server/handlers
    verification_focus:
      - エンドポイント定義
      - リクエスト/レスポンス形式
      - 認証・認可

  domain:
    code_paths:
      - server/domain
      - server/models
    verification_focus:
      - ビジネスルール
      - ドメインロジック
      - バリデーション

  service:
    code_paths:
      - server/services
    verification_focus:
      - ユースケース実装
      - トランザクション処理

# グループ定義
groups:
  frontend:
    types: [ui]
    description: "フロントエンド関連"

  backend:
    types: [api, domain, service]
    description: "バックエンド関連"

  all:
    types: [ui, api, domain, service]
    description: "全てのSPEC"

options:
  concurrency: 3
  pass_threshold: 50
```

**Note**: `spec_types` と `mapping` の両方が定義されている場合、`spec_types` が優先されます。これにより既存の設定を段階的に移行できます。

## SPECファイルの書き方

SPECファイルはMarkdown形式で記述します。

### 基本構造

```markdown
# ページタイトル

## 基本情報

| 項目 | 内容 |
|------|------|
| パス | `/path/to/page` |
| 必要権限 | `PERMISSION_NAME` |

## 概要

このページの概要を記述します。

## 画面構成

### セクション名

- 要素1
- 要素2

## 処理フロー

1. ステップ1
2. ステップ2

## バリデーション

| 項目 | ルール |
|------|--------|
| フィールド名 | バリデーションルール |

## エラーケース

| ケース | 表示 |
|--------|------|
| エラー名 | エラーメッセージ |
```

### 関連ファイルの指定

SPECファイル内で関連するコードファイルを指定できます：

```markdown
## 関連コンポーネント

| コンポーネント | ファイル |
|----------------|----------|
| LoginForm | `~/components/LoginForm` |
```

## 環境変数

| 変数名 | 説明 |
|--------|------|
| `ANTHROPIC_API_KEY` | Claude APIキー |
| `OPENAI_API_KEY` | OpenAI APIキー |
| `GOOGLE_API_KEY` | Gemini APIキー |
| `SPEC_VERIFY_API_KEY` | 汎用APIキー（プロバイダー設定に依存） |

## CI/CD連携

### GitHub Actions

```yaml
name: SPEC Verification

on: [push, pull_request]

jobs:
  verify:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Go
        uses: actions/setup-go@v5
        with:
          go-version: '1.21'

      - name: Install gh-spec-verify
        run: go install github.com/k-totani/gh-spec-verify/cmd/gh-spec-verify@latest

      - name: Run verification
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
        run: gh-spec-verify check --format json > spec-verify-result.json

      - name: Upload results
        uses: actions/upload-artifact@v4
        with:
          name: spec-verify-result
          path: spec-verify-result.json
```

## 出力例

```
🔍 SPEC検証を開始します...

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📄 login.md
   タイトル: ログイン画面
   パス: /login
   関連コード: 3ファイル
   ✅ 一致度: 85%
   ✓ 一致:
     - ユーザー名入力フィールド
     - パスワード入力フィールド
     - ログインボタン
   ✗ 不一致:
     - パスワードリセットリンク

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 サマリー

   総SPEC数: 1
   平均一致度: 85.0%
   高一致(≥80%): 1件
   低一致(<50%): 0件

   詳細:
   ████████░░  85% login.md
```

## ライセンス

MIT License
