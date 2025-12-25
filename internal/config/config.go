package config

import (
	"fmt"
	"os"
	"path/filepath"

	"gopkg.in/yaml.v3"
)

// Config は spec-verify の設定を表す
type Config struct {
	// SPECファイルが格納されているディレクトリ
	SpecsDir string `yaml:"specs_dir"`

	// ソースコードのルートディレクトリ
	CodeDir string `yaml:"code_dir"`

	// 使用するAIプロバイダー (claude, openai, gemini)
	AIProvider string `yaml:"ai_provider"`

	// AIプロバイダーのAPIキー（環境変数から取得することを推奨）
	AIAPIKey string `yaml:"ai_api_key,omitempty"`

	// SPECタイプごとのコードディレクトリマッピング
	Mapping map[string]string `yaml:"mapping"`

	// APIソース定義（エンドポイント抽出用）
	APISources []APISource `yaml:"api_sources,omitempty"`

	// 検証時のオプション
	Options VerifyOptions `yaml:"options"`
}

// APISource はAPIエンドポイントのソース定義
type APISource struct {
	// タイプ: express, fastify, openapi, graphql, go-echo, go-gin, rails, django, auto
	Type string `yaml:"type"`

	// ファイルパターン（glob形式）
	Patterns []string `yaml:"patterns"`

	// オプション設定
	Options map[string]string `yaml:"options,omitempty"`
}

// VerifyOptions は検証時のオプション
type VerifyOptions struct {
	// 並列実行数
	Concurrency int `yaml:"concurrency"`

	// 最低合格ライン（パーセント）- 全体平均
	PassThreshold int `yaml:"pass_threshold"`

	// 個別閾値（パーセント）- この値未満のSPECがあれば失敗
	// 0の場合は無効
	FailUnder int `yaml:"fail_under"`

	// 詳細出力を有効にする
	Verbose bool `yaml:"verbose"`
}

// DefaultConfig はデフォルト設定を返す
func DefaultConfig() *Config {
	return &Config{
		SpecsDir:   "specs/",
		CodeDir:    "src/",
		AIProvider: "claude",
		Mapping: map[string]string{
			"ui":  "client/components",
			"api": "server/routes",
		},
		Options: VerifyOptions{
			Concurrency:   3,
			PassThreshold: 50,
			FailUnder:     0, // 0は無効
			Verbose:       false,
		},
	}
}

// Load は設定ファイルを読み込む
func Load(path string) (*Config, error) {
	// デフォルト設定から開始
	cfg := DefaultConfig()

	// ファイルが存在しない場合はデフォルトを返す
	if _, err := os.Stat(path); os.IsNotExist(err) {
		return cfg, nil
	}

	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read config file: %w", err)
	}

	if err := yaml.Unmarshal(data, cfg); err != nil {
		return nil, fmt.Errorf("failed to parse config file: %w", err)
	}

	// 環境変数からAPIキーを取得（設定ファイルより優先）
	if envKey := os.Getenv("SPEC_VERIFY_API_KEY"); envKey != "" {
		cfg.AIAPIKey = envKey
	}

	// プロバイダー固有の環境変数もチェック
	if cfg.AIAPIKey == "" {
		switch cfg.AIProvider {
		case "claude":
			cfg.AIAPIKey = os.Getenv("ANTHROPIC_API_KEY")
		case "openai":
			cfg.AIAPIKey = os.Getenv("OPENAI_API_KEY")
		case "gemini":
			cfg.AIAPIKey = os.Getenv("GOOGLE_API_KEY")
		}
	}

	return cfg, nil
}

// Save は設定ファイルを保存する
func (c *Config) Save(path string) error {
	// APIキーは保存しない（セキュリティのため）
	configToSave := *c
	configToSave.AIAPIKey = ""

	data, err := yaml.Marshal(&configToSave)
	if err != nil {
		return fmt.Errorf("failed to marshal config: %w", err)
	}

	if err := os.WriteFile(path, data, 0644); err != nil {
		return fmt.Errorf("failed to write config file: %w", err)
	}

	return nil
}

// FindConfigFile は設定ファイルを探す
func FindConfigFile() string {
	candidates := []string{
		".specverify.yml",
		".specverify.yaml",
		"specverify.yml",
		"specverify.yaml",
	}

	for _, candidate := range candidates {
		if _, err := os.Stat(candidate); err == nil {
			return candidate
		}
	}

	return ".specverify.yml"
}

// GetCodePath はSPECタイプに対応するコードパスを返す
func (c *Config) GetCodePath(specType string) string {
	if mapped, ok := c.Mapping[specType]; ok {
		return filepath.Join(c.CodeDir, mapped)
	}
	return c.CodeDir
}
