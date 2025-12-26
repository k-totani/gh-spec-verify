package config

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

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

	// SPECタイプごとのコードディレクトリマッピング（後方互換用）
	Mapping map[string]string `yaml:"mapping,omitempty"`

	// 詳細なSPECタイプ定義（新機能）
	SpecTypes map[string]SpecType `yaml:"spec_types,omitempty"`

	// グループ定義（新機能）
	Groups map[string]Group `yaml:"groups,omitempty"`

	// APIソース定義（エンドポイント抽出用）- 後方互換
	APISources []RouteSource `yaml:"api_sources,omitempty"`

	// ルートソース定義（ページ/API両方対応）
	RouteSources []RouteSource `yaml:"route_sources,omitempty"`

	// 検証時のオプション
	Options VerifyOptions `yaml:"options"`
}

// RouteSource はルート（API/ページ）のソース定義
type RouteSource struct {
	// タイプ: express, fastify, openapi, graphql, go-echo, go-gin, rails, django, auto
	Type string `yaml:"type"`

	// ファイルパターン（glob形式）
	Patterns []string `yaml:"patterns"`

	// カテゴリ: ui（ページルート）, api（APIエンドポイント）
	// 省略時はapiと判定、パターンに基づいて自動判定も行う
	Category string `yaml:"category,omitempty"`

	// オプション設定
	Options map[string]string `yaml:"options,omitempty"`
}

// APISource は後方互換のためのエイリアス
type APISource = RouteSource

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

// SpecType はSPECタイプの詳細定義
type SpecType struct {
	// コードパス（複数指定可能）
	CodePaths []string `yaml:"code_paths"`

	// 検証観点（AIへのヒント）
	VerificationFocus []string `yaml:"verification_focus,omitempty"`

	// ファイルパターン（オプション、glob形式）
	FilePatterns []string `yaml:"file_patterns,omitempty"`

	// 除外パターン（オプション）
	ExcludePatterns []string `yaml:"exclude_patterns,omitempty"`
}

// Group はSPECタイプのグループ
type Group struct {
	// 含まれるタイプ
	Types []string `yaml:"types"`

	// 説明
	Description string `yaml:"description,omitempty"`
}

// DefaultConfig はデフォルト設定を返す
func DefaultConfig() *Config {
	return &Config{
		SpecsDir:   "specs/",
		CodeDir:    "src/",
		AIProvider: "gemini",
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
// オプションでAPIキーを直接渡すことも可能（CLI引数用）
func Load(path string, opts ...LoadOption) (*Config, error) {
	// .envファイルを先に読み込む（環境変数より低優先）
	_ = LoadEnvFile()

	// デフォルト設定から開始
	cfg := DefaultConfig()

	// ファイルが存在しない場合はデフォルトを返す
	if _, err := os.Stat(path); os.IsNotExist(err) {
		// 環境変数からAPIキーを取得
		cfg.AIAPIKey = GetAPIKeyFromEnv(cfg.AIProvider)
		applyLoadOptions(cfg, opts)
		return cfg, nil
	}

	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read config file: %w", err)
	}

	if err := yaml.Unmarshal(data, cfg); err != nil {
		return nil, fmt.Errorf("failed to parse config file: %w", err)
	}

	// オプションを適用（CLI引数が最優先）
	// 注意: providerの変更を先に適用してから、APIキーを取得する
	applyLoadOptions(cfg, opts)

	// 環境変数からAPIキーを取得（設定ファイルより優先、ただしCLI引数は除く）
	// CLI引数でAPIキーが指定されていない場合のみ、環境変数から取得
	if !hasAPIKeyOption(opts) {
		if envKey := GetAPIKeyFromEnv(cfg.AIProvider); envKey != "" {
			cfg.AIAPIKey = envKey
		}
	}

	return cfg, nil
}

// hasAPIKeyOption はLoadOptionにAPIキー指定が含まれるか確認
func hasAPIKeyOption(opts []LoadOption) bool {
	// LoadOptionは関数なので直接判定できない
	// 代わりに、applyLoadOptions後にAPIKeyが設定されているかで判断
	// この関数は空のConfigでオプションを適用して確認する
	testCfg := &Config{}
	for _, opt := range opts {
		opt(testCfg)
	}
	return testCfg.AIAPIKey != ""
}

// LoadOption は設定読み込み時のオプション
type LoadOption func(*Config)

// WithAPIKey はAPIキーを直接指定するオプション
func WithAPIKey(key string) LoadOption {
	return func(cfg *Config) {
		if key != "" {
			cfg.AIAPIKey = key
		}
	}
}

// WithProvider はAIプロバイダーを指定するオプション
func WithProvider(provider string) LoadOption {
	return func(cfg *Config) {
		if provider != "" {
			cfg.AIProvider = provider
		}
	}
}

func applyLoadOptions(cfg *Config, opts []LoadOption) {
	for _, opt := range opts {
		opt(cfg)
	}
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

// GetCodePath はSPECタイプに対応するコードパスを返す（後方互換用）
func (c *Config) GetCodePath(specType string) string {
	// 新しいspec_typesを優先
	if st, ok := c.SpecTypes[specType]; ok && len(st.CodePaths) > 0 {
		return filepath.Join(c.CodeDir, st.CodePaths[0])
	}
	// 従来のmappingにフォールバック
	if mapped, ok := c.Mapping[specType]; ok {
		return filepath.Join(c.CodeDir, mapped)
	}
	return c.CodeDir
}

// GetCodePaths はSPECタイプに対応する全てのコードパスを返す
func (c *Config) GetCodePaths(specType string) []string {
	// 新しいspec_typesを優先
	if st, ok := c.SpecTypes[specType]; ok && len(st.CodePaths) > 0 {
		paths := make([]string, len(st.CodePaths))
		for i, p := range st.CodePaths {
			paths[i] = filepath.Join(c.CodeDir, p)
		}
		return paths
	}
	// 従来のmappingにフォールバック
	if mapped, ok := c.Mapping[specType]; ok {
		return []string{filepath.Join(c.CodeDir, mapped)}
	}
	return []string{c.CodeDir}
}

// GetVerificationFocus はSPECタイプの検証観点を返す
func (c *Config) GetVerificationFocus(specType string) []string {
	if st, ok := c.SpecTypes[specType]; ok {
		return st.VerificationFocus
	}
	return nil
}

// GetTypesByGroup はグループに含まれるSPECタイプを返す
func (c *Config) GetTypesByGroup(groupName string) []string {
	if group, ok := c.Groups[groupName]; ok {
		return group.Types
	}
	return nil
}

// GetAllSpecTypes は定義されている全てのSPECタイプ名を返す
func (c *Config) GetAllSpecTypes() []string {
	typeSet := make(map[string]bool)

	// spec_typesから取得
	for typeName := range c.SpecTypes {
		typeSet[typeName] = true
	}

	// mappingからも取得（後方互換）
	for typeName := range c.Mapping {
		typeSet[typeName] = true
	}

	types := make([]string, 0, len(typeSet))
	for typeName := range typeSet {
		types = append(types, typeName)
	}
	return types
}

// GetAllGroups は定義されている全てのグループ名を返す
func (c *Config) GetAllGroups() []string {
	groups := make([]string, 0, len(c.Groups))
	for groupName := range c.Groups {
		groups = append(groups, groupName)
	}
	return groups
}

// HasSpecType は指定されたSPECタイプが定義されているか確認する
func (c *Config) HasSpecType(specType string) bool {
	if _, ok := c.SpecTypes[specType]; ok {
		return true
	}
	if _, ok := c.Mapping[specType]; ok {
		return true
	}
	return false
}

// HasGroup は指定されたグループが定義されているか確認する
func (c *Config) HasGroup(groupName string) bool {
	_, ok := c.Groups[groupName]
	return ok
}

// GetSpecTypeInfo はSPECタイプの詳細情報を返す（存在しない場合はnil）
func (c *Config) GetSpecTypeInfo(specType string) *SpecType {
	if st, ok := c.SpecTypes[specType]; ok {
		return &st
	}
	// mappingから疑似的なSpecTypeを生成
	if mapped, ok := c.Mapping[specType]; ok {
		return &SpecType{
			CodePaths: []string{mapped},
		}
	}
	return nil
}

// GetAllRouteSources はapi_sourcesとroute_sourcesを統合して返す
// カテゴリが未設定の場合は自動判定する
func (c *Config) GetAllRouteSources() []RouteSource {
	var sources []RouteSource

	// route_sources を優先
	for _, src := range c.RouteSources {
		s := src
		if s.Category == "" {
			s.Category = inferCategory(s.Patterns)
		}
		sources = append(sources, s)
	}

	// api_sources も追加（後方互換）
	for _, src := range c.APISources {
		s := src
		if s.Category == "" {
			s.Category = inferCategory(s.Patterns)
		}
		sources = append(sources, s)
	}

	return sources
}

// inferCategory はパターンからカテゴリを推測する
func inferCategory(patterns []string) string {
	for _, p := range patterns {
		// UIパターンの検出
		if containsAny(p, []string{
			"routes", "pages", "views", "screens",
			".tsx", ".jsx", ".vue", ".svelte",
			"client", "frontend", "app/routes",
		}) && !containsAny(p, []string{"api", "server"}) {
			return "ui"
		}
		// APIパターンの検出
		if containsAny(p, []string{
			"api", "server", "backend",
			"openapi", "swagger",
		}) {
			return "api"
		}
	}
	// デフォルトはapi
	return "api"
}

// containsAny は文字列が指定されたキーワードのいずれかを含むか（大文字小文字区別なし）
func containsAny(s string, keywords []string) bool {
	sLower := strings.ToLower(s)
	for _, kw := range keywords {
		if strings.Contains(sLower, strings.ToLower(kw)) {
			return true
		}
	}
	return false
}
